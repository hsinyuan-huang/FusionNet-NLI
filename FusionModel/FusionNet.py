import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers

class FusionNet(nn.Module):
    """Network for the FusionNet Module."""
    def __init__(self, opt, embedding=None, padding_idx=0):
        super(FusionNet, self).__init__()

        # Input size to RNN: word emb + char emb + question emb + manual features
        input_size = 0

        layers.set_my_dropout_prob(opt['my_dropout_p'])
        layers.set_seq_dropout(opt['do_seq_dropout'])

        if opt['use_wemb']:
            # Word embeddings
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=padding_idx)
            if embedding is not None:
                self.embedding.weight.data = embedding
                if opt['fix_embeddings'] or opt['tune_partial'] == 0:
                    opt['fix_embeddings'] = True
                    opt['tune_partial'] = 0
                    for p in self.embedding.parameters():
                        p.requires_grad = False
                else:
                    assert opt['tune_partial'] < embedding.size(0)
                    fixed_embedding = embedding[opt['tune_partial']:]
                    # a persistent buffer for the nn.Module
                    self.register_buffer('fixed_embedding', fixed_embedding)
                    self.fixed_embedding = fixed_embedding
            embedding_dim = opt['embedding_dim']
            input_size += embedding_dim
        else:
            opt['fix_embeddings'] = True
            opt['tune_partial'] = 0
        if opt['use_cove']:
            self.CoVe = layers.MTLSTM(opt, embedding)
            input_size += self.CoVe.output_size
        if opt['use_pos']:
            self.pos_embedding = nn.Embedding(opt['pos_size'], opt['pos_dim'])
            input_size += opt['pos_dim']
        if opt['use_ner']:
            self.ner_embedding = nn.Embedding(opt['ner_size'], opt['ner_dim'])
            input_size += opt['ner_dim']
        if opt['full_att_type'] == 2:
            aux_input = opt['num_features']
        else:
            aux_input = 1

        # Setup the vector size for [premise, hypothesis]
        # they will be modified in the following code
        cur_hidden_size = input_size
        print('Initially, the vector_size is {} (+ {})'.format(cur_hidden_size, aux_input))

        # RNN premise encoder
        self.P_rnn = layers.RNNEncoder(cur_hidden_size, opt['hidden_size'], opt['enc_rnn_layers'], aux_size = aux_input)
        # RNN hypothesis encoder
        self.H_rnn = layers.RNNEncoder(cur_hidden_size, opt['hidden_size'], opt['enc_rnn_layers'], aux_size = aux_input)
        cur_hidden_size = opt['hidden_size'] * 2

        # Output sizes of rnn encoders
        print('After Input LSTM, the vector_size is [', cur_hidden_size, '] *', opt['enc_rnn_layers'])

        # Multi-level Fusion
        if opt['full_att_type'] == 0:
            self.full_attn_P = layers.FullAttention(cur_hidden_size, cur_hidden_size, 1)
            self.full_attn_H = layers.FullAttention(cur_hidden_size, cur_hidden_size, 1)
        elif opt['full_att_type'] == 1:
            self.full_attn_P = layers.FullAttention(input_size + opt['enc_rnn_layers'] * cur_hidden_size, cur_hidden_size, 1)
            self.full_attn_H = layers.FullAttention(input_size + opt['enc_rnn_layers'] * cur_hidden_size, cur_hidden_size, 1)
        elif opt['full_att_type'] == 2:
            self.full_attn_P = layers.FullAttention(input_size + opt['enc_rnn_layers'] * cur_hidden_size,
                                                    opt['enc_rnn_layers'] * cur_hidden_size, opt['enc_rnn_layers'])
            self.full_attn_H = layers.FullAttention(input_size + opt['enc_rnn_layers'] * cur_hidden_size,
                                                    opt['enc_rnn_layers'] * cur_hidden_size, opt['enc_rnn_layers'])
        else:
            raise NotImplementedError('full_att_type = %s' % opt['full_att_type'])
        cur_hidden_size = self.full_attn_P.output_size * 2

        # RNN premise inference
        self.P_infer_rnn = layers.RNNEncoder(cur_hidden_size, opt['hidden_size'], opt['inf_rnn_layers'])
        # RNN hypothesis inference
        self.H_infer_rnn = layers.RNNEncoder(cur_hidden_size, opt['hidden_size'], opt['inf_rnn_layers'])
        cur_hidden_size = opt['hidden_size'] * 2 * opt['inf_rnn_layers']

        print('Before answer finding, hidden size is', cur_hidden_size)

        # Question merging
        if opt['final_merge'] == 'linear_self_attn':
            self.self_attn_P = layers.LinearSelfAttn(cur_hidden_size)
            self.self_attn_H = layers.LinearSelfAttn(cur_hidden_size)
        elif opt['final_merge'] != 'avg':
            raise NotImplementedError('final_merge = %s' % opt['final_merge'])

        self.classifier = layers.MLPFunc(cur_hidden_size * 4, cur_hidden_size, opt['number_of_class'])

        # Store config
        self.opt = opt

    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_f, x2_pos, x2_ner, x2_mask):
        """Inputs:
        x1 = premise word indices                [batch * len_1]
        x1_f = premise word features indices     [batch * len_1 * nfeat]
        x1_pos = premise POS tags                [batch * len_1]
        x1_ner = premise entity tags             [batch * len_1]
        x1_mask = premise padding mask           [batch * len_1]
        x2 = hypothesis word indices             [batch * len_2]
        x2_f = hypothesis word features indices  [batch * len_2 * nfeat]
        x2_pos = hypothesis POS tags             [batch * len_2]
        x2_ner = hypothesis entity tags          [batch * len_2]
        x2_mask = hypothesis padding mask        [batch * len_2]
        """
        # Prepare premise and hypothesis input
        Prnn_input_list = []
        Hrnn_input_list = []
        if self.opt['use_wemb']:
            # Word embedding for both premise and hypothesis
            emb = self.embedding if self.training else self.eval_embed
            x1_emb, x2_emb = emb(x1), emb(x2)
            # Dropout on embeddings
            if self.opt['dropout_emb'] > 0:
                x1_emb = layers.dropout(x1_emb, p=self.opt['dropout_emb'], training=self.training)
                x2_emb = layers.dropout(x2_emb, p=self.opt['dropout_emb'], training=self.training)
            Prnn_input_list.append(x1_emb)
            Hrnn_input_list.append(x2_emb)
        if self.opt['use_cove']:
            _, x1_cove = self.CoVe(x1, x1_mask)
            _, x2_cove = self.CoVe(x2, x2_mask)
            if self.opt['dropout_emb'] > 0:
                x1_cove = layers.dropout(x1_cove, p=self.opt['dropout_emb'], training=self.training)
                x2_cove = layers.dropout(x2_cove, p=self.opt['dropout_emb'], training=self.training)
            Prnn_input_list.append(x1_cove)
            Hrnn_input_list.append(x2_cove)
        if self.opt['use_pos']:
            x1_pos_emb = self.pos_embedding(x1_pos)
            x2_pos_emb = self.pos_embedding(x2_pos)
            Prnn_input_list.append(x1_pos_emb)
            Hrnn_input_list.append(x2_pos_emb)
        if self.opt['use_ner']:
            x1_ner_emb = self.ner_embedding(x1_ner)
            x2_ner_emb = self.ner_embedding(x2_ner)
            Prnn_input_list.append(x1_ner_emb)
            Hrnn_input_list.append(x2_ner_emb)
        x1_input = torch.cat(Prnn_input_list, 2)
        x2_input = torch.cat(Hrnn_input_list, 2)

        # Now the features are ready
        # x1_input: [batch_size, doc_len, input_size]
        # x2_input: [batch_size, doc_len, input_size]

        if self.opt['full_att_type'] == 2:
            x1_f = layers.dropout(x1_f, p=self.opt['dropout_EM'], training=self.training)
            x2_f = layers.dropout(x2_f, p=self.opt['dropout_EM'], training=self.training)
            Paux_input, Haux_input = x1_f, x2_f
        else:
            Paux_input = x1_f[:, :, 0].contiguous().view(x1_f.size(0), x1_f.size(1), 1)
            Haux_input = x2_f[:, :, 0].contiguous().view(x2_f.size(0), x2_f.size(1), 1)

        # Encode premise with RNN
        P_abstr_ls = self.P_rnn(x1_input, x1_mask, aux_input=Paux_input)
        # Encode hypothesis with RNN
        H_abstr_ls = self.H_rnn(x2_input, x2_mask, aux_input=Haux_input)

        # Fusion
        if self.opt['full_att_type'] == 0:
            P_atts = P_abstr_ls[-1].contiguous()
            H_atts = H_abstr_ls[-1].contiguous()
            P_xs = P_abstr_ls[-1].contiguous()
            H_xs = H_abstr_ls[-1].contiguous()
        elif self.opt['full_att_type'] == 1:
            P_atts = torch.cat([x1_input] + P_abstr_ls, 2)
            H_atts = torch.cat([x2_input] + H_abstr_ls, 2)
            P_xs = P_abstr_ls[-1].contiguous()
            H_xs = H_abstr_ls[-1].contiguous()
        elif self.opt['full_att_type'] == 2:
            P_atts = torch.cat([x1_input] + P_abstr_ls, 2)
            H_atts = torch.cat([x2_input] + H_abstr_ls, 2)
            P_xs = torch.cat(P_abstr_ls, 2)
            H_xs = torch.cat(H_abstr_ls, 2)
        aP_xs = self.full_attn_P(P_atts, H_atts, P_xs, H_xs, x2_mask)
        aH_xs = self.full_attn_H(H_atts, P_atts, H_xs, P_xs, x1_mask)
        P_hiddens = torch.cat([P_xs, aP_xs], 2)
        H_hiddens = torch.cat([H_xs, aH_xs], 2)

        # Inference on premise and hypothesis
        P_hiddens = torch.cat(self.P_infer_rnn(P_hiddens, x1_mask), 2)
        H_hiddens = torch.cat(self.H_infer_rnn(H_hiddens, x2_mask), 2)

        # Merge hiddens for answer classification
        if self.opt['final_merge'] == 'avg':
            P_merge_weights = layers.uniform_weights(P_hiddens, x1_mask)
            H_merge_weights = layers.uniform_weights(H_hiddens, x2_mask)
        elif self.opt['final_merge'] == 'linear_self_attn':
            P_merge_weights = self.self_attn_P(P_hiddens, x1_mask)
            H_merge_weights = self.self_attn_H(H_hiddens, x2_mask)
        P_avg_hidden = layers.weighted_avg(P_hiddens, P_merge_weights)
        H_avg_hidden = layers.weighted_avg(H_hiddens, H_merge_weights)
        P_max_hidden = torch.max(P_hiddens, 1)[0]
        H_max_hidden = torch.max(H_hiddens, 1)[0]

        # Predict scores for different classes
        scores = self.classifier(torch.cat([P_avg_hidden, H_avg_hidden, P_max_hidden, H_max_hidden], 1))

        return scores # -inf to inf
