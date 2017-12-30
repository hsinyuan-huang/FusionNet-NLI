import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

# ------------------------------------------------------------------------------
# Neural Modules
# ------------------------------------------------------------------------------

def set_seq_dropout(option): # option = True or False
    global do_seq_dropout
    do_seq_dropout = option

def set_my_dropout_prob(p): # p between 0 to 1
    global my_dropout_p
    my_dropout_p = p

def seq_dropout(x, p=0, training=False):
    """
    x: batch * len * input_size
    """
    if training == False or p == 0:
        return x
    dropout_mask = Variable(1.0 / (1-p) * torch.bernoulli((1-p) * (x.data.new(x.size(0), x.size(2)).zero_() + 1)), requires_grad=False)
    return dropout_mask.unsqueeze(1).expand_as(x) * x

def dropout(x, p=0, training=False):
    """
    x: (batch * len * input_size) or (any other shape)
    """
    if do_seq_dropout and len(x.size()) == 3: # if x is (batch * len * input_size)
        return seq_dropout(x, p=p, training=training)
    else:
        return F.dropout(x, p=p, training=training)

class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, rnn_type=nn.LSTM, aux_size=0):
        super(RNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size_ = (input_size + 2 * hidden_size * i)
            if i == 0: input_size_ += aux_size
            self.rnns.append(rnn_type(input_size_, hidden_size, num_layers=1, bidirectional=True))

    def forward(self, x, x_mask, aux_input=None):
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)
        if aux_input is not None:
            aux_input = aux_input.transpose(0, 1)

        # Encode all layers
        hiddens = [x]
        for i in range(self.num_layers):
            rnn_input = torch.cat(hiddens, 2)
            if i == 0 and aux_input is not None:
                rnn_input = torch.cat([rnn_input, aux_input], 2)

            # Apply dropout to input
            if my_dropout_p > 0:
                rnn_input = dropout(rnn_input, p=my_dropout_p, training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            hiddens.append(rnn_output)

        # Transpose back
        hiddens = [h.transpose(0, 1) for h in hiddens]
        return hiddens[1:]

class MTLSTM(nn.Module):
    def __init__(self, opt, embedding=None, padding_idx=0):
        """Initialize an MTLSTM

        Arguments:
            embedding (Float Tensor): If not None, initialize embedding matrix with specified embedding vectors
        """
        super(MTLSTM, self).__init__()

        self.embedding = nn.Embedding(opt['vocab_size'], opt['embedding_dim'], padding_idx=padding_idx)
        if embedding is not None:
            self.embedding.weight.data = embedding

        state_dict = torch.load(opt['MTLSTM_path'])
        self.rnn1 = nn.LSTM(300, 300, num_layers=1, bidirectional=True)
        self.rnn2 = nn.LSTM(600, 300, num_layers=1, bidirectional=True)

        state_dict1 = dict([(name, param.data) if isinstance(param, Parameter) else (name, param)
                        for name, param in state_dict.items() if '0' in name])
        state_dict2 = dict([(name.replace('1', '0'), param.data) if isinstance(param, Parameter) else (name.replace('1', '0'), param)
                        for name, param in state_dict.items() if '1' in name])
        self.rnn1.load_state_dict(state_dict1)
        self.rnn2.load_state_dict(state_dict2)

        for p in self.embedding.parameters():
            p.requires_grad = False
        for p in self.rnn1.parameters():
            p.requires_grad = False
        for p in self.rnn2.parameters():
            p.requires_grad = False

        self.output_size = 600

    def setup_eval_embed(self, eval_embed, padding_idx=0):
        """Allow evaluation vocabulary size to be greater than training vocabulary size

        Arguments:
            eval_embed (Float Tensor): Initialize eval_embed to be the specified embedding vectors
        """
        self.eval_embed = nn.Embedding(eval_embed.size(0), eval_embed.size(1), padding_idx = padding_idx)
        self.eval_embed.weight.data = eval_embed

        for p in self.eval_embed.parameters():
            p.requires_grad = False

    def forward(self, x_idx, x_mask):
        """A pretrained MT-LSTM (McCann et. al. 2017).
        This LSTM was trained with 300d 840B GloVe on the WMT 2017 machine translation dataset.

        Arguments:
            x_idx (Long Tensor): a Long Tensor of size (batch * len).
            x_mask (Byte Tensor): a Byte Tensor of mask for the input tensor (batch * len).
        """
        emb = self.embedding if self.training else self.eval_embed
        x_hiddens = emb(x_idx)

        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        lens, indices = torch.sort(lengths, 0, True)

        output1, _ = self.rnn1(pack(x_hiddens[indices], lens.tolist(), batch_first=True))
        output2, _ = self.rnn2(output1)

        output1 = unpack(output1, batch_first=True)[0]
        output2 = unpack(output2, batch_first=True)[0]

        _, _indices = torch.sort(indices, 0)
        output1 = output1[_indices]
        output2 = output2[_indices]

        return output1, output2

# Attention layer
class FullAttention(nn.Module):
    def __init__(self, full_size, hidden_size, num_level):
        super(FullAttention, self).__init__()
        assert(hidden_size % num_level == 0)
        self.full_size = full_size
        self.hidden_size = hidden_size
        self.attsize_per_lvl = hidden_size // num_level
        self.num_level = num_level
        self.linear = nn.Linear(full_size, hidden_size, bias=False)
        self.linear_final = Parameter(torch.ones(1, hidden_size), requires_grad = True)
        self.output_size = hidden_size
        print("Full Attention: (atten. {} -> {}, take {}) x {}".format(self.full_size, self.attsize_per_lvl, hidden_size // num_level, self.num_level))

    def forward(self, x1_att, x2_att, x1, x2, x2_mask):
        """
        x1_att: batch * len1 * full_size
        x2_att: batch * len2 * full_size
        x1: batch * len1 * hidden_size
        x2: batch * len2 * hidden_size
        x2_mask: batch * len2
        """
        x1_att = dropout(x1_att, p=my_dropout_p, training=self.training)
        x2_att = dropout(x2_att, p=my_dropout_p, training=self.training)

        x1_key = F.relu(self.linear(x1_att.view(-1, self.full_size)))
        x2_key = F.relu(self.linear(x2_att.view(-1, self.full_size)))
        final_v = self.linear_final.expand_as(x2_key)
        x2_key = final_v * x2_key

        x1_rep = x1_key.view(-1, x1.size(1), self.num_level, self.attsize_per_lvl).transpose(1, 2).contiguous().view(-1, x1.size(1), self.attsize_per_lvl)
        x2_rep = x2_key.view(-1, x2.size(1), self.num_level, self.attsize_per_lvl).transpose(1, 2).contiguous().view(-1, x2.size(1), self.attsize_per_lvl)

        scores = x1_rep.bmm(x2_rep.transpose(1, 2)).view(-1, self.num_level, x1.size(1), x2.size(1)) # batch * num_level * len1 * len2

        x2_mask = x2_mask.unsqueeze(1).unsqueeze(2).expand_as(scores)
        scores.data.masked_fill_(x2_mask.data, -float('inf'))

        alpha_flat = F.softmax(scores.view(-1, x2.size(1)))
        alpha = alpha_flat.view(-1, x1.size(1), x2.size(1))

        size_per_level = self.hidden_size // self.num_level
        atten_seq = alpha.bmm(x2.contiguous().view(-1, x2.size(1), self.num_level, size_per_level).transpose(1, 2).contiguous().view(-1, x2.size(1), size_per_level))

        return atten_seq.view(-1, self.num_level, x1.size(1), size_per_level).transpose(1, 2).contiguous().view(-1, x1.size(1), self.hidden_size)

# For summarizing a set of vectors into a single vector
class LinearSelfAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x = dropout(x, p=my_dropout_p, training=self.training)

        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha

# Answer finding
class MLPFunc(nn.Module):
    """
    A multi-layer perceptron function for x: o = v'tanh(Wx+b).
    """
    def __init__(self, input_size, hidden_size, num_class):
        super(MLPFunc, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.linear_final = nn.Linear(hidden_size, num_class, bias=False)

    def forward(self, x):
        """
        x = batch * input_size
        """
        x = dropout(x, p=my_dropout_p, training=self.training)
        h = F.tanh(self.linear(x))
        h = dropout(h, p=my_dropout_p, training=self.training)
        o = self.linear_final(h)
        return o # batch * num_classes

# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------

# by default in PyTorch, +-*/ are all element-wise
def uniform_weights(x, x_mask): # used in lego_reader.py
    """Return uniform weights over non-masked input."""
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha

def weighted_avg(x, weights): # used in lego_reader.py
    """ x = batch * len * d
        weights = batch * len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)
