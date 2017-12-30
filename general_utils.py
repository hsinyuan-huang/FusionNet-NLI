import re
import os
import sys
import random
import string
import logging
import argparse
import unicodedata
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
import jsonlines
import numpy as np

#===========================================================================
#================= All for preprocessing SQuAD data set ====================
#===========================================================================

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def load_glove_vocab(file, wv_dim):
    vocab = set()
    with open(file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            vocab.add(token)
    return vocab

def space_extend(matchobj):
    return ' ' + matchobj.group(0) + ' '

def pre_proc(text):
    # make hyphens, spaces clean
    text = re.sub(u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/', space_extend, text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text

class SNLIData:
    def __init__(self, label, sent1, sent2):
        self.label = label
        self.P = sent1 # Premise
        self.H = sent2 # Hypothesis

def process_jsonlines(data_file):
    with jsonlines.open(data_file) as reader:
        snli_label = []
        snli_sent1 = []
        snli_sent2 = []
        for obj in reader:
            if obj['gold_label'] != '-':
                snli_label.append(obj['gold_label'])
                snli_sent1.append(obj['sentence1'])
                snli_sent2.append(obj['sentence2'])
        return SNLIData(snli_label, snli_sent1, snli_sent2)

def feature_gen(A_docs, B_docs):
    A_tags = [[w.tag_ for w in doc] for doc in A_docs]
    A_ents = [[w.ent_type_ for w in doc] for doc in A_docs]
    A_features = []

    for textA, textB in zip(A_docs, B_docs):
        counter_ = Counter(w.text.lower() for w in textA)
        total = sum(counter_.values())
        term_freq = [counter_[w.text.lower()] / total for w in textA]

        question_word = {w.text for w in textB}
        question_lower = {w.text.lower() for w in textB}
        question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in textB}
        match_origin = [w.text in question_word for w in textA]
        match_lower = [w.text.lower() in question_lower for w in textA]
        match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in textA]
        A_features.append(list(zip(term_freq, match_origin, match_lower, match_lemma)))

    return A_tags, A_ents, A_features

def build_embedding(embed_file, targ_vocab, wv_dim):
    vocab_size = len(targ_vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))
    emb[0] = 0 # <PAD> should be all 0 (using broadcast)

    w2id = {w: i for i, w in enumerate(targ_vocab)}
    with open(embed_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb

def token2id(docs, vocab, unk_id=None):
    w2id = {w: i for i, w in enumerate(vocab)}
    ids = [[w2id[w] if w in w2id else unk_id for w in doc] for doc in docs]
    return ids

#===========================================================================
#=================== Load Training and Evaluation data =====================
#===========================================================================

def text2class(ans):
    if ans == "neutral": return 0
    if ans == "entailment": return 1
    if ans == "contradiction": return 2
    assert(True)

def load_train_data(opt, train_meta, train_data):
    with open(train_meta, 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)

    with open(train_data, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    opt['num_features'] = len(data['premise_features'][0][0])

    train = list(zip( # list() due to lazy evaluation of zip
        data['premise_ids'],
        data['premise_features'],
        data['premise_tags'],
        data['premise_ents'],
        data['hypothesis_ids'],
        data['hypothesis_features'],
        data['hypothesis_tags'],
        data['hypothesis_ents'],
        [text2class(ans) for ans in data['answers']]
    ))
    return train, embedding, opt

def load_eval_data(opt, eval_data): # can be extended to true test set
    with open(eval_data, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(data['embedding'])

    assert opt['embedding_dim'] == embedding.size(1)
    assert opt['num_features'] == len(data['premise_features'][0][0])

    eval_set = list(zip(
        data['premise_ids'],
        data['premise_features'],
        data['premise_tags'],
        data['premise_ents'],
        data['hypothesis_ids'],
        data['hypothesis_features'],
        data['hypothesis_tags'],
        data['hypothesis_ents']
    ))
    return eval_set, embedding, [text2class(ans) for ans in data['answers']]

#===========================================================================
#================ For batch generation (train & predict) ===================
#===========================================================================

class BatchGen:
    def __init__(self, data, batch_size, gpu, evaluation=False):
        '''
        input:
            data - list of lists
            batch_size - int
        '''
        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu

        # random shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)

        # chunk into batches (if i + batch_size > data.size(0), it's fine)
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                assert len(batch) == 8
            else:
                assert len(batch) == 9 # + answer

            P_len = max(len(x) for x in batch[0])
            H_len = max(len(x) for x in batch[4])
            feature_len = len(batch[1][0][0])

            # Premise Tokens
            P_id = torch.LongTensor(batch_size, P_len).fill_(0)
            for i, doc in enumerate(batch[0]):
                P_id[i, :len(doc)] = torch.LongTensor(doc)

            # Premise Feature
            P_feature = torch.Tensor(batch_size, P_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                for j, feature in enumerate(doc):
                    P_feature[i, j, :] = torch.Tensor(feature)

            # Premise PoS
            P_tag = torch.LongTensor(batch_size, P_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                P_tag[i, :len(doc)] = torch.LongTensor(doc)

            # Premise NER
            P_ent = torch.LongTensor(batch_size, P_len).fill_(0)
            for i, doc in enumerate(batch[3]):
                P_ent[i, :len(doc)] = torch.LongTensor(doc)

            # Hypothesis Tokens
            H_id = torch.LongTensor(batch_size, H_len).fill_(0)
            for i, doc in enumerate(batch[4]):
                H_id[i, :len(doc)] = torch.LongTensor(doc)

            # Hypothesis Features
            H_feature = torch.Tensor(batch_size, H_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[5]):
                for j, feature in enumerate(doc):
                    H_feature[i, j, :] = torch.Tensor(feature)

            # Hypothesis PoS
            H_tag = torch.LongTensor(batch_size, H_len).fill_(0)
            for i, doc in enumerate(batch[6]):
                H_tag[i, :len(doc)] = torch.LongTensor(doc)

            # Hypothesis NER
            H_ent = torch.LongTensor(batch_size, H_len).fill_(0)
            for i, doc in enumerate(batch[7]):
                H_ent[i, :len(doc)] = torch.LongTensor(doc)

            # Premise, Hypothesis Masks
            P_mask = torch.eq(P_id, 0)
            H_mask = torch.eq(H_id, 0)

            # Label: neutral (0), entailment (1), contradiction (2)
            if not self.eval:
                label = torch.LongTensor(batch[8])

            if self.gpu: # page locked memory for async data transfer
                P_id = P_id.pin_memory()
                P_feature = P_feature.pin_memory()
                P_tag = P_tag.pin_memory()
                P_ent = P_ent.pin_memory()

                H_id = H_id.pin_memory()
                H_feature = H_feature.pin_memory()
                H_tag = H_tag.pin_memory()
                H_ent = H_ent.pin_memory()

                P_mask = P_mask.pin_memory()
                H_mask = H_mask.pin_memory()

            if self.eval:
                yield (P_id, P_feature, P_tag, P_ent, P_mask,
                       H_id, H_feature, H_tag, H_ent, H_mask)
            else:
                yield (P_id, P_feature, P_tag, P_ent, P_mask,
                       H_id, H_feature, H_tag, H_ent, H_mask, label)
