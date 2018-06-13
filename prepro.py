import re
import spacy
import msgpack
import unicodedata
import numpy as np
import argparse
import collections
import os.path
import multiprocessing
import logging
import random
from general_utils import normalize_text, build_embedding, load_glove_vocab, pre_proc, feature_gen, token2id, process_jsonlines

# Fixed Parameters for MultiNLI_1.0
trn_file = 'multinli_1.0/multinli_1.0_train.jsonl'
trn_meta_msgpack = 'multinli_1.0/train_meta.msgpack'
trn_data_msgpack = 'multinli_1.0/train_data.msgpack'

dev_file = 'multinli_1.0/multinli_1.0_dev_mismatched.jsonl'
dev_msgpack = 'multinli_1.0/dev_mismatch_preprocessed.msgpack'

tst_file = 'multinli_1.0/multinli_1.0_dev_matched.jsonl'
tst_msgpack = 'multinli_1.0/dev_match_preprocessed.msgpack'

# Parameters
parser = argparse.ArgumentParser(
    description='Preprocess the data.'
)
parser.add_argument('--wv_file', default='glove/glove.840B.300d.txt',
                    help='path to word vector file.')
parser.add_argument('--wv_dim', type=int, default=300,
                    help='word vector dimension.')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='number of threads for preprocessing.')
parser.add_argument('--seed', type=int, default=1023,
                    help='random seed for data shuffling, embedding init, etc.')

args = parser.parse_args()
wv_file = args.wv_file
wv_dim = args.wv_dim
nlp = spacy.load('en', parser=False)

random.seed(args.seed)
np.random.seed(args.seed)

#================================================================
#=========================== GloVe ==============================
#================================================================

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

log.info('start data preparing... (using {} threads)'.format(args.threads))

glove_vocab = load_glove_vocab(wv_file, wv_dim) # return a "set" of vocabulary
log.info('glove loaded.')

#===============================================================
#=================== Work on training data =====================
#===============================================================

train = process_jsonlines(trn_file)
log.info('train jsonline data flattened.')

trP_iter = (pre_proc(p) for p in train.P)
trH_iter = (pre_proc(h) for h in train.H)
trP_docs = [doc for doc in nlp.pipe(
    trP_iter, batch_size=64, n_threads=args.threads)]
trH_docs = [doc for doc in nlp.pipe(
    trH_iter, batch_size=64, n_threads=args.threads)]

# tokens
trP_tokens = [[normalize_text(w.text) for w in doc] for doc in trP_docs]
trH_tokens = [[normalize_text(w.text) for w in doc] for doc in trH_docs]
log.info('All tokens for training are obtained.')

# features
trP_tags, trP_ents, trP_features = feature_gen(trP_docs, trH_docs)
trH_tags, trH_ents, trH_features = feature_gen(trH_docs, trP_docs)
log.info('features for training is generated.')

def build_train_vocab(A, B): # vocabulary will also be sorted accordingly
    counter = collections.Counter(w for doc in A + B for w in doc)
    vocab = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)

    total = sum(counter.values())
    matched = sum(counter[t] for t in vocab)
    log.info('vocab {1}/{0} OOV {2}/{3} ({4:.4f}%)'.format(
        len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    return vocab

# vocab
tr_vocab = build_train_vocab(trH_tokens, trP_tokens)
trP_ids = token2id(trP_tokens, tr_vocab, unk_id=1)
trH_ids = token2id(trH_tokens, tr_vocab, unk_id=1)

# tags
vocab_tag = list(nlp.tagger.tag_names)
trP_tag_ids = token2id(trP_tags, vocab_tag)
trH_tag_ids = token2id(trH_tags, vocab_tag)

# entities
vocab_ent = [''] + nlp.entity.cfg[u'actions'][1]
trP_ent_ids = token2id(trP_ents, vocab_ent)
trH_ent_ids = token2id(trH_ents, vocab_ent)

log.info('Found {} POS tags.'.format(len(vocab_tag)))
log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))
log.info('vocabulary for training is built.')

tr_embedding = build_embedding(wv_file, tr_vocab, wv_dim)
log.info('got embedding matrix for training.')

meta = {
    'vocab': tr_vocab,
    'embedding': tr_embedding.tolist()
}
with open(trn_meta_msgpack, 'wb') as f:
    msgpack.dump(meta, f, encoding='utf8')

result = {
    'premise_ids': trP_ids,
    'premise_features': trP_features, # exact match, tf
    'premise_tags': trP_tag_ids, # POS tagging
    'premise_ents': trP_ent_ids, # Entity recognition
    'hypothesis_ids': trH_ids,
    'hypothesis_features': trH_features, # exact match, tf
    'hypothesis_tags': trH_tag_ids, # POS tagging
    'hypothesis_ents': trH_ent_ids, # Entity recognition
    'answers': train.label
}
with open(trn_data_msgpack, 'wb') as f:
    msgpack.dump(result, f, encoding='utf8')

log.info('saved training to disk.')

#==========================================================
#=================== Work on dev&test =====================
#==========================================================

def preprocess_eval_data(filename, output_msgpack):
    EvalData = process_jsonlines(filename)

    filename = os.path.basename(filename)
    log.info(filename + ' flattened.')

    EvalDataP_iter = (pre_proc(p) for p in EvalData.P)
    EvalDataH_iter = (pre_proc(h) for h in EvalData.H)
    EvalDataP_docs = [doc for doc in nlp.pipe(
        EvalDataP_iter, batch_size=64, n_threads=args.threads)]
    EvalDataH_docs = [doc for doc in nlp.pipe(
        EvalDataH_iter, batch_size=64, n_threads=args.threads)]

    # tokens
    EvalDataP_tokens = [[normalize_text(w.text) for w in doc] for doc in EvalDataP_docs]
    EvalDataH_tokens = [[normalize_text(w.text) for w in doc] for doc in EvalDataH_docs]
    log.info('All tokens for ' + filename + ' are obtained.')

    # features
    EvalDataP_tags, EvalDataP_ents, EvalDataP_features = feature_gen(EvalDataP_docs, EvalDataH_docs)
    EvalDataH_tags, EvalDataH_ents, EvalDataH_features = feature_gen(EvalDataH_docs, EvalDataP_docs)
    log.info('features for ' + filename + ' is generated.')

    def build_EvalData_vocab(A, B): # most vocabulary comes from tr_vocab
        existing_vocab = set(tr_vocab)
        new_vocab = list(set([w for doc in A + B for w in doc if w not in existing_vocab and w in glove_vocab]))
        vocab = tr_vocab + new_vocab
        log.info('train vocab {0}, total vocab {1}'.format(len(tr_vocab), len(vocab)))
        return vocab

    # vocab
    EvalData_vocab = build_EvalData_vocab(EvalDataP_tokens, EvalDataH_tokens) # tr_vocab is a subset of EvalData_vocab
    EvalDataP_ids = token2id(EvalDataP_tokens, EvalData_vocab, unk_id=1)
    EvalDataH_ids = token2id(EvalDataH_tokens, EvalData_vocab, unk_id=1)

    # tags
    EvalDataP_tag_ids = token2id(EvalDataP_tags, vocab_tag)
    EvalDataH_tag_ids = token2id(EvalDataH_tags, vocab_tag) # vocab_tag same as training

    # entities
    EvalDataP_ent_ids = token2id(EvalDataP_ents, vocab_ent) # vocab_ent same as training
    EvalDataH_ent_ids = token2id(EvalDataH_ents, vocab_ent) # vocab_ent same as training
    log.info('vocabulary for ' + filename + ' is built.')

    EvalData_embedding = build_embedding(wv_file, EvalData_vocab, wv_dim) # tr_embedding is a submatrix of EvalData_embedding
    log.info('got embedding matrix for ' + filename)

    result = {
        'premise_ids': EvalDataP_ids,
        'premise_features': EvalDataP_features, # exact match, tf
        'premise_tags': EvalDataP_tag_ids, # POS tagging
        'premise_ents': EvalDataP_ent_ids, # Entity recognition
        'hypothesis_ids': EvalDataH_ids,
        'hypothesis_features': EvalDataH_features, # exact match, tf
        'hypothesis_tags': EvalDataH_tag_ids, # POS tagging
        'hypothesis_ents': EvalDataH_ent_ids, # Entity recognition
        'vocab': EvalData_vocab,
        'embedding': EvalData_embedding.tolist(),
        'answers': EvalData.label
    }
    with open(output_msgpack, 'wb') as f:
        msgpack.dump(result, f)

    log.info('saved ' + output_msgpack + ' to disk.')

preprocess_eval_data(dev_file, dev_msgpack)
preprocess_eval_data(tst_file, tst_msgpack)
