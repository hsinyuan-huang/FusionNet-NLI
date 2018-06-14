import re
import os
import sys
import random
import string
import logging
import argparse
import pickle
from shutil import copyfile
from datetime import datetime
from collections import Counter, defaultdict
import torch
import msgpack
import numpy as np
from FusionModel.model import FusionNet_Model
from general_utils import BatchGen, load_train_data, load_eval_data

parser = argparse.ArgumentParser(
    description='Train FusionNet model for Natural Language Inference.'
)
# system
parser.add_argument('--name', default='', help='additional name of the current run')
parser.add_argument('--log_file', default='output.log',
                    help='path for log file.')
parser.add_argument('--log_per_updates', type=int, default=80,
                    help='log model loss per x updates (mini-batches).')

parser.add_argument('--train_meta', default='multinli_1.0/train_meta.msgpack',
                    help='path to preprocessed training meta file.')
parser.add_argument('--train_data', default='multinli_1.0/train_data.msgpack',
                    help='path to preprocessed training data file.')
parser.add_argument('--dev_data', default='multinli_1.0/dev_mismatch_preprocessed.msgpack',
                    help='path to preprocessed validation data file.')
parser.add_argument('--test_data', default='multinli_1.0/dev_match_preprocessed.msgpack',
                    help='path to preprocessed testing (dev set 2) data file.')

parser.add_argument('--MTLSTM_path', default='glove/MT-LSTM.pth')
parser.add_argument('--model_dir', default='models',
                    help='path to store saved models.')
parser.add_argument('--save_all', dest="save_best_only", action='store_false',
                    help='save all models in addition to the best.')
parser.add_argument('--do_not_save', action='store_true', help='don\'t save any model')
parser.add_argument('--seed', type=int, default=1023,
                    help='random seed for data shuffling, dropout, etc.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
# training
parser.add_argument('-e', '--epoches', type=int, default=20)
parser.add_argument('-bs', '--batch_size', type=int, default=32)
parser.add_argument('-op', '--optimizer', default='adamax',
                    help='supported optimizer: adamax, sgd, adadelta, adam')
parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
                    help='finetune top-x embeddings (including <PAD>, <UNK>).')
parser.add_argument('--fix_embeddings', action='store_true',
                    help='if true, `tune_partial` will be ignored.')
# model
parser.add_argument('--number_of_class', type=int, default=3)
parser.add_argument('--final_merge', default='linear_self_attn')

parser.add_argument('--hidden_size', type=int, default=125)
parser.add_argument('--enc_rnn_layers', type=int, default=2, help="Encoding RNN layers")
parser.add_argument('--inf_rnn_layers', type=int, default=2, help="Inference RNN layers")
parser.add_argument('--full_att_type', type=int, default=2)

parser.add_argument('--pos_size', type=int, default=56,
                    help='how many kinds of POS tags.')
parser.add_argument('--pos_dim', type=int, default=12,
                    help='the embedding dimension for POS tags.')
parser.add_argument('--ner_size', type=int, default=19,
                    help='how many kinds of named entity tags.')
parser.add_argument('--ner_dim', type=int, default=8,
                    help='the embedding dimension for named entity tags.')

parser.add_argument('--no_seq_dropout', dest='do_seq_dropout', action='store_false')
parser.add_argument('--my_dropout_p', type=float, default=0.3)
parser.add_argument('--dropout_emb', type=float, default=0.3)
parser.add_argument('--dropout_EM', type=float, default=0.6)

args = parser.parse_args()

if args.name != '':
    args.model_dir = args.model_dir + '_' + args.name
    args.log_file = os.path.dirname(args.log_file) + 'output_' + args.name + '.log'

# set model dir
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
ch.setFormatter(formatter)
log.addHandler(ch)

def main():
    log.info('[program starts.]')
    opt = vars(args) # changing opt will change args
    train, train_embedding, opt = load_train_data(opt, args.train_meta, args.train_data)
    dev, dev_embedding, dev_ans = load_eval_data(opt, args.dev_data)
    test, test_embedding, test_ans = load_eval_data(opt, args.test_data)
    log.info('[Data loaded.]')

    model = FusionNet_Model(opt, train_embedding)
    if args.cuda: model.cuda()
    log.info("[dev] Total number of params: {}".format(model.total_param))

    best_acc = 0.0

    for epoch in range(1, 1 + args.epoches):
        log.warning('Epoch {}'.format(epoch))

        # train
        batches = BatchGen(train, batch_size=args.batch_size, gpu=args.cuda)
        start = datetime.now()
        for i, batch in enumerate(batches):
            model.update(batch)
            if i % args.log_per_updates == 0:
                log.info('updates[{0:6}] train loss[{1:.5f}] remaining[{2}]'.format(
                    model.updates, model.train_loss.avg,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))

        # dev eval
        model.setup_eval_embed(dev_embedding)
        if args.cuda: model.cuda()

        batches = BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
        predictions = []
        for batch in batches:
            predictions.extend(model.predict(batch))
        acc = sum([x == y for x, y in zip(predictions, dev_ans)]) / len(dev_ans) * 100.0

        # test (or dev 2) eval
        model.setup_eval_embed(test_embedding)
        if args.cuda: model.cuda()

        batches = BatchGen(test, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
        predictions = []
        for batch in batches:
            predictions.extend(model.predict(batch))
        corr_acc = sum([x == y for x, y in zip(predictions, test_ans)]) / len(test_ans) * 100.0

        # save for predict
        if args.do_not_save == False:
            if args.save_best_only:
                if (acc + corr_acc)/2 > best_acc:
                    model_file = os.path.join(model_dir, 'best_model.pt')
                    model.save_for_predict(model_file, epoch)
                    log.info('[new best model saved.]')
            else:
                model_file = os.path.join(model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
                model.save_for_predict(model_file, epoch)
                if (acc + corr_acc)/2 > best_acc:
                    copyfile(
                        os.path.join(model_dir, model_file),
                        os.path.join(model_dir, 'best_model.pt'))
                    log.info('[new best model saved.]')
        if (acc + corr_acc)/2 > best_acc:
            best_acc = (acc + corr_acc)/2

        log.warning("Epoch {0} - dev Acc: {1:.3f}, dev2 Acc: {2:.3f} (best Acc: {3:.3f})".format(epoch, acc, corr_acc, best_acc))

if __name__ == '__main__':
    main()
