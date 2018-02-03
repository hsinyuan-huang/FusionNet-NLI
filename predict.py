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
    description='Predict using FusionNet model for Natural Language Inference.'
)
parser.add_argument('-m', '--model', default='',
                    help='testing model pathname, e.g. "models/checkpoint_epoch_11.pt"')
parser.add_argument('--test_data', default='snli_1.0/test_preprocessed.msgpack',
                    help='path to preprocessed testing (dev set 2) data file.')
parser.add_argument('-bs', '--batch_size', default=32)
parser.add_argument('--show', type=int, default=30)
parser.add_argument('--seed', type=int, default=1023,
                    help='random seed for data shuffling, dropout, etc.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
ch.setFormatter(formatter)
log.addHandler(ch)

def main():
    log.info('[program starts.]')
    checkpoint = torch.load(args.model)

    opt = checkpoint['config']
    state_dict = checkpoint['state_dict']
    model = FusionNet_Model(opt, state_dict = state_dict)
    log.info('[Model loaded.]')

    test, test_embedding, test_ans = load_eval_data(opt, args.test_data)
    model.setup_eval_embed(test_embedding)
    log.info('[Data loaded.]')

    if args.cuda:
        model.cuda()

    batches = BatchGen(test, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
    predictions = []
    for batch in batches:
        predictions.extend(model.predict(batch))
    acc = sum([x == y for x, y in zip(predictions, test_ans)]) / len(test_ans) * 100.0
    print("Accuracy =", acc)
    print(predictions[:args.show])
    print(test_ans[:args.show])

if __name__ == '__main__':
    main()
