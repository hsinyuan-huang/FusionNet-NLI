import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging

from torch.autograd import Variable
from .utils import AverageMeter
from .FusionNet import FusionNet

logger = logging.getLogger(__name__)

class FusionNet_Model(object):
    """
    High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, embedding=None, state_dict=None):
        # Book-keeping.
        self.opt = opt
        self.updates = state_dict['updates'] if state_dict and 'updates' in state_dict else 0
        self.eval_embed_transfer = True
        self.train_loss = AverageMeter()

        # Building network.
        self.network = FusionNet(opt, embedding)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            for k, v in list(self.network.state_dict().items()):
                if k not in state_dict['network']:
                    state_dict['network'][k] = v
            self.network.load_state_dict(state_dict['network'])

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters)
        elif opt['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])
        if state_dict and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

        if opt['fix_embeddings']:
            wvec_size = 0
        else:
            wvec_size = (opt['vocab_size'] - opt['tune_partial']) * opt['embedding_dim']
        self.total_param = sum([p.nelement() for p in parameters]) - wvec_size

    def update(self, ex):
        # Train mode
        self.network.train()

        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True)) for e in ex[:10]]
            targets = Variable(ex[10].cuda(async=True))
        else:
            inputs = [Variable(e) for e in ex[:10]]
            targets = Variable(ex[10])

        # Run forward
        scores = self.network(*inputs) # output: [batch_size, 3]

        # Compute loss and accuracies
        loss = F.cross_entropy(scores, targets)
        self.train_loss.update(loss.data[0], ex[0].size(0))

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.opt['grad_clipping'])

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_embeddings()
        self.eval_embed_transfer = True

    def predict(self, ex, best_nth=1):
        # Eval mode
        self.network.eval()

        # Transfer trained embedding to evaluation embedding
        if self.eval_embed_transfer:
            self.update_eval_embed()
            self.eval_embed_transfer = False

        # Transfer to GPU
        if self.opt['cuda']:
            # volatile means no gradient is needed
            inputs = [Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:10]]
        else:
            inputs = [Variable(e, volatile=True) for e in ex[:10]]

        # Run forward
        scores = self.network(*inputs) # output: [batch_size, 3]

        # Transfer to CPU/normal tensors and find classes for instances
        scores = scores.data.cpu()
        predictions = torch.max(scores, 1)[1].tolist()

        return predictions # list of classes

    # allow the evaluation embedding be larger than training embedding
    # this is helpful if we have pretrained word embeddings
    def setup_eval_embed(self, eval_embed, padding_idx = 0):
        # eval_embed should be a supermatrix of training embedding
        self.network.eval_embed = nn.Embedding(eval_embed.size(0),
                                               eval_embed.size(1),
                                               padding_idx = padding_idx)
        self.network.eval_embed.weight.data = eval_embed
        for p in self.network.eval_embed.parameters():
            p.requires_grad = False
        self.eval_embed_transfer = True

        if self.opt['use_cove']:
            self.network.CoVe.setup_eval_embed(eval_embed)

    def update_eval_embed(self):
        # update evaluation embedding to trained embedding
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial']
            self.network.eval_embed.weight.data[0:offset] \
                = self.network.embedding.weight.data[0:offset]
        else:
            offset = 10
            self.network.eval_embed.weight.data[0:offset] \
                = self.network.embedding.weight.data[0:offset]

    def reset_embeddings(self):
        # Reset fixed embeddings to original value
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial']
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def save_for_predict(self, filename, epoch):
        network_state = self.network.state_dict()
        if 'eval_embed.weight' in network_state:
            del network_state['eval_embed.weight']
        if 'fixed_embedding' in network_state:
            del network_state['fixed_embedding']
        params = {
            'state_dict': {'network': network_state},
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')

    def cuda(self):
        self.network.cuda()
