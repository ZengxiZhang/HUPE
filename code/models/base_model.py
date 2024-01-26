

import time
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import natsort
import glob


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt.get('gpu_ids', None) is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        self.scaler = None

    def load_network(self, load_path, network, strict=True, submodule=None):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        if not (submodule is None or submodule.lower() == 'none'.lower()):
            network = network.__getattr__(submodule)
        if str(type(load_path))=='<class \'bool\'>':
            return False
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

