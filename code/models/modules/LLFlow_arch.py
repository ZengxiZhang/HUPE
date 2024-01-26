


import math
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.RRDBNet_arch import RRDBNet
from models.modules.ConditionEncoder import ConEncoder1, NoEncoder
from models.modules.FlowUpsamplerNet import FlowUpsamplerNet
import models.modules.thops as thops
import models.modules.flow as flow
from models.modules.color_encoder import ColorEncoder
from utils.util import opt_get
from models.modules.flow import unsqueeze2d, squeeze2d
from torch.cuda.amp import autocast
import time
class LLFlow(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, K=None, opt=None, step=None):
        super(LLFlow, self).__init__()
        self.crop_size = opt['datasets']['train']['crop_size']
        self.opt = opt
        self.quant = 255 if opt_get(opt, ['datasets', 'train', 'quant']) is \
                            None else opt_get(opt, ['datasets', 'train', 'quant'])

        if opt['cond_encoder'] == 'ConEncoder1':# default =True
            self.RRDB = ConEncoder1(in_nc, out_nc, nf, nb, gc, scale, opt)
        elif opt['cond_encoder'] ==  'NoEncoder':
            self.RRDB = None # NoEncoder(in_nc, out_nc, nf, nb, gc, scale, opt)
        elif opt['cond_encoder'] == 'RRDBNet':
            # if self.opt['encode_color_map']: print('Warning: ''encode_color_map'' is not implemented in RRDBNet')
            self.RRDB = RRDBNet(in_nc, out_nc, nf, nb, gc, scale, opt)
        else:
            print('WARNING: Cannot find the conditional encoder %s, select RRDBNet by default.' % opt['cond_encoder'])
            # if self.opt['encode_color_map']: print('Warning: ''encode_color_map'' is not implemented in RRDBNet')
            opt['cond_encoder'] = 'RRDBNet'
            self.RRDB = RRDBNet(in_nc, out_nc, nf, nb, gc, scale, opt)


        if self.opt['encode_color_map']:
            self.color_map_encoder = ColorEncoder(nf=nf, opt=opt)

        hidden_channels = opt_get(opt, ['network_G', 'flow', 'hidden_channels'])
        hidden_channels = hidden_channels or 64
        self.RRDB_training = True  # Default is true

        train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])
        set_RRDB_to_train = False
        if set_RRDB_to_train and self.RRDB:
            self.set_rrdb_training(True)

        self.flowUpsamplerNet = \
            FlowUpsamplerNet((self.crop_size, self.crop_size, 3), hidden_channels, K,
                             flow_coupling=opt['network_G']['flow']['coupling'], opt=opt)
        self.i = 0
        if self.opt['to_yuv']:
            self.A_rgb2yuv = torch.nn.Parameter(torch.tensor([[0.299, -0.14714119, 0.61497538],
                                                              [0.587, -0.28886916, -0.51496512],
                                                              [0.114, 0.43601035, -0.10001026]]), requires_grad=False)
            self.A_yuv2rgb = torch.nn.Parameter(torch.tensor([[1., 1., 1.],
                                                              [0., -0.39465, 2.03211],
                                                              [1.13983, -0.58060, 0]]), requires_grad=False)
        if self.opt['align_maxpool']:
            self.max_pool = torch.nn.MaxPool2d(3)

    def set_rrdb_training(self, trainable):
        if self.RRDB_training != trainable:
            for p in self.RRDB.parameters():
                p.requires_grad = trainable
            self.RRDB_training = trainable
            return True
        return False


    @autocast()
    def forward(self, gt=None, lr=None, z=None, eps_std=None, reverse=False, epses=None, reverse_with_grad=False,
                lr_enc=None,
                add_gt_noise=False, step=None, y_label=None, align_condition_feature=False):
        with torch.no_grad():
            return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                        add_gt_noise=add_gt_noise)

   
    def rrdbPreprocessing(self, lr):
        rrdbResults = self.RRDB(lr, get_steps=True)
        block_idxs = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        if len(block_idxs) > 0:
            low_level_features = [rrdbResults["block_{}".format(idx)] for idx in block_idxs]
            concat = torch.cat(low_level_features, dim=1)

            if opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'concat']) or False:
                keys = ['last_lr_fea', 'fea_up1', 'fea_up2', 'fea_up4']
                if 'fea_up0' in rrdbResults.keys():
                    keys.append('fea_up0')
                if 'fea_up-1' in rrdbResults.keys():
                    keys.append('fea_up-1')
                for k in keys:
                    h = rrdbResults[k].shape[2]
                    w = rrdbResults[k].shape[3]
                    rrdbResults[k] = torch.cat([rrdbResults[k], F.interpolate(concat, (h, w))], dim=1)
        return rrdbResults

    def get_score(self, disc_loss_sigma, z):
        score_real = 0.5 * (1 - 1 / (disc_loss_sigma ** 2)) * thops.sum(z ** 2, dim=[1, 2, 3]) - \
                     z.shape[1] * z.shape[2] * z.shape[3] * math.log(disc_loss_sigma)
        return -score_real

    def reverse_flow(self, lr, z, y_onehot, eps_std, epses=None, lr_enc=None, add_gt_noise=True):
        lr_noise = lr[:,:3,:,:] # 3 channel
        encoder_input = lr[:,3:8,:,:] # 5 channel
        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        pixels = thops.pixels(lr) * self.opt['scale'] ** 2
        if add_gt_noise:
            logdet = logdet - float(-np.log(self.quant) * pixels)
        lr_enc = self.rrdbPreprocessing(encoder_input)
        z = squeeze2d(lr_noise,8)
        x = self.flowUpsamplerNet(rrdbResults=lr_enc, z=z, eps_std=eps_std, reverse=True, epses=epses,
                                          logdet=logdet)
        return x
