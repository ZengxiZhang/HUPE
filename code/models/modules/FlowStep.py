


import torch
from torch import nn as nn
import time
import models.modules
import models.modules.Permutations
from models.modules import flow, thops, FlowAffineCouplingsAblation
from utils.util import opt_get
from models.modules.flow import unsqueeze2d, squeeze2d


def getConditional(rrdbResults, position):
    img_ft = rrdbResults if isinstance(rrdbResults, torch.Tensor) else rrdbResults[position]
    return img_ft


class FlowStep(nn.Module):
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "squeeze_invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_alternating_2_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlign": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1SubblocksShuf": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder4": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0, flow_permutation="invconv", flow_coupling="additive",
                 LU_decomposed=False, opt=None, image_injector=None, idx=None, acOpt=None, normOpt=None, in_shape=None,
                 position=None):
        # check configures
        assert flow_permutation in FlowStep.FlowPermutation, \
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.image_injector = image_injector

        self.norm_type = normOpt['type'] if normOpt else 'ActNorm2d'
        self.position = normOpt['position'] if normOpt else None

        self.in_shape = in_shape
        self.position = position
        self.acOpt = acOpt

        # 1. actnorm
        self.actnorm = models.modules.FlowActNorms.ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = models.modules.Permutations.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)

        # 3. coupling
        if flow_coupling == "CondAffineSeparatedAndCond":
            self.affine = models.modules.FlowAffineCouplingsAblation.CondAffineSeparatedAndCond(in_channels=in_channels,
                                                                                                opt=opt)
        elif flow_coupling == "noCoupling":
            pass
        else:
            raise RuntimeError("coupling not Found:", flow_coupling)

    def forward(self, input, logdet=None, reverse=False, rrdbResults=None):
        if not reverse:
            return self.normal_flow(input, logdet, rrdbResults)
        else:
            return self.reverse_flow(input, logdet, rrdbResults)

    def normal_flow(self, z, logdet, rrdbResults=None):
        # 12,144 64->144
        # 48,72 64->72
        # 192,36 64->36
        if self.flow_coupling == "bentIdentityPreAct":# default = 'noCoupling'
            z, logdet = self.bentIdentPar(z, logdet, reverse=False)
        # 1. actnorm
        if self.norm_type == "ConditionalActNormImageInjector":# default = 'ActNorm2d'
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.actnorm(z, img_ft=img_ft, logdet=logdet, reverse=False)
        elif self.norm_type == "noNorm":
            pass
        else:# 从这里走
            z, logdet = self.actnorm(z, logdet=logdet, reverse=False)#调整了均值等一系列东西，类似于预处理吧
        # 2. permute
        # self.flow_permutation = 'invconv'
        # 调用的是 models.module.Permutations的InvertibleConv1x1
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False)
        need_features = self.affine_need_features()# default = False
        # 3. coupling 
        scale_t = None
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            # 默认在每个阶段的第三到第六个flow的模型中使用到rrdbResults
            # print(self.position)# 三个阶段分别是 fea_up0.5, fea_up0.25, fea_up0。125
            img_ft = getConditional(rrdbResults, self.position)#维度不变,这个代码里，这个函数相当于啥也不做
            z, logdet, scale_t = self.affine(input = z, logdet = logdet, reverse = False, ft=img_ft)
            if scale_t.shape[1] == 4:
                scale_t = unsqueeze2d(scale_t, 2)
            if scale_t.shape[1] == 16:
                scale_t = unsqueeze2d(scale_t, 4)
            if scale_t.shape[1] == 64:
                scale_t = unsqueeze2d(scale_t, 8)
        return z, logdet, scale_t

    def reverse_flow(self, z, logdet, rrdbResults=None):

        need_features = self.affine_need_features()
        scale_t= None
        # 1.coupling
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet, scale_t = self.affine(input = z, logdet = logdet, reverse = True, ft=img_ft)
            if scale_t.shape[1] == 4:
                scale_t = unsqueeze2d(scale_t, 2)
            if scale_t.shape[1] == 16:
                scale_t = unsqueeze2d(scale_t, 4)
            if scale_t.shape[1] == 64:
                scale_t = unsqueeze2d(scale_t, 8)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)
        # 3. actnorm
        z, logdet = self.actnorm(z, logdet = logdet, reverse = True)

        return z, logdet, scale_t

    def affine_need_features(self):
        need_features = False
        try:
            need_features = self.affine.need_features
        except:
            pass
        return need_features
