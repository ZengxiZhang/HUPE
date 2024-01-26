
import torch
from torch import nn as nn

from models.modules import thops
from models.modules.flow import Conv2d, Conv2dZeros
from utils.util import opt_get
from models.modules.flow import unsqueeze2d, squeeze2d
import time
from models.modules.cbam_fromdehazeflow import CBAM
class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = opt_get(opt, ['network_G', 'flow', 'conditionInFeaDim'], 320)
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels

        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'], 0.0001)

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        # self.SF = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
        #                       out_channels=self.channels_for_co * 2,
        #                       hidden_channels=self.hidden_channels,
        #                       kernel_hidden=self.kernel_hidden,
        #                       n_hidden_layers=self.n_hidden_layers)
        # self.SF = self.F(in_channels=self.channels_for_nn,
        #                       out_channels=self.channels_for_co * 2,
        #                       hidden_channels=self.hidden_channels,
        #                       kernel_hidden=self.kernel_hidden,
        #                       n_hidden_layers=self.n_hidden_layers)
        # self.SF = FFTConvBlock(in_size = self.channels_for_nn,# + self.in_channels_rrdb,
        #                       out_size = self.channels_for_co * 2,
        #                       downsample = False,
        #                       relu_slope = 0.2,
        #                        use_FFT_AMP  = True,
        #                       use_FFT_PHASE = False)
        self.SF = FFTConvBlock(in_size = self.channels_for_nn,# + self.in_channels_rrdb,
                              out_size = self.channels_for_co * 2,
                              downsample = False,
                              relu_slope = 0.2)
        # self.SF_P = FFTConvBlock(in_size = self.channels_for_nn,# + self.in_channels_rrdb,
        #                       out_size = self.channels_for_co * 2,
        #                       downsample = False,
        #                       relu_slope = 0.2,
        #                        use_FFT_AMP  = False,
        #                       use_FFT_PHASE = True)

        ################################## 用来做公式计算 ###################################
        self.fFeatures = self.F(in_channels=self.in_channels_rrdb,
                                out_channels=int(self.in_channels *4/3) ,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)
        self.opt = opt
        self.le_curve = opt['le_curve'] if opt['le_curve'] is not None else False
        if self.le_curve:
            self.fCurve = self.F(in_channels=self.in_channels_rrdb,
                                 out_channels=self.in_channels,
                                 hidden_channels=self.hidden_channels,
                                 kernel_hidden=self.kernel_hidden,
                                 n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        # ft 是 条件encoder得到的特征
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)
            # Feature Conditional
            scaleFt, shiftFt, scaleFt_ = self.feature_extract(ft, self.fFeatures)
            # scaleFt_  = torch.cat((scaleFt,scaleFt,scaleFt),1)
            # shiftFt_  = torch.cat((shiftFt,shiftFt,shiftFt),1)
            z = z * scaleFt
            z = z + shiftFt
            logdet = logdet + self.get_logdet(scaleFt)
            # Self Conditional
            z1, z2 = self.split(z)
            # scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            scale, shift = self.feature_extract_aff(z1, ft, self.SF)
            # scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input
            # Self Conditional
            z1, z2 = self.split(z)
            # scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            scale, shift = self.feature_extract_aff(z1, ft, self.SF)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale
            z2 = z2 - shift
            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale)
            # Feature Conditional
            scaleFt, shiftFt, scaleFt_ = self.feature_extract(ft, self.fFeatures)
            # scaleFt_ = torch.cat((scaleFt, scaleFt, scaleFt),1)
            # shiftFt_ = torch.cat((shiftFt, shiftFt, shiftFt),1)
            z = z - shiftFt
            z = z / scaleFt
            logdet = logdet - self.get_logdet(scaleFt)
            output = z
        return output, logdet, scaleFt_

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        h = f(z)
        # shift, scale = thops.split_feature(h, "cross")
        # scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        a, t = thops.split_feature(h, "getta")
        t = torch.sigmoid( t + 2.) + self.affine_eps 
        t_  = torch.cat((t, t, t),1)
        a = a * ( 1 - t_ )
        # return scale, shift
        return t_, a, t

    # def feature_extract_aff(self, z1, ft, f):
    def feature_extract_aff(self, z1, ft, A):
        # print(z1.shape)
        # print(ft.shape)
        # z = torch.cat([z1, ft], dim = 1)
        z = z1
        # print(z.shape)
        h = A(z)
        # print(h.shape)
        shift, scale = thops.split_feature(h, "cross")
        # print(shift.shape)
        # print(scale.shape)
        # time.sleep(1000)
        # scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        scale = (torch.sigmoid(scale+ 2.) + self.affine_eps)
   
        return scale, shift
    # def feature_extract_aff(self, z1, ft, f):
    #     z = torch.cat([z1, ft], dim=1)
        
    #     h = f(z)
    #     shift, scale = thops.split_feature(h, "cross")


    #     scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
    #     return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn ]
        z2 = z[:,  self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            # layers.append(nn.BatchNorm2d(hidden_channels)),
            layers.append(nn.ReLU(inplace=False))
        layers.append(CBAM(gate_channels=64))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)

    # def SpatialFrequency(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
    #     layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

    #     for _ in range(n_hidden_layers):
    #         layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
    #         # layers.append(nn.BatchNorm2d(hidden_channels)),
    #         layers.append(nn.ReLU(inplace=False))
    #     layers.append(CBAM(gate_channels=64))
    #     layers.append(Conv2dZeros(hidden_channels, out_channels))

    #     return nn.Sequential(*layers)
# Adaptive Dynamic Filter Block
class Context(nn.Module):
    def __init__(self, in_channels=24, kernel_size=3):
        super().__init__()
        self.conv_sa = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=in_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_ca = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2) 

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        sa_x = self.conv_sa(input_x)  
        y = self.avg_pool(input_x)
        ca_x = self.conv_ca(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out  = sa_x + ca_x
        return out

class AFG(nn.Module):
    def __init__(self, in_channels=24, kernel_size=3):
        super(AFG, self).__init__()
        self.kernel_size = kernel_size
        self.sekg = Context(in_channels, kernel_size)
        self.fusion = nn.Conv2d(in_channels*3, in_channels, 1, 1, 0)
        self.kernel = nn.Conv2d(in_channels, in_channels*kernel_size*kernel_size, 1, 1, 0)
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)

    def forward(self, x, pha, amp):
        fusion = self.fusion(torch.cat([x, pha, amp], dim=1))
        b, c, h, w = x.size()
        att = self.sekg(fusion)
        # return att
        kers = self.kernel(att)
        filter_x = kers.reshape([b, c, self.kernel_size*self.kernel_size, h, w])
        unfold_x = self.unfold(x).reshape(b, c, -1, h, w)
        out = (unfold_x * filter_x).sum(2)
        
        # return out + x
        return out
def conv_down(in_size, out_size, bias=False):
    layer = nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

class FFTConvBlock(nn.Module):
    # def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_FFT_PHASE=False, use_FFT_AMP=False):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False):
        super(FFTConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0,dtype=torch.float32)
        self.use_csff = use_csff
        # self.use_FFT_PHASE = use_FFT_PHASE
        # self.use_FFT_AMP = use_FFT_AMP

        self.resConv = nn.Sequential(*[
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True,dtype=torch.float32),
            # nn.LeakyReLU(relu_slope, inplace=False),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True,dtype=torch.float32),
            # nn.LeakyReLU(relu_slope, inplace=False)
        ])

        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0,dtype=torch.float32)
        # self.fftConv2 = nn.Sequential(*[
        #     nn.Conv2d(out_size, out_size, 1, 1, 0),
        #     nn.LeakyReLU(relu_slope, inplace=False),
        #     nn.Conv2d(out_size, out_size, 1, 1, 0)
        # ])
        self.fftConvP2 = nn.Sequential(*[
            nn.Conv2d(out_size, out_size, 1, 1, 0),
            # nn.LeakyReLU(relu_slope, inplace=False),
            nn.Conv2d(out_size, out_size, 1, 1, 0)
        ])
        self.fftConvA2 = nn.Sequential(*[
            nn.Conv2d(out_size, out_size, 1, 1, 0),
            # nn.LeakyReLU(relu_slope, inplace=False),
            nn.Conv2d(out_size, out_size, 1, 1, 0)
        ])

        # self.fusion = nn.Conv2d(out_size*3, out_size, 1, 1, 0)
        # self.fusion =nn.Sequential(*[ nn.Conv2d(out_size*3, out_size, 3, 1, 1),
        #                 nn.LeakyReLU(relu_slope, inplace=False),
        #                 nn.Conv2d(out_size, out_size, 3, 1, 1),])
        # self.fusion =nn.Sequential(*[ nn.Conv2d(out_size*3, out_size, 3, 1, 1),
        #                 nn.LeakyReLU(relu_slope, inplace=False)])
        self.fusion =nn.Sequential(*[ nn.Conv2d(out_size*3, out_size, 3, 1, 1),
                        nn.Conv2d(out_size, out_size, 3, 1, 1),])

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)
        # self.afg = (AFG(out_size, 3))

    def forward(self, x, enc=None, dec=None):
        res_out = self.resConv(x)
        identity = self.identity(x)
        out = res_out + identity
        out = out.float()
        x_fft = torch.fft.rfft2(out, norm='backward')
        x_amp = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)

        x_phase_ = self.fftConvP2(x_phase)
        x_fft_out_p = torch.fft.irfft2(x_amp*torch.exp(1j*x_phase_), norm='backward')
        # print('x_amp: ',str(torch.max(x_amp).item()), str(torch.min(x_amp).item()))
        x_amp_ = self.fftConvA2(x_amp)
        # x_amp_=x_amp
        # print('x_amp_: ',str(torch.max(x_amp_).item()), str(torch.min(x_amp_).item()))
        x_fft_out_a = torch.fft.irfft2(x_amp_*torch.exp(1j*x_phase), norm='backward')
        
        # print('s: ',str(torch.max(out).item()), str(torch.min(out).item()))
        # print('p: ',str(torch.max(x_fft_out_p).item()), str(torch.min(x_fft_out_p).item()))
        # print('a: ',str(torch.max(x_fft_out_a).item()), str(torch.min(x_fft_out_a).item()))
        # out = self.fusion(torch.cat([out, x_fft_out_p, x_fft_out_a], dim=1))
        out = self.fusion(torch.cat([out, x_fft_out_a, x_fft_out_p], dim=1))
        # out = self.fusion(torch.cat([out, x_fft_out_p], dim=1))
        # out= self.afg(out,out,out)
        # print('out: ',str(torch.max(out).item()), str(torch.min(out).item()))
        # out = self.fusion(torch.cat([out, x_fft_out_a,x_fft_out_p], dim=1))
        # out = self.fusion(torch.cat([out, out, out], dim=1))
        # out = self.fusion(torch.cat([out, out, out], dim=1))
        # print(out.type())
        # time.sleep(1000)
        
        # print(torch.max(out).item(), torch.min(out).item())
        # time.sleep(100)
        # print(torch.min(out).item())
        return out
        # return self.fusion(torch.cat([out, out, x_fft_out_a], dim=1))
