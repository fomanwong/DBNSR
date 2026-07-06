import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import math
import matplotlib.pyplot as plt
class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class ResAttentionBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.LeakyReLU(True), res_scale=1):
        super(ResAttentionBlock, self).__init__()
        m = []
        for i in range(1):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    if dilation==1:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias)
    elif dilation==2:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=dilation)

    else:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=3, bias=bias, dilation=dilation)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

def EzConv(in_channel, out_channel, kernel_size):
    return nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1,
                     padding=kernel_size // 2, bias=True)

class Upsample(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(EzConv(n_feats, 4 * n_feats, 3))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(EzConv(n_feats, 9 * n_feats, 3))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsample, self).__init__(*m)

class SSRB(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale, conv=default_conv):
        super(SSRB, self).__init__()
        self.spa = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.spc = ResAttentionBlock(conv, n_feats, 1, act=act, res_scale=res_scale)

    def forward(self, x):
        return self.spc(self.spa(x))


class HFE(nn.Module):
    def __init__(self, n_feats, n_blocks, act, res_scale):
        super(HFE, self).__init__()
        kernel_size = 3
        m = []
        for i in range(n_blocks):
            m.append(SSRB(n_feats, kernel_size, act=act, res_scale=res_scale))
        self.net = nn.Sequential(*m)

    def forward(self, x):
        res = self.net(x)
        res += x
        return res



class BranchUnit(nn.Module):
    def __init__(self, n_colors, n_feats, n_blocks, act, res_scale, up_scale, use_tail=True,conv=default_conv):
        super(BranchUnit, self).__init__()
        kernel_size = 3
        self.head = conv(n_colors, n_feats, kernel_size)
        self.body = HFE(n_feats, n_blocks, act, res_scale)
        self.upsample = Upsample(up_scale, n_feats)
        self.tail = None

        if use_tail:
            self.tail = conv(n_feats, n_colors, kernel_size)

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.upsample(y)
        if self.tail is not None:
            y = self.tail(y)

        return y

class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out = out * x
        return out

class HBSR(nn.Module):
    def __init__(self, n_subs, n_ovls, n_colors, n_blocks, n_feats, n_scale, res_scale, use_share=True):
        super(HBSR, self).__init__()
        #*************************************************************************
        self.branch1 = BranchUnit(n_subs, n_feats, n_blocks, nn.ReLU(True), res_scale, up_scale=2, conv=default_conv)
        self.branch2 = BranchUnit(n_colors, n_feats, n_blocks, nn.ReLU(True), res_scale, up_scale=n_scale//2, use_tail=False, conv=default_conv)
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        CC=30
        self.end_idx = []
        self.sca = 2
        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)
        self.skip_conv = default_conv(n_colors, n_colors, 3)
        self.final = default_conv(n_colors, n_colors, 3)
        #*************************************************************************
        self.Abundance = nn.Sequential(
            nn.Conv2d(n_colors, 8 * CC, kernel_size=3, padding=1),
            SA(),
            torch.nn.Tanh(),
            nn.Conv2d(8 * CC, 4 * CC, kernel_size=3, padding=1),
            SA(),
            torch.nn.Tanh(),
            nn.Conv2d(4 * CC, 2 * CC, kernel_size=3, padding=1),
            SA(),
            torch.nn.Tanh(),
            nn.Conv2d(2 * CC, 1 * CC, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )
        self.SRhead = nn.Conv2d(CC, n_colors, kernel_size=3, padding=1)
        self.endmember = nn.Conv2d(CC, n_colors, kernel_size=1, bias=False)
        self.final1 = default_conv(n_feats, CC, 3)
        self.final2 = default_conv(n_feats, n_colors, 3)

    def forward(self, ms, lms):
        abu = self.Abundance(ms)
        rec_input = self.endmember(abu)
        abu = self.SRhead(abu)

        stack_abu_x = torch.cat([abu, ms], dim=0)
        b, c, h, w = abu.shape
        y = torch.zeros(b*2, c, self.sca * h, self.sca * w).cuda()
        channel_counter = torch.zeros(c).cuda()
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]
            xi = stack_abu_x[:, sta_ind:end_ind, :, :]
            xi = self.branch1(xi)
            y[:, sta_ind:end_ind, :, :] += xi
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1
        y1, y2 = y.chunk(2, dim=0)
        y1 = y1 / channel_counter.unsqueeze(1).unsqueeze(2)
        y1 = self.branch2(y1)
        y1 = self.final1(y1)
        x1 = self.endmember(y1)

        y2 = y2 / channel_counter.unsqueeze(1).unsqueeze(2)
        y2 = self.branch2(y2)
        x2 = self.final2(y2)

        SR = x1 + self.final(x2) + self.skip_conv(lms)
        return SR, rec_input
