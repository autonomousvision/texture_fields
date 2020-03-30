import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mesh2tex.layers import ResnetBlockConv2d, EqualizedLR


class Resnet(nn.Module):
    def __init__(self, img_size, z_dim=128, c_dim=128, embed_size=256,
                 nfilter=32, nfilter_max=1024, leaky=True, eq_lr=False):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.eq_lr = eq_lr
        self.c_dim = c_dim

        # Activation function
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        # Submodules
        nlayers = int(np.log2(img_size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [
            ResnetBlockConv2d(
                nf, nf, actvn=self.actvn, eq_lr=eq_lr)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlockConv2d(
                    nf0, nf1, actvn=self.actvn, eq_lr=eq_lr),
            ]

        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc_mean = nn.Linear(self.nf0*s0*s0, z_dim)
        self.fc_logstd = nn.Linear(self.nf0*s0*s0, z_dim)
        self.fc_inject_c = nn.Linear(self.c_dim, 1*nf)
        if self.eq_lr:
            self.conv_img = EqualizedLR(self.conv_img)
            self.fc = EqualizedLR(self.fc)

    def forward(self, x, geom_descr):
        c = geom_descr['global']
        batch_size = x.size(0)

        out = self.conv_img(x)
        add = self.fc_inject_c(c).view(out.size(0), self.nf, 1, 1)
        out = out + add
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0*self.s0*self.s0)

        mean = self.fc_mean(self.actvn(out))
        logstd = self.fc_logstd(self.actvn(out))
        return mean, logstd
