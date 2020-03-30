import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mesh2tex.layers import ResnetBlockConv2d, EqualizedLR, pixel_norm


class Resnet_Conditional(nn.Module):
    def __init__(self, geometry_encoder, img_size, c_dim=128, embed_size=256,
                 nfilter=64, nfilter_max=1024,
                 leaky=True, eq_lr=False, pixel_norm=False,
                 factor=1.):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.eq_lr = eq_lr
        self.use_pixel_norm = pixel_norm

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
                nf, nf, actvn=self.actvn,
                eq_lr=eq_lr,
                factor=factor,
                pixel_norm=pixel_norm)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlockConv2d(
                    nf0, nf1, actvn=self.actvn, eq_lr=eq_lr,
                    factor=factor,
                    pixel_norm=pixel_norm),
            ]

        self.conv_img = nn.Conv2d(4, 1*nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0*s0*s0, 1)

        if self.eq_lr:
            self.conv_img = EqualizedLR(self.conv_img)
            self.fc = EqualizedLR(self.fc)

        # Initialization
        nn.init.zeros_(self.fc.weight)

    def forward(self, x, depth, geom_descr):
        batch_size = x.size(0)

        depth = depth.clone()
        depth[depth == float("Inf")] = 0
        depth[depth == -1*float("Inf")] = 0
        
        x_and_depth = torch.cat([x, depth], dim=1)

        out = self.conv_img(x_and_depth)
        out = self.resnet(out)

        if self.use_pixel_norm:
            out = pixel_norm(out)
        out = out.view(batch_size, self.nf0*self.s0*self.s0)
        out = self.fc(self.actvn(out))
        out = out.squeeze()
        return out
