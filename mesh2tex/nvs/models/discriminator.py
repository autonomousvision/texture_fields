import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mesh2tex.layers import ResnetBlockConv2d


class Resnet(nn.Module):
    def __init__(self, img_size, embed_size=256,
                 nfilter=64, nfilter_max=1024, leaky=True):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        # Activation function
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        # Submodules
        nlayers = int(np.log2(img_size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [
            ResnetBlockConv2d(nf, nf, actvn=self.actvn)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlockConv2d(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3 +1 , 1*nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0*s0*s0, 1)

        # Initialization
        nn.init.zeros_(self.fc.weight)

    def forward(self, x, depth):
        batch_size = x.size(0)

        depth = depth.clone()
        depth[depth == float("Inf")] = 0
        depth[depth == -1*float("Inf")] = 0
        
        x_and_depth = torch.cat([x, depth], dim=1)

        out = self.conv_img(x_and_depth)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0*self.s0*self.s0)
        out = self.fc(self.actvn(out))
        out = out.squeeze()

        return out
