import torch
import torch.nn as nn
import torch.nn.functional as F
    
    
class UNetEachLayerC(nn.Module):
    def __init__(self, c_dim, white_bg=True, resnet_leaky=None):
        super().__init__()
        # Attributes
        self.c_dim = c_dim
        self.white_bg = white_bg

        # Submodules
        self.conv_0 = nn.Conv2d(1, 64, 3, padding=1, stride=2)
        self.conv_1 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.conv_2 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.conv_3 = nn.Conv2d(256, 512, 3, padding=1, stride=2)
        self.conv_4 = nn.Conv2d(512, 1024, 3, padding=1, stride=2)

        self.conv_trp_0 = nn.ConvTranspose2d(1024, 512, 3, padding=1, stride=2, output_padding=1)
        self.conv_trp_1 = nn.ConvTranspose2d(1024, 256, 3, padding=1, stride=2, output_padding=1)
        self.conv_trp_2 = nn.ConvTranspose2d(512, 128, 3, padding=1, stride=2, output_padding=1)
        self.conv_trp_3 = nn.ConvTranspose2d(256, 64, 3, padding=1, stride=2, output_padding=1)
        self.conv_trp_4 = nn.ConvTranspose2d(128, 3, 3, padding=1, stride=2, output_padding=1)
    
        self.fc_0 = nn.Linear(c_dim, 64)
        self.fc_1 = nn.Linear(c_dim, 128)
        self.fc_2 = nn.Linear(c_dim, 256)
        self.fc_3 = nn.Linear(c_dim, 512)
        self.fc_4 = nn.Linear(c_dim, 1024)

    def forward(self, depth, c):
        assert(c.size(0) == depth.size(0))

        batch_size = depth.size(0)
        c_dim = self.c_dim

        mask = (depth != float('Inf'))
        depth = depth.clone()
        depth[~mask] = 0.

        net = depth

        # Downsample
        # 64 x 128 x 128
        net0 = self.conv_0(net) + self.fc_0(c).view(batch_size, 64, 1, 1)
        net0 = F.relu(net0)
        # 128 x 64 x 64
        net1 = self.conv_1(net0) + self.fc_1(c).view(batch_size, 128, 1, 1)
        net1 = F.relu(net1)
        # 256 x 32 x 32
        net2 = self.conv_2(net1) + self.fc_2(c).view(batch_size, 256, 1, 1)
        net2 = F.relu(net2)
        # 512 x 16 x 16
        net3 = self.conv_3(net2) + self.fc_3(c).view(batch_size, 512, 1, 1)
        net3 = F.relu(net3)
        # 1024 x 8 x 8
        net4 = self.conv_4(net3) + self.fc_4(c).view(batch_size, 1024, 1, 1)
        net4 = F.relu(net4)    
        
        # Upsample
         # 512 x 16 x 16
        net = F.relu(self.conv_trp_0(net4))
        # 256 x 32 x 32
        net = torch.cat([net, net3], dim=1)
        net = F.relu(self.conv_trp_1(net))
        # 128 x 64 x 64
        net = torch.cat([net, net2], dim=1)
        net = F.relu(self.conv_trp_2(net))
        # 64 x 128 x 128
        net = torch.cat([net, net1], dim=1)
        net = F.relu(self.conv_trp_3(net))
        # 3 x 256 x 256
        net = torch.cat([net, net0], dim=1)
        net = self.conv_trp_4(net) 
        net = torch.sigmoid(net)

        if self.white_bg:
            mask = mask.float()
            net = mask * net + (1 - mask) * torch.ones_like(net)
        else:
            mask = mask.float()
            net = mask * net + (1 - mask) * torch.zeros_like(net)

        return net