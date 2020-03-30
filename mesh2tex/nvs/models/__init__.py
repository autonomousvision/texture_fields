import torch
import torch.nn as nn
from torch import distributions as dist
import trimesh
from mesh2tex.nvs.models import (
    encoder, decoder, discriminator
)

encoder_dict = {
    'resnet18': encoder.Resnet18,
}

decoder_dict = {
    'each_layer_c': decoder.UNetEachLayerC,
}

discriminator_dict = {
    'resnet': discriminator.Resnet,
}


class NovelViewSynthesis(nn.Module):
    def __init__(self, decoder, encoder):
        super().__init__()

        self.decoder = decoder
        self.encoder = encoder

    def forward(self, depth, condition):
        """Generate an image .

        Args:
            depth (torch.FloatTensor): tensor of size B x 1 x N x M
                representing depth of at pixels
        Returns:
            img (torch.FloatTensor): tensor of size B x 3 x N x M representing
                output image
        """
        batch_size, _, N, M = depth.size()

        assert(depth.size(1) == 1)

        c = self.encode(condition)
        img = self.decode(depth, c)

        return img

    def encode(self, cond):
        """Encode mesh using sampled 3D location on the mesh.

        Args:
            input_image (torch.FloatTensor): tensor of size B x 3 x N x M
                input image

        Returns:
            c (torch.FloatTensor): tensor of size B x C with encoding of
                the input image
        """
        z = self.encoder(cond)
        return z

    def decode(self, depth, c):
        """Decode image from 3D locations, conditional encoding and latent
        encoding.

        Args:
            depth (torch.FloatTensor): tensor of size B x 1 x N x M
                representing depth of at pixels
            c (torch.FloatTensor): tensor of size B x C with the encoding of
                the 3D meshes

        Returns:
            rgb (torch.FloatTensor): tensor of size B x 3 x N representing
                color at given 3d locations
        """
        rgb = self.decoder(depth, c)
        return rgb
