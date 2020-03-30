import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import trimesh
from mesh2tex.texnet.models import (
    image_encoder, decoder, discriminator, vae_encoder
)

encoder_dict = {
    'resnet18': image_encoder.Resnet18,
}

decoder_dict = {
    'each_layer_c': decoder.DecoderEachLayerC,
    'each_layer_c_larger': decoder.DecoderEachLayerCLarger,
}

discriminator_dict = {
    'resnet_conditional': discriminator.Resnet_Conditional,
}

vae_encoder_dict = {
    'resnet': vae_encoder.Resnet,
}


class TextureNetwork(nn.Module):
    def __init__(self, decoder, geometry_encoder, encoder=None,
                 vae_encoder=None, p0_z=None, white_bg=True):
        super().__init__()
        
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.decoder = decoder
        self.encoder = encoder
        self.geometry_encoder = geometry_encoder
        self.vae_encoder = vae_encoder
        self.p0_z = p0_z
        self.white_bg = white_bg

    def forward(self, depth, cam_K, cam_W, geometry,
                condition=None, z=None, sample=True):
        """Generate an image .

        Args:
            depth (torch.FloatTensor): tensor of size B x 1 x N x M
                representing depth of at pixels
            cam_K (torch.FloatTensor): tensor of size B x 3 x 4 representing
                camera projectin matrix
            cam_W (torch.FloatTensor): tensor of size B x 3 x 4 representing
                camera world matrix
            geometry (dict): representation of geometry
            condition
            z
            sample (Boolean): wether to sample latent code or take MAP
        Returns:
            img (torch.FloatTensor): tensor of size B x 3 x N x M representing
                output image
        """
        batch_size, _, N, M = depth.size()
        assert(depth.size(1) == 1)
        assert(cam_K.size() == (batch_size, 3, 4))
        assert(cam_W.size() == (batch_size, 3, 4))

        loc3d, mask = self.depth_map_to_3d(depth, cam_K, cam_W)
        geom_descr = self.encode_geometry(geometry)

        if self.encoder is not None:
            z = self.encode(condition)
            z = z.cuda()
        elif z is None:
            z = self.get_z_from_prior((batch_size,), sample=sample)

        loc3d = loc3d.view(batch_size, 3, N * M)
        x = self.decode(loc3d, geom_descr, z)
        x = x.view(batch_size, 3, N, M)

        if self.white_bg is False:
            x_bg = torch.zeros_like(x)
        else:
            x_bg = torch.ones_like(x)

        img = (mask * x).permute(0, 1, 3, 2) + (1 - mask.permute(0, 1, 3, 2)) * x_bg

        return img

    def load_mesh2facecenter(in_path):
        mesh = trimesh.load(in_path, process=False)
        faces_center = mesh.triangles_center
        return mesh, faces_center

    def elbo(self, image_real, depth, cam_K, cam_W, geometry):
        batch_size, _, N, M = depth.size()

        assert(depth.size(1) == 1)
        assert(cam_K.size() == (batch_size, 3, 4))
        assert(cam_W.size() == (batch_size, 3, 4))

        loc3d, mask = self.depth_map_to_3d(depth, cam_K, cam_W)
        geom_descr = self.encode_geometry(geometry)

        q_z = self.infer_z(image_real, geom_descr)
        z = q_z.rsample()

        loc3d = loc3d.view(batch_size, 3, N * M)
        x = self.decode(loc3d, geom_descr, z)
        x = x.view(batch_size, 3, N, M)

        if self.white_bg is False:
            x_bg = torch.zeros_like(x)
        else:
            x_bg = torch.ones_like(x)

        image_fake = (mask * x).permute(0, 1, 3, 2) + (1 - mask.permute(0, 1, 3, 2)) * x_bg

        recon_loss = F.mse_loss(image_fake, image_real).sum(dim=-1)
        kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
        elbo = recon_loss.mean() + kl.mean()/float(N*M*3)
        return elbo, recon_loss.mean(), kl.mean()/float(N*M*3), image_fake

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

    def encode_geometry(self, geometry):
        """Encode mesh using sampled 3D location on the mesh.

        Args:
            geometry (dict): representation of teometry
        Returns:
            geom_descr (dict): geometry discriptor

        """
        geom_descr = self.geometry_encoder(geometry)
        return geom_descr

    def decode(self, loc3d, c, z):
        """Decode image from 3D locations, conditional encoding and latent
        encoding.

        Args:
            loc3d (torch.FloatTensor): tensor of size B x 3 x K
                with 3D locations of the query
            c (torch.FloatTensor): tensor of size B x C with the encoding of
                the 3D meshes
            z (torch.FloatTensor): tensor of size B x Z with latent codes

        Returns:
            rgb (torch.FloatTensor): tensor of size B x 3 x N representing
                color at given 3d locations
        """
        rgb = self.decoder(loc3d, c, z)
        return rgb

    def depth_map_to_3d(self, depth, cam_K, cam_W):
        """Derive 3D locations of each pixel of a depth map.

        Args:
            depth (torch.FloatTensor): tensor of size B x 1 x N x M
                with depth at every pixel
            cam_K (torch.FloatTensor): tensor of size B x 3 x 4 representing
                camera matrices
            cam_W (torch.FloatTensor): tensor of size B x 3 x 4 representing
                world matrices
        Returns:
            loc3d (torch.FloatTensor): tensor of size B x 3 x N x M
                representing color at given 3d locations
            mask (torch.FloatTensor):  tensor of size B x 1 x N x M with
                a binary mask if the given pixel is present or not
        """
       
        assert(depth.size(1) == 1)
        batch_size, _, N, M = depth.size()
        device = depth.device
        # Turn depth around. This also avoids problems with inplace operations
        depth = -depth .permute(0, 1, 3, 2)
        
        zero_one_row = torch.tensor([[0., 0., 0., 1.]])
        zero_one_row = zero_one_row.expand(batch_size, 1, 4).to(device)

        # add row to world mat
        cam_W = torch.cat((cam_W, zero_one_row), dim=1)

        # clean depth image for mask
        mask = (depth.abs() != float("Inf")).float()
        depth[depth == float("Inf")] = 0
        depth[depth == -1*float("Inf")] = 0

        # 4d array to 2d array k=N*M
        d = depth.reshape(batch_size, 1, N * M)

        # create pixel location tensor
        px, py = torch.meshgrid([torch.arange(0, N), torch.arange(0, M)])
        px, py = px.to(device), py.to(device)

        p = torch.cat((
            px.expand(batch_size, 1, px.size(0), px.size(1)), 
            (M - py).expand(batch_size, 1, py.size(0), py.size(1))
        ), dim=1)
        p = p.reshape(batch_size, 2, py.size(0) * py.size(1))
        p = (p.float() / M * 2)      
        
        # create terms of mapping equation x = P^-1 * d*(qp - b)
        P = cam_K[:, :2, :2].float().to(device)    
        q = cam_K[:, 2:3, 2:3].float().to(device)   
        b = cam_K[:, :2, 2:3].expand(batch_size, 2, d.size(2)).to(device)
        Inv_P = torch.inverse(P).to(device)   

        rightside = (p.float() * q.float() - b.float()) * d.float()
        x_xy = torch.bmm(Inv_P, rightside)
        
        # add depth and ones to location in world coord system
        x_world = torch.cat((x_xy, d, torch.ones_like(d)), dim=1)

        # derive loactoion in object coord via loc3d = W^-1 * x_world
        Inv_W = torch.inverse(cam_W)
        loc3d = torch.bmm(
            Inv_W.expand(batch_size, 4, 4),
            x_world
        ).reshape(batch_size, 4, N, M)

        loc3d = loc3d[:, :3].to(device)
        mask = mask.to(device)
        return loc3d, mask

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        """Draw latent code z from prior either using sampling or
        using the MAP.

        Args:
            size (torch.Size): size of sample to draw.
            sample (Boolean): wether to sample or to use the MAP

        Return:
            z (torch.FloatTensor): tensor of shape *size x Z representing
                the latent code
        """
        if sample:
            z = self.p0_z.sample(size)
        else:
            z = self.p0_z.mean
            z = z.expand(*size, *z.size())

        return z

    def infer_z(self, image, c, **kwargs):
        if self.vae_encoder is not None:
            mean_z, logstd_z = self.vae_encoder(image, c, **kwargs)
        else:
            batch_size = image.size(0)
            mean_z = torch.empty(batch_size, 0).to(self._device)
            logstd_z = torch.empty(batch_size, 0).to(self._device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z

    def infer_z_transfer(self, image, c, **kwargs):
        if self.vae_encoder is not None:
            mean_z, logstd_z = self.vae_encoder(image, c, **kwargs)
        else:
            batch_size = image.size(0)
            mean_z = torch.empty(batch_size, 0).to(self._device)
        return mean_z
