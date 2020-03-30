import torch
import numpy as np
import trimesh
import os
from trimesh.util import array_to_string
from mesh2tex import geometry
from torchvision.utils import save_image


class Generator3D(object):
    def __init__(self, model, device=None):

        self.model = model
        self.device = device

    def generate_images_4eval(self, batch, out_dir, model_names):
        depth = batch['2d.depth'].to(self.device)
        img_real = batch['2d.img'].to(self.device)
        condition = batch['condition'].to(self.device)
        batch_size = depth.size(0)
        num_views = depth.size(1)
        # assert(num_views == 5)
        # Save real images
        
        out_dir_real = out_dir + "/real/"
        out_dir_fake = out_dir + "/fake/"
        out_dir_condition = out_dir + "/condition/"
        if not os.path.exists(out_dir_real):
            os.makedirs(out_dir_real)
        if not os.path.exists(out_dir_fake):
            os.makedirs(out_dir_fake)
        if not os.path.exists(out_dir_condition):
            os.makedirs(out_dir_condition)

        for j in range(batch_size):
            save_image(
                condition[j].cpu(),
                os.path.join(out_dir_condition, '%s.png' % model_names[j]))
                
        for v in range(num_views):
            depth_ = depth[:, v]
            img_real_ = img_real[:, v]
    
            self.model.eval()

            with torch.no_grad():
                img_fake_ = self.model(depth_, condition)
                
            for j in range(batch_size):
                save_image(
                    img_real_[j].cpu(),
                    os.path.join(
                        out_dir_real, '%s%03d.png' % (model_names[j], v)
                    ))
                save_image(
                    img_fake_[j].cpu(),
                    os.path.join(
                        out_dir_fake, '%s%03d.png' % (model_names[j], v)
                    ))

    def generate_images_4eval_condi_hd(self, batch, out_dir, model_names):
        depth = batch['2d.depth'] #.to(self.device)
        img_real = batch['2d.img'] #.to(self.device)
        condition = batch['condition'].to(self.device)
        batch_size = depth.size(0)
        num_views = depth.size(1)
        # if depth.size(1) >= 10:
        #     num_views = 10
        # assert(num_views == 5)
        # Save real images
        out_dir_real = out_dir + "/real/"
        out_dir_fake = out_dir + "/fake/"
        out_dir_condition = out_dir + "/condition/"
        if not os.path.exists(out_dir_real):
            os.makedirs(out_dir_real)
        if not os.path.exists(out_dir_fake):
            os.makedirs(out_dir_fake)
        if not os.path.exists(out_dir_condition):
            os.makedirs(out_dir_condition)
        viewbatchsize = 2
        viewbatchnum = int(num_views / viewbatchsize)
        # points_batches = points.split(10, dim=0)
        for j in range(batch_size):
            for vidx in range(viewbatchnum):
                lower = vidx * viewbatchsize
                upper = (vidx + 1) * viewbatchsize

                depth_ = depth[j][lower:upper]
                img_real_ = img_real[j][lower:upper]
                condition_ = condition[j][:4].expand(
                        viewbatchsize, condition.size(1),
                    condition.size(2), condition.size(3))

                self.model.eval()
                with torch.no_grad():
                    img_fake = self.model(depth_.to(self.device), condition_.to(self.device))
                for v in range(viewbatchsize):
                    save_image(
                        img_real_[v],
                        os.path.join(out_dir_real,
                                    '%s%03d.png' % (model_names[j], vidx * viewbatchsize + v)))
                    save_image(
                        img_fake[v].cpu(),
                        os.path.join(out_dir_fake,
                                    '%s%03d.png' % (model_names[j], vidx * viewbatchsize + v)))
                save_image(
                    condition[j].cpu(),
                    os.path.join(out_dir_condition,
                                '%s.png' % (model_names[j])))
