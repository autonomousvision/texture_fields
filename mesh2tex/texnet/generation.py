import torch
import numpy as np
import os
from trimesh.util import array_to_string
from mesh2tex import geometry
from torchvision.utils import save_image
from torch.nn.functional import interpolate

#TODO comment the generation functions


class Generator3D(object):
    def __init__(self, model, device=None):

        self.model = model
        self.device = device

    def save_mesh(self, mesh, out_file, digits=10):
        '''
        Saving meshes to OFF file

        '''
        digits = int(digits)
        # prepend a 3 (face count) to each face
        if mesh.visual.face_colors is None:
            faces_stacked = np.column_stack((
                np.ones(len(mesh.faces)) * 3,
                mesh.faces)).astype(np.int64)
        else:
            assert(mesh.visual.face_colors.shape[0] == mesh.faces.shape[0])
            faces_stacked = np.column_stack((
                np.ones(len(mesh.faces)) * 3,
                mesh.faces, mesh.visual.face_colors[:, :3])).astype(np.int64)
        export = 'OFF\n'
        # the header is vertex count, face count, edge number
        export += str(len(mesh.vertices)) + ' ' + str(len(mesh.faces)) + ' 0\n'
        export += array_to_string(
            mesh.vertices, col_delim=' ', row_delim='\n', digits=digits) + '\n'
        export += array_to_string(faces_stacked, col_delim=' ', row_delim='\n')

        with open(out_file, 'w') as f:
            f.write(export)

        return mesh

    def generate_images_4eval_condi(self, batch, out_dir, model_names):
        '''
        Generate textures in the conditional setting (given image)

        '''

        # Extract depth, gt, camera info, shape pc and condition
        depth = batch['2d.depth'].to(self.device)
        img_real = batch['2d.img'].to(self.device)
        cam_K = batch['2d.camera_mat'].to(self.device)
        cam_W = batch['2d.world_mat'].to(self.device)
        mesh_repr = geometry.get_representation(batch, self.device)
        mesh_points = mesh_repr['points']
        mesh_normals = mesh_repr['normals']
        condition = batch['condition'].to(self.device)

        # Determine constants and check
        batch_size = depth.size(0)
        num_views = depth.size(1)

        # Define Output folders
        out_dir_real = out_dir + "/real/"
        out_dir_fake = out_dir + "/fake/"
        out_dir_condition = out_dir + "/condition/"
        if not os.path.exists(out_dir_real):
            os.makedirs(out_dir_real)
        if not os.path.exists(out_dir_fake):
            os.makedirs(out_dir_fake)
        if not os.path.exists(out_dir_condition):
            os.makedirs(out_dir_condition)

        # Batch loop
        for j in range(batch_size):
            
            # Expand shape info to tensors
            # for all views of the same objects
            geom_repr = {
                'points': mesh_points[j][:num_views].expand(
                    num_views, mesh_points.size(1),
                    mesh_points.size(2)),
                'normals': mesh_normals[j][:num_views].expand(
                    num_views, mesh_normals.size(1),
                    mesh_normals.size(2)),
            }

            depth_ = depth[j][:num_views]
            img_real_ = img_real[j][:num_views]
            condition_ = condition[j][:num_views].expand(
                num_views, condition.size(1),
                condition.size(2), condition.size(3))
            cam_K_ = cam_K[j][:num_views]
            cam_W_ = cam_W[j][:num_views]

            # Generate images and save
            self.model.eval()
            with torch.no_grad():
                img_fake = self.model(depth_, cam_K_, cam_W_,
                                      geom_repr, condition_)

            save_image(
                condition[j].cpu(),
                os.path.join(out_dir_condition,
                             '%s.png' % (model_names[j])))

            for v in range(num_views):
                save_image(
                    img_real_[v],
                    os.path.join(out_dir_real,
                                 '%s%03d.png' % (model_names[j], v)))
                save_image(
                    img_fake[v].cpu(),
                    os.path.join(out_dir_fake,
                                 '%s%03d.png' % (model_names[j], v)))

    def generate_images_4eval_condi_hd(self, batch, out_dir, model_names):
        '''
        Generate textures in hd images given condition

        '''

        # Extract depth, gt, camera info, shape pc and condition
        depth = batch['2d.depth']
        img_real = batch['2d.img']
        cam_K = batch['2d.camera_mat']
        cam_W = batch['2d.world_mat']
        mesh_repr = geometry.get_representation(batch, self.device)
        mesh_points = mesh_repr['points']
        mesh_normals = mesh_repr['normals']        
        condition = batch['condition']

        # Determine constants and check
        batch_size = depth.size(0)
        num_views = depth.size(1)

        # Define Output folders
        out_dir_real = out_dir + "/real/"
        out_dir_fake = out_dir + "/fake/"
        out_dir_condition = out_dir + "/condition/"
        if not os.path.exists(out_dir_real):
            os.makedirs(out_dir_real)
        if not os.path.exists(out_dir_fake):
            os.makedirs(out_dir_fake)
        if not os.path.exists(out_dir_condition):
            os.makedirs(out_dir_condition)
        
        # Loop through batch and views, because of memory requirement
        viewbatchsize = 1
        viewbatchnum = int(num_views / viewbatchsize)
        for j in range(batch_size):
            for vidx in range(viewbatchnum):
                lower = vidx * viewbatchsize
                upper = (vidx + 1) * viewbatchsize

                # Expand shape info to tensors
                # for all views of the same objects
                geom_repr = {
                    'points': mesh_points[j][:4].expand(
                        viewbatchsize, mesh_points.size(1),
                        mesh_points.size(2)),
                    'normals': mesh_normals[j][:4].expand(
                        viewbatchsize, mesh_normals.size(1),
                        mesh_normals.size(2)),
                }

                depth_ = depth[j][lower:upper].to(self.device)
                img_real_ = img_real[j][lower:upper]
                if len(condition.size()) == 1:
                    condition_ = condition[j:j+1].expand(
                                viewbatchsize)
                else:
                    condition_ = condition[j:j+1][:4].expand(
                        viewbatchsize, condition.size(1),
                        condition.size(2), condition.size(3)).to(self.device)
                cam_K_ = cam_K[j][lower:upper].to(self.device)
                cam_W_ = cam_W[j][lower:upper].to(self.device)

                # Generate images and save
                self.model.eval()
                with torch.no_grad():
                    img_fake = self.model(depth_, cam_K_, cam_W_,
                                          geom_repr, condition_)
                if len(condition.size()) != 1:
                    save_image(
                        condition[j].cpu(),
                        os.path.join(out_dir_condition,
                                     '%s.png' % (model_names[j])))

                for v in range(viewbatchsize):
                    save_image(
                        img_real_[v],
                        os.path.join(
                            out_dir_real,
                            '%s%03d.png' % (model_names[j],
                                            vidx * viewbatchsize + v)))
                    save_image(
                        img_fake[v].cpu(),
                        os.path.join(
                            out_dir_fake,
                            '%s%03d.png' % (model_names[j],
                                            vidx * viewbatchsize + v)))

    def generate_images_4eval_vae(self, batch, out_dir, model_names):
        '''
        Generate texture using the VAE

        '''
        # Extract depth, gt, camera info, shape pc and condition
        depth = batch['2d.depth'].to(self.device)
        img_real = batch['2d.img'].to(self.device)
        cam_K = batch['2d.camera_mat'].to(self.device)
        cam_W = batch['2d.world_mat'].to(self.device)
        mesh_repr = geometry.get_representation(batch, self.device)
        mesh_points = mesh_repr['points']
        mesh_normals = mesh_repr['normals']

        # Determine constants and check
        batch_size = depth.size(0)
        num_views = depth.size(1)
        if depth.size(1) >= 10:
            num_views = 10

        # Define Output folders
        out_dir_real = out_dir + "/real/"
        out_dir_fake = out_dir + "/fake/"
        if not os.path.exists(out_dir_real):
            os.makedirs(out_dir_real)
        if not os.path.exists(out_dir_fake):
            os.makedirs(out_dir_fake)

        # batch loop
        for j in range(batch_size):
            geom_repr = {
                'points': mesh_points[j][:num_views].expand(
                    num_views, mesh_points.size(1), mesh_points.size(2)),
                'normals': mesh_normals[j][:num_views].expand(
                    num_views, mesh_normals.size(1), mesh_normals.size(2)),
            }
            depth_ = depth[j][:num_views]
            img_real_ = img_real[j][:num_views]
            cam_K_ = cam_K[j][:num_views]
            cam_W_ = cam_W[j][:num_views]

            # Sample latent code
            z_ = np.random.normal(0, 1, 512)
            inter = torch.from_numpy(z_).float().to(self.device)
            z = inter.expand(num_views, 512)

            # Generate images and save
            self.model.eval()
            with torch.no_grad():
                img_fake = self.model(depth_, cam_K_, cam_W_,
                                      geom_repr, z=z, sample=False)

            for v in range(num_views):
                save_image(
                    img_real_[v],
                    os.path.join(out_dir_real, '%s%03d.png'
                                 % (model_names[j], v)))
                save_image(
                    img_fake[v].cpu(),
                    os.path.join(out_dir_fake, '%s%03d.png'
                                 % (model_names[j], v)))

    def generate_images_4eval_vae_interpol(self, batch, out_dir, model_names):
        '''
        Interpolates between latent encoding 
        of first and second element of batch 

        '''
        # Extract depth, gt, camera info, shape pc and condition
        depth = batch['2d.depth'].to(self.device)
        img_real = batch['2d.img'].to(self.device)
        cam_K = batch['2d.camera_mat'].to(self.device)
        cam_W = batch['2d.world_mat'].to(self.device)
        mesh_repr = geometry.get_representation(batch, self.device)
        mesh_points = mesh_repr['points']
        mesh_normals = mesh_repr['normals']

        # Determine constants and check
        batch_size = depth.size(0)
        num_views = depth.size(1)
        if depth.size(1) >= 10:
            num_views = 10

        # Define Output folders
        out_dir_real = out_dir + "/real/"
        out_dir_fake = out_dir + "/fake/"
        if not os.path.exists(out_dir_real):
            os.makedirs(out_dir_real)
        if not os.path.exists(out_dir_fake):
            os.makedirs(out_dir_fake)

        # Derive latent texture code as starting point of interpolation
        geom_repr = {
            'points': mesh_points[:1],
            'normals': mesh_normals[:1],
        }
        self.model.eval()
        shape_encoding = self.model.encode_geometry(geom_repr)
        image_input = img_real[0][:1]
        img = interpolate(image_input, size=[128, 128])
        latent_input = self.model.infer_z_transfer(img, shape_encoding)

        # Derive latent texture code as end point of interpolation
        geom_repr2 = {
            'points': mesh_points[1:2],
            'normals': mesh_normals[1:2],
        }
        shape_encoding2 = self.model.encode_geometry(geom_repr2)
        image_input2 = img_real[1][:1]
        img2 = interpolate(image_input2, size=[128, 128])
        latent_input2 = self.model.infer_z_transfer(img2, shape_encoding2)

        # Derive stepsize
        steps = 20
        step = (latent_input2-latent_input)/steps

        # batch loop
        for j in range(1, batch_size):
            
            geom_repr = {
                'points': mesh_points[j][:num_views].expand(
                    num_views, mesh_points.size(1), mesh_points.size(2)),
                'normals': mesh_normals[j][:num_views].expand(
                    num_views, mesh_normals.size(1), mesh_normals.size(2)),
            }

            depth_ = depth[j][:num_views]
            img_real_ = img_real[j][:num_views]
            cam_K_ = cam_K[j][:num_views]
            cam_W_ = cam_W[j][:num_views]
            
            self.model.eval()
            # steps loop
            for num in range(steps):
                inter = latent_input + step*num
                z = inter.expand(num_views, 512)
                with torch.no_grad():
                    img_fake = self.model(depth_, cam_K_, cam_W_,
                                          geom_repr, z=z, sample=False)
                for v in range(1):
                    save_image(
                        img_real_[v],
                        os.path.join(
                            out_dir_real, '%s%03d_%03d.png'
                            % (model_names[j], v, num)))
                    save_image(
                        img_fake[v].cpu(),
                        os.path.join(
                            out_dir_fake, '%s%03d_%03d.png'
                            % (model_names[j], v, num)))

    def generate_images_4eval_gan(self, batch, out_dir, model_names):
        '''
        Generate Texture  using a GAN

        '''
        # Extract depth, gt, camera info, shape pc and condition
        depth = batch['2d.depth'].to(self.device)
        img_real = batch['2d.img'].to(self.device)
        cam_K = batch['2d.camera_mat'].to(self.device)
        cam_W = batch['2d.world_mat'].to(self.device)
        mesh_repr = geometry.get_representation(batch, self.device)
        mesh_points = mesh_repr['points']
        mesh_normals = mesh_repr['normals']

        # Determine constants and check
        batch_size = depth.size(0)
        num_views = depth.size(1)
        if depth.size(1) >= 10:
            num_views = 10

        # Define Output folders
        out_dir_real = out_dir + "/real/"
        out_dir_fake = out_dir + "/fake/"
        out_dir_condition = out_dir + "/condition/"
        if not os.path.exists(out_dir_real):
            os.makedirs(out_dir_real)
        if not os.path.exists(out_dir_fake):
            os.makedirs(out_dir_fake)
        if not os.path.exists(out_dir_condition):
            os.makedirs(out_dir_condition)

        # batch loop
        for j in range(batch_size):
            
            geom_repr = {
                'points': mesh_points[j][:num_views].expand(
                    num_views, mesh_points.size(1),
                    mesh_points.size(2)),
                'normals': mesh_normals[j][:num_views].expand(
                    num_views, mesh_normals.size(1),
                    mesh_normals.size(2)),
            }

            depth_ = depth[j][:num_views]
            img_real_ = img_real[j][:num_views]
            cam_K_ = cam_K[j][:num_views]
            cam_W_ = cam_W[j][:num_views]

            self.model.eval()
            with torch.no_grad():
                img_fake = self.model(depth_, cam_K_, cam_W_,
                                      geom_repr, sample=False)
            for v in range(num_views):
                save_image(
                    img_real_[v],
                    os.path.join(
                        out_dir_real, '%s%03d.png' % (model_names[j], v)))
                save_image(
                    img_fake[v].cpu(),
                    os.path.join(
                        out_dir_fake, '%s%03d.png' % (model_names[j], v)))


def make_3d_grid(bb_min, bb_max, shape):
    '''
    Outputs gird points of a 3d grid

    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p
