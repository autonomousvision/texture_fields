
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from mesh2tex import data, geometry
from mesh2tex.texnet import training, generation
from mesh2tex.texnet import models


def get_models(cfg, dataset=None, device=None):
    # Get configs
    encoder = cfg['model']['encoder']
    decoder = cfg['model']['decoder']
    geometry_encoder = cfg['model']['geometry_encoder']
    vae_encoder = cfg['model']['vae_encoder']
    discriminator = cfg['model']['discriminator']

    encoder_kwargs = cfg['model']['encoder_kwargs']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    geometry_encoder_kwargs = cfg['model']['geometry_encoder_kwargs']
    discriminator_kwargs = cfg['model']['discriminator_kwargs']
    vae_encoder_kwargs = cfg['model']['vae_encoder_kwargs']
    img_size = cfg['data']['img_size']
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']
    white_bg = cfg['model']['white_bg']
    # Create generator

    if encoder == "idx":
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = models.encoder_dict[encoder](
            c_dim=c_dim, **encoder_kwargs
        ).to(device)

    decoder = models.decoder_dict[decoder](
        c_dim=c_dim, z_dim=z_dim, **decoder_kwargs
    ).to(device)

    geometry_encoder = geometry.encoder_dict[geometry_encoder](
        c_dim=c_dim, **geometry_encoder_kwargs
    ).to(device)

    if vae_encoder is not None:
        vae_encoder = models.vae_encoder_dict[vae_encoder](
            img_size=img_size, c_dim=c_dim, z_dim=z_dim, **vae_encoder_kwargs
        ).to(device)

    p0_z = get_prior_z(cfg, device)

    generator = models.TextureNetwork(
        decoder, geometry_encoder, encoder, vae_encoder, p0_z, white_bg
    )

    # Create discriminator
    discriminator = models.discriminator_dict[discriminator](
        geometry_encoder,
        img_size=img_size, **discriminator_kwargs
    ).to(device)

    # Output dict
    models_out = {
        'generator': generator,
        'discriminator': discriminator,
    }

    return models_out


def get_optimizers(models, cfg):
    model_g = models['generator']
    model_d = models['discriminator']

    lr_g = cfg['training']['lr_g']
    lr_d = cfg['training']['lr_d']
    optimizer_g = optim.RMSprop(model_g.parameters(), lr=lr_g)
    optimizer_d = optim.RMSprop(model_d.parameters(), lr=lr_d)

    optimizers = {
        'generator': optimizer_g,
        'discriminator': optimizer_d,
    }
    return optimizers


def get_dataset(mode, cfg, input_sampling=True):
    # Config
    path_shapes = cfg['data']['path_shapes']
    img_size = cfg['data']['img_size']
    pc_subsampling = cfg['training']['pc_subsampling']
    pcl_knn = cfg['data']['pcl_knn']
    
    # Fields
    transform_img = transforms.Compose([
        data.ResizeImage((img_size, img_size), order=0),
    ])

    transform_img_input = transforms.Compose([
        data.ResizeImage((224, 224), order=0),
    ])

    transform_depth = torchvision.transforms.Compose([
        data.ImageToDepthValue(),
        data.ResizeImage((img_size, img_size), order=0),
    ])

    pcl_transform = [data.SubsamplePointcloud(pc_subsampling)]
    if pcl_knn is not None:
        pcl_transform += [data.ComputeKNNPointcloud(pcl_knn)]

    pcl_transform = transforms.Compose(pcl_transform)

    if mode == 'train':
        fields = {
            '2d': data.DepthImageField(
                'image', 'depth', transform_img, transform_depth, 'png',
                'exr', with_camera=True, random_view=True),
            'pointcloud': data.PointCloudField('pointcloud.npz', pcl_transform),
            'condition': data.ImagesField('input_image',
                                          transform_img_input, 'jpg'),
        }
        mode_ = 'train'

    elif mode == 'val_eval':
        fields = {
            '2d': data.DepthImageField(
                'image', 'depth', transform_img, transform_depth, 'png',
                'exr', with_camera=True, random_view=True),
            'pointcloud': data.PointCloudField('pointcloud.npz', pcl_transform),
            'condition': data.ImagesField('input_image',
                                          transform_img_input, 'jpg'),
        }
        mode_ = 'val'

    elif mode == 'val_vis':
        fields = {
            '2d': data.DepthImageVisualizeField(
                'visualize/image', 'visualize/depth', 
                transform_img, transform_depth, 'png',
                'exr', with_camera=True, random_view=True
                ),
            'pointcloud': data.PointCloudField('pointcloud.npz', pcl_transform),
            'condition': data.ImagesField('input_image',
                                          transform_img_input, 'jpg'),
        }
        mode_ = 'val'

    elif mode == 'test_eval':
        fields = {
            '2d': data.DepthImageVisualizeField(
                'image', 'depth', 
                transform_img, transform_depth, 'png',
                'exr', with_camera=True, random_view=True
                ),
            'pointcloud': data.PointCloudField('pointcloud.npz', pcl_transform),
            'condition': data.ImagesField('input_image',
                                          transform_img_input, 'jpg',
                                          random_view=input_sampling),
            'idx': data.IndexField(),
        }
        mode_ = 'test'

    elif mode == 'test_vis':
        fields = {
            '2d': data.DepthImageVisualizeField(
                'visualize/image', 'visualize/depth', 
                transform_img, transform_depth, 'png',
                'exr', with_camera=True, random_view=True
                ),
            'pointcloud': data.PointCloudField('pointcloud.npz', pcl_transform),
            'condition': data.ImagesField('input_image',
                                          transform_img_input, 'jpg',
                                          random_view=input_sampling),
            'idx': data.IndexField(),
        }
        mode_ = 'test'

    else:
        print('Invalid data loading mode')

    # Dataset
    if cfg['data']['shapes_multiclass']:
        ds_shapes = data.Shapes3dDataset(
            path_shapes, fields, split=mode_, no_except=True,
        )
    else:
        ds_shapes = data.Shapes3dClassDataset(
            path_shapes, fields, split=mode_, no_except=False,
        )

    if mode_ == 'val' or mode_ == 'test':
        ds = ds_shapes
    else:
        ds = data.CombinedDataset([ds_shapes, ds_shapes])

    return ds


def get_dataloader(mode, cfg):
    # Config
    batch_size = cfg['training']['batch_size']
    with_shuffle = cfg['data']['with_shuffle']

    ds_shapes = get_dataset(mode, cfg)
    data_loader = torch.utils.data.DataLoader(
        ds_shapes, batch_size=batch_size, num_workers=12, shuffle=with_shuffle)
        #gcollate_fn=data.collate_remove_none)

    return data_loader


def get_meshloader(mode, cfg):
    # Config

    path_meshes = cfg['data']['path_meshes']

    batch_size = cfg['training']['batch_size']

    fields = {
        'meshes': data.MeshField('mesh'),
    }

    ds_shapes = data.Shapes3dClassDataset(
        path_meshes, fields, split=None, no_except=True,
    )

    data_loader = torch.utils.data.DataLoader(
        ds_shapes, batch_size=batch_size, num_workers=12, shuffle=True)
        # collate_fn=data.collate_remove_none)

    return data_loader


def get_trainer(models, optimizers, cfg, device=None):
    out_dir = cfg['training']['out_dir']

    print_every = cfg['training']['print_every']
    visualize_every = cfg['training']['visualize_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    backup_every = cfg['training']['backup_every']

    model_selection_metric = cfg['training']['model_selection_metric']
    model_selection_mode = cfg['training']['model_selection_mode']

    ma_beta = cfg['training']['moving_average_beta']
    multi_gpu = cfg['training']['multi_gpu']
    gp_reg = cfg['training']['gradient_penalties_reg']
    w_pix = cfg['training']['weight_pixelloss']
    w_gan = cfg['training']['weight_ganloss']
    w_vae = cfg['training']['weight_vaeloss']
    experiment = cfg['training']['experiment']
    model_url = cfg['model']['model_url']
    trainer = training.Trainer(
        models['generator'], models['discriminator'],
        optimizers['generator'], optimizers['discriminator'],
        ma_beta=ma_beta,
        gp_reg=gp_reg,
        w_pix=w_pix, w_gan=w_gan, w_vae=w_vae,
        multi_gpu=multi_gpu,
        experiment=experiment,
        out_dir=out_dir,
        model_selection_metric=model_selection_metric,
        model_selection_mode=model_selection_mode,
        print_every=print_every,
        visualize_every=visualize_every,
        checkpoint_every=checkpoint_every,
        backup_every=backup_every,
        validate_every=validate_every,
        device=device,
        model_url=model_url
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):

    generator = generation.Generator3D(
        model,
        device=device,
    )
    return generator


def get_prior_z(cfg, device, **kwargs):
    z_dim = cfg['model']['z_dim']
    p0_z = dist.Normal(
        torch.zeros(z_dim, device=device),
        torch.ones(z_dim, device=device)
    )

    return p0_z
