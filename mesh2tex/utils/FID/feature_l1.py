import os
import pathlib
import torch
import numpy as np
from imageio import imread
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d

from mesh2tex.utils.FID.inception import InceptionV3


def get_activations(images, model, batch_size=64, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    d0 = images.shape[0]
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))
    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        batch = Variable(batch, volatile=True)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def _compute_statistics_of_path(path0, path1, model, batch_size, dims, cuda):
    path0 = pathlib.Path(path0)
    path1 = pathlib.Path(path1)

    files_list = os.listdir(path0)
    files0 = [os.path.join(path0, f) for f in files_list]
    files1 = [os.path.join(path1, f) for f in files_list]
    assert(len(files0) == len(files1))

    # First set of images
    imgs = np.array([imread(str(fn)).astype(np.float32) for fn in files0])
    # Bring images to shape (B, 3, H, W)
    imgs = imgs.transpose((0, 3, 1, 2))[:, :3]
    # Rescale images to be between 0 and 1
    imgs /= 255
    feat0 = get_activations(imgs, model, batch_size, dims, cuda, False)

    # Second set of images
    imgs = np.array([imread(str(fn)).astype(np.float32) for fn in files1])
    # Bring images to shape (B, 3, H, W)
    imgs = imgs.transpose((0, 3, 1, 2))[:, :3]
    # Rescale images to be between 0 and 1
    imgs /= 255
    feat1 = get_activations(imgs, model, batch_size, dims, cuda, False)

    feature_l1 = np.mean(np.abs(feat0 - feat1))
    return feature_l1


def _compute_statistics_of_tensors(images_fake, images_real, model, batch_size,
                                   dims, cuda):

    # First set of images
    imgs = images_fake.cpu().numpy().astype(np.float32)
    feat0 = get_activations(imgs, model, batch_size, dims, cuda, False)

    # Second set of images
    imgs = images_real.cpu().numpy().astype(np.float32)
    feat1 = get_activations(imgs, model, batch_size, dims, cuda, False)

    feature_l1 = np.mean(np.abs(feat0 - feat1))
    return feature_l1


def calculate_feature_l1_given_paths(paths, batch_size, cuda, dims):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    feature_l1 = _compute_statistics_of_path(
        paths[0], paths[1], model, batch_size, dims, cuda)

    return feature_l1


def calculate_feature_l1_given_tensors(tensor1, tensor2,
                                       batch_size, cuda, dims):
    """Calculates the FID of two image tensors"""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    feature_l1 = _compute_statistics_of_tensors(
        tensor1, tensor2, model, batch_size, dims, cuda)

    return feature_l1
