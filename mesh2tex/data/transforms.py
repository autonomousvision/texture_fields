import numpy as np
from skimage.transform import resize
from scipy.spatial import cKDTree as KDTree


class PointcloudNoise(object):
    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out


class SubsamplePointcloud(object):
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        data_out = data.copy()
        points = data[None]
        normals = data['normals']

        indices = np.random.randint(points.shape[1], size=self.N)
        data_out[None] = points[:, indices]
        data_out['normals'] = normals[:, indices]

        return data_out


class ComputeKNNPointcloud(object):
    def __init__(self, K):
        self.K = K

    def __call__(self, data):
        data_out = data.copy()
        points = data[None]
        kdtree = KDTree(points.T)
        knn_idx = kdtree.query(points.T, k=self.K)[1]
        knn_idx = knn_idx.T

        data_out['knn_idx'] = knn_idx

        return data_out


class ImageToGrayscale(object):
    def __call__(self, img):
        r, g, b = img[..., 0:1], img[..., 1:2], img[..., 2:3]
        out = 0.2990 * r + 0.5870 * g + 0.1140 * b
        return out


class ImageToDepthValue(object):
    def __call__(self, img):    
        return img[..., :1]


class ResizeImage(object):
    def __init__(self, size, order=1):
        self.size = size
        self.order = order

    def __call__(self, img):
        img_out = resize(img, self.size, order=self.order,
                         clip=False, mode='constant',
                         anti_aliasing=False)
        img_out = img_out.astype(img.dtype)
        return img_out
