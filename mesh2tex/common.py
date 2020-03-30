import torch
from scipy.spatial import cKDTree as KDTree


def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
    indices = []
    distances = []

    for (p1, p2) in zip(points_src, points_tgt):
        p1 = p1.detach().cpu().numpy().T
        p2 = p2.detach().cpu().numpy().T

        kdtree = KDTree(p2)
        dist, idx = kdtree.query(p1, k=k, n_jobs=-1)
        indices.append(idx)
        distances.append(dist)

    indices = torch.LongTensor(indices)
    distances = torch.FloatTensor(distances)

    return indices, distances


def normalize_imagenet(x):
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x
