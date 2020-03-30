from mesh2tex.geometry import (
    pointnet
)


encoder_dict = {
    'simple': pointnet.SimplePointnet,
    'resnet': pointnet.ResnetPointnetConv,
}


def get_representation(batch, device=None):
    mesh_points = batch['pointcloud'].to(device)
    mesh_normals = batch['pointcloud.normals'].to(device)
    geom_repr = {
        'points': mesh_points,
        'normals': mesh_normals,
    }
    if 'pointcloud.knn_idx' in batch:
        knn_idx = batch['pointcloud.knn_idx'].to(device)
        geom_repr['knn_idx'] = knn_idx

    return geom_repr

