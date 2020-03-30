import mesh2tex.utils.FID.fid_score as FID
import mesh2tex.utils.FID.feature_l1 as feature_l1
import mesh2tex.utils.SSIM_L1.ssim_l1_score as SSIM


def evaluate_generated_images(metric, path_fake, path_real, batch_size=64):
    """
    Start requested evaluation functions

    args:
        metric
        path_fake (Path to fake images)
        path_real (Path to real images)
        batch_size

    return:
        val_dict: dict with all metrics
    """
    if metric == 'FID':
        paths = (path_fake, path_real)
        value = FID.calculate_fid_given_paths(paths,
                                              batch_size,
                                              True,
                                              2048)
        val_dict = {'FID': value}

    elif metric == 'SSIM_L1':
        paths = (path_fake, path_real)
        value = SSIM.calculate_ssim_l1_given_paths(paths)
        val_dict = {'SSIM': value[0],
                    'L1': value[1]}
    elif metric == 'FeatL1':
        paths = (path_fake, path_real)
        value = feature_l1.calculate_feature_l1_given_paths(
            paths, batch_size, True, 2048)
        val_dict = {'FeatL1': value}

    elif metric == 'all':
        paths = (path_fake, path_real)
        value = SSIM.calculate_ssim_l1_given_paths(paths)
        value_FID = FID.calculate_fid_given_paths(
            paths, batch_size, True, 2048)
        value_FeatL1 = feature_l1.calculate_feature_l1_given_paths(
            paths, batch_size, True, 2048)
        val_dict = {'FID': value_FID,
                    'SSIM': value[0],
                    'L1': value[1],
                    'FeatL1': value_FeatL1}

    return val_dict
