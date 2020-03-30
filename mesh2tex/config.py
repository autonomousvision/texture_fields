import yaml
from mesh2tex import texnet
from mesh2tex import nvs
method_dict = {
    'texnet': texnet,
    'nvs': nvs,
}


# General config
def load_config(path, default_path=None):
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Individual configs
def get_models(cfg, dataset=None, device=None):
    method = cfg['method']
    models = method_dict[method].config.get_models(cfg,
                                                   dataset=dataset,
                                                   device=device)
    return models


def get_optimizers(models, cfg):
    method = cfg['method']
    optimizers = method_dict[method].config.get_optimizers(models, cfg)
    return optimizers


def get_dataset(split, cfg, input_sampling=True):
    method = cfg['method']
    dataset = method_dict[method].config.get_dataset(split, cfg,
                                                     input_sampling)
    return dataset


def get_dataloader(split, cfg):
    method = cfg['method']
    dataloader = method_dict[method].config.get_dataloader(split, cfg)
    return dataloader


def get_meshloader(split, cfg):
    method = cfg['method']
    dataloader = method_dict[method].config.get_meshloader(split, cfg)
    return dataloader


def get_generator(model, cfg, device):
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator


def get_trainer(models, optimizers, cfg, device=None):
    method = cfg['method']
    print("method: " + method)
    trainer = method_dict[method].config.get_trainer(
        models, optimizers, cfg, device=device)
    return trainer
