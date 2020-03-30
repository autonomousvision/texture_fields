"""Base file for starting training
"""

import torch
import argparse
from mesh2tex import config
import matplotlib

matplotlib.use('Agg')


parser = argparse.ArgumentParser(
    description='Train a Texture Field.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified '
                         'number of seconds with exit code 2.')
args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
exit_after = args.exit_after


models = config.get_models(cfg, device=device)
optimizers = config.get_optimizers(models, cfg)


train_loader = config.get_dataloader('train', cfg)
val_loader = config.get_dataloader('val_eval', cfg)

if cfg['training']['vis_fixviews'] is True:
    vis_loader = config.get_dataloader('val_vis', cfg)
else:
    vis_loader = None


trainer = config.get_trainer(models, optimizers, cfg, device=device)

trainer.train(train_loader, val_loader, vis_loader,
              exit_after=exit_after, n_epochs=None)
