import argparse
import pandas as pd
import os
import glob

from mesh2tex import config
from mesh2tex.eval import evaluate_generated_images

categories = {'02958343': 'cars', '03001627': 'chairs',
              '02691156': 'airplanes', '04379243': 'tables',
              '02828884': 'benches', '02933112': 'cabinets',
              '04256520': 'sofa', '03636649': 'lamps',
              '04530566': 'vessels'}

parser = argparse.ArgumentParser(
    description='Generate Color for given mesh.'
)

parser.add_argument('config', type=str, help='Path to config file.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
base_path = cfg['test']['vis_dir']


if cfg['data']['shapes_multiclass']:
    category_paths = glob.glob(os.path.join(base_path, '*'))
else:
    category_paths = [base_path]

for category_path in category_paths:
    cat_id = os.path.basename(category_path)
    category = categories.get(cat_id, cat_id)
    path1 = os.path.join(category_path, 'fake/')
    path2 = os.path.join(category_path, 'real/')
    print('Evaluating %s (%s)' % (category, category_path))

    evaluation = evaluate_generated_images('all', path1, path2)
    name = base_path

    df = pd.DataFrame(evaluation, index=[category])
    df.to_pickle(os.path.join(category_path, 'eval.pkl'))
    df.to_csv(os.path.join(category_path, 'eval.csv'))

print('Evaluation finished')
