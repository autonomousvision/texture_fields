import os
import logging
from torch.utils import data
import numpy as np
import yaml


logger = logging.getLogger(__name__)


# Fields
class Field(object):
    def load(self, data_path, idx):
        raise NotImplementedError

    def check_complete(self, files):
        raise NotImplementedError


class Shapes3dDataset(data.Dataset):
    def __init__(self, dataset_folder, fields, split=None,
                 classes=None, no_except=True, transform=None):
        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = yaml.load(f)
        else:
            metadata = {}

        # If classes is None, use all subfolders
        if classes is None:
            classes = os.listdir(dataset_folder)
            classes = [c for c in classes
                       if os.path.isdir(os.path.join(dataset_folder, c))]

        # Get all sub-datasets
        self.datasets_classes = []
        for c in classes:
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Class %s does not exist in dataset.' % c)

            metadata_c = metadata.get(c, {'id': c, 'name': 'n/a'})
            dataset = Shapes3dClassDataset(subpath, fields, split,
                                           metadata_c, no_except,
                                           transform=transform)
            self.datasets_classes.append(dataset)

        self._concat_dataset = data.ConcatDataset(self.datasets_classes)

    def __len__(self):
        return len(self._concat_dataset)

    def __getitem__(self, idx):
        return self._concat_dataset[idx]


class Shapes3dClassDataset(data.Dataset):
    def __init__(self, dataset_folder, fields, split=None,
                 metadata=dict(), no_except=True, transform=None):
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.metadata = metadata
        self.no_except = no_except
        self.transform = transform
        # Get (filtered) model list
        if split is None:
            models = [
                f for f in os.listdir(dataset_folder)
                if os.path.isdir(os.path.join(dataset_folder, f))
            ]
        else:
            split_file = os.path.join(dataset_folder, split + '.lst')
            with open(split_file, 'r') as f:
                models = f.read().split('\n')

        # self.models = list(filter(self.test_model_complete, models))
        self.models = models

    def test_model_complete(self, model):
        model_path = os.path.join(self.dataset_folder, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False
        else:
            return True

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model = self.models[idx]
        model_path = os.path.join(self.dataset_folder, model)
        data = {}
        for field_name, field in self.fields.items():
            try:
                field_data = field.load(model_path, idx)
            except Exception:
                if self.no_except:
                    logger.warn(
                        'Error occured when loading field %s of model %s'
                        % (field_name, model)
                    )
                    return None
                else:
                    raise
            if isinstance(field_data, dict):

                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_model(self, idx):
        return self.models[idx]


class CombinedDataset(data.Dataset):
    def __init__(self, datasets, idx_main=0):
        self.datasets = datasets
        self.idx_main = idx_main

    def __len__(self):
        return len(self.datasets[self.idx_main])

    def __getitem__(self, idx):
        out = []
        for it, ds in enumerate(self.datasets):
            if it != self.idx_main:
                x_idx = np.random.randint(0, len(ds))
            else:
                x_idx = idx
            out.append(ds[x_idx])
        return out


# Collater
def collate_remove_none(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter(check_element_valid, batch))
    return data.dataloader.default_collate(batch)


def check_element_valid(batch):
    if batch is None:
        return False
    elif isinstance(batch, list):
        for b in batch:
            if not check_element_valid(b):
                return False
    elif isinstance(batch, dict):
        for b in batch.values():
            if not check_element_valid(b):
                return False
    return True


# Worker initialization to ensure true randomeness
def worker_init_fn(worker_id):
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)
