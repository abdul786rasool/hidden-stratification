import os
import logging
import codecs
import random
from collections import defaultdict
from PIL import Image
import kaggle
import pickle
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive

from .base import GEORGEDataset


class CifarTenDataset(GEORGEDataset):

    _channels = 3
    _resolution = 28
    _normalization_stats = {'mean': (0.4898 , 0.4804, 0.445), 'std': (0.2462, 0.2426, 0.2606 )}
    _pil_mode = "RGB"

    def __init__(self, root, split, transform=None, resize=True, download=False, subsample_8=False,
                 ontology='3-sup-class', augment=False):
        assert (transform is None)
        transform = get_transform_CIFAR(resize=resize, augment=augment)
        self.subclass_proportions = {8: 0.05} if ('train' in split and subsample_8) else {}
        self.category_to_label = {'airplane':0, 'automobile':1, 'ship':2, 'truck':3,
                                'bird':4, 'cat':5, 'deer':6, 'dog':7, 'frog':8, 'horse':9}
        self.class_div = {0:0, 4:0 , 2:1}
        super().__init__('CIFAR10', root, split, transform=transform, download=download,
                         ontology=ontology)

    def _load_samples(self):
        """Loads the U-MNIST dataset from the data file created by self._download"""
        data_file = f'{self.split}.pt'
        logging.info(f'Loading {self.split} split...')
        data, original_labels = torch.load(os.path.join(self.processed_folder, data_file))

        logging.info('Original label counts:')
        logging.info(np.bincount(original_labels))

        # subsample some subset of subclasses
        if self.subclass_proportions:
            logging.info(f'Subsampling subclasses: {self.subclass_proportions}')
            data, original_labels = self.subsample_digits(data, original_labels,
                                                          self.subclass_proportions)
            logging.info('New label counts:')
            logging.info(np.bincount(original_labels))

        # determine superclass partition of original_labels
        if self.ontology == 'four-comp':
            superclass_labels = (original_labels > 3).long()
            self.superclass_names = ['< 4', '≥ 4']
        elif self.ontology == '3-sup-class':
            superclass_labels = torch.tensor(list(map(lambda x : self.class_div.get(x,2),original_labels))).long()
        else:
            raise ValueError(f'Ontology {self.ontology} not supported.')

        X = data
        Y_dict = {'superclass': superclass_labels, 'true_subclass': original_labels.clone()}
        return X, Y_dict

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (x_dict, y_dict) where x_dict is a dictionary mapping all
                possible inputs and y_dict is a dictionary for all possible labels.
        """
        x = self.X[idx]
        image = Image.fromarray(x.numpy(), mode=self._pil_mode)
        if self.transform is not None:
            image = self.transform(image)
        x = image

        y_dict = {k: v[idx] for k, v in self.Y_dict.items()}
        return x, y_dict

    def subsample_digits(self, data, labels, subclass_proportions, seed=0):
        prev_state = random.getstate()
        random.seed(seed)
        data_mod_seed = random.randint(0, 2**32)
        random.seed(data_mod_seed)

        for label, freq in subclass_proportions.items():
            logging.info(f'Subsampling {label} fine class, keeping {freq*100} percent...')
            inds = [i for i, x in enumerate(labels) if x == label]
            inds = set(random.sample(inds, int((1 - freq) * len(inds))))
            labels = torch.tensor([lab for i, lab in enumerate(labels) if i not in inds])
            data = torch.stack([datum for i, datum in enumerate(data) if i not in inds])

        random.setstate(prev_state)
        return data, labels

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'CIFAR10', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'CIFAR10', 'processed')

    def _check_exists(self):
        return all(
            os.path.exists(os.path.join(self.processed_folder, f'{split}.pt'))
            for split in ['train', 'val', 'test'])

    def _download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('emadtolba/cifar10-comp',path=self.raw_folder, unzip=True)

        # process and save as torch files
        logging.info('Processing...')
        training_set, validation_set, test_set = self.read_data_labels(os.path.join(self.raw_folder, 'train_images.npy'),
                                             os.path.join(self.raw_folder, 'train_labels.csv'))

        with open(os.path.join(self.processed_folder, 'train.pt'), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, 'val.pt'), 'wb') as f:
            torch.save(validation_set, f)            
        with open(os.path.join(self.processed_folder, 'test.pt'), 'wb') as f:
            torch.save(test_set, f)
        logging.info('Done downloading!')



    def read_data_labels(self,file,csv_file):
        data = np.load(file)
        data = data.transpose(0, 2, 3, 1)
        original_labels = pd.read_csv(csv_file)['Category']
        original_labels = np.array(list(map( lambda x: self.category_to_label[x], original_labels)))
        x_train, x_, y_train, y_ = train_test_split(data, original_labels, test_size=0.3, random_state=0)
        x_val, x_test, y_val, y_test = train_test_split(x_, y_, test_size=2/3, random_state=0)
        return (torch.tensor(x_train),torch.tensor(y_train)), (torch.tensor(x_val),torch.tensor(y_val)), (torch.tensor(x_test),torch.tensor(y_test))
    

    def _create_val_split(self, seed=0, val_proportion=0.2):
        data, original_labels = torch.load(os.path.join(self.processed_folder, 'train.pt'))
        original_labels = original_labels.numpy()
        original_label_counts = np.bincount(original_labels)
        assert all(i > 0 for i in original_label_counts), \
            'set(labels) must consist of consecutive numbers in [0, S]'
        val_quota = np.round(original_label_counts * val_proportion).astype(int)

        # reset seed here in case random fns called again (i.e. if get_loaders called twice)
        prev_state = random.getstate()
        random.seed(seed)
        shuffled_idxs = random.sample(range(len(data)), len(data))
        random.setstate(prev_state)

        train_idxs = []
        val_idxs = []
        val_counts = defaultdict(int)

        # Iterate through shuffled dataset to extract valset idxs
        for i in shuffled_idxs:
            label = original_labels[i]
            if val_counts[label] < val_quota[label]:
                val_idxs.append(i)
                val_counts[label] += 1
            else:
                train_idxs.append(i)

        train_idxs = sorted(train_idxs)
        val_idxs = sorted(val_idxs)
        assert len(set(val_idxs) & set(train_idxs)) == 0, \
            'valset and trainset must be mutually exclusive'

        logging.info(f'Creating training set with class counts:\n' +
                     f'{np.bincount(original_labels[train_idxs])}')
        trainset = (data[train_idxs], torch.tensor(original_labels[train_idxs]))
        with open(os.path.join(self.processed_folder, 'train.pt'), 'wb') as f:
            torch.save(trainset, f)

        logging.info(f'Creating validation set with class counts:\n' +
                     f'{np.bincount(original_labels[val_idxs])}')
        valset = (data[val_idxs], torch.tensor(original_labels[val_idxs]))
        with open(os.path.join(self.processed_folder, 'val.pt'), 'wb') as f:
            torch.save(valset, f)
        logging.info(f'Split complete!')



def get_transform_CIFAR(resize=True, augment=False):
    test_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(**CifarTenDataset._normalization_stats)
    ]
    if resize:
        test_transform_list.insert(0, transforms.Resize((32, 32)))
    if not augment:
        return transforms.Compose(test_transform_list)

    train_transform_list = [
        transforms.RandomCrop(CifarTenDataset._resolution, padding=4),
        transforms.RandomHorizontalFlip()
    ] + test_transform_list
    return transforms.Compose(train_transform_list)

