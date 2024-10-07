import itertools
import os
import logging
from collections import Counter
from PIL import Image

import numpy as np
import pandas as pd
import torch

from stratification.classification.datasets.base import GEORGEDataset


class WOSDataset(GEORGEDataset):
    """WOS Dataset
    """

    # used to determine subclasses (index here used for querying sample class)
    
    split_dict = {'train': 0, 'val': 1, 'test': 2}

    def __init__(self, root, split, transform=None, download=False, ontology='default',
                 augment=False):
        assert (transform is None)
        
        super().__init__('wos', root, split, transform=transform, download=download,
                         ontology=ontology)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'WOS')

    def _check_exists(self):
        """Checks whether or not the wos labels CSV has been initialized."""
        return (os.path.isfile(os.path.join(self.processed_folder, 'X.txt')) and 
                os.path.isfile(os.path.join(self.processed_folder, 'label.csv')) and
                os.path.isfile(os.path.join(self.processed_folder, 'label_level1.csv'))
                )

    def _download(self):
        """Raises an error if the raw dataset has not yet been downloaded."""
        raise ValueError('Download the WOS Dataset. It should contain main_tensor.pt, label.csv, label_level1.csv')

    def _load_samples(self):
        """Loads the WOS dataset"""
        #embeddings = np.array(torch.load(os.path.join(self.processed_folder,'main_tensor.pt')))
        #embeddings = np.squeeze(embeddings)
        file_path = os.path.join(self.processed_folder,'X.txt')
        with open(file_path, 'r', encoding='utf-8') as file:
            texts = file.read().splitlines()
        texts = np.array(texts)
        superclass = np.array(pd.read_csv(os.path.join(self.processed_folder,'label_level1.csv'))['0'])
        true_subclass = np.array(pd.read_csv(os.path.join(self.processed_folder,'label.csv'))['0'])
        self.create_splits(true_subclass.shape[0])
        splits = pd.read_csv(os.path.join(self.processed_folder, 'split.csv'),names=['split'])

        # split dataset
        split_mask = (splits == self.split_dict[self.split]).squeeze()
        #embeddings = embeddings[split_mask]
        texts = texts[split_mask]
        superclass = superclass[split_mask]
        true_subclass = true_subclass[split_mask]
        assert(texts.shape[0]==superclass.shape[0])
        assert(texts.shape[0]==true_subclass.shape[0])

        #X = torch.from_numpy(embeddings)
        X = texts
        Y_dict = {
            'superclass': torch.from_numpy(superclass),
            'true_subclass': torch.from_numpy(true_subclass)
        }
        return X, Y_dict


    def create_splits(self,n):
        if os.path.isfile(os.path.join(self.processed_folder, 'split.csv')):
            return

        num_zeros = int(n * 0.6)
        num_ones = int(n * 0.2)
        num_twos = n - (num_zeros + num_ones)  # Ensure the total is n
        array = np.array([0] * num_zeros + [1] * num_ones + [2] * num_twos)
        np.random.shuffle(array)
        np.savetxt(os.path.join(self.processed_folder, 'split.csv'), array, delimiter=',', fmt='%d')    
                   

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (x: torch.Tensor, y: dict) where X is a tensor representing an image
                and y is a dictionary of possible labels.
        """
        x = self.X[idx]
        y_dict = {name: label[idx] for name, label in self.Y_dict.items()}
        return x, y_dict
