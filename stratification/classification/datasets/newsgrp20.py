import itertools
import os
import logging
from collections import Counter
from PIL import Image

import numpy as np
import pandas as pd
import torch

from stratification.classification.datasets.base import GEORGEDataset


class newsgrp20Dataset(GEORGEDataset):
    """20newsgroup Dataset
    """

    # used to determine subclasses (index here used for querying sample class)
    
    split_dict = {'train': 0, 'val': 1, 'test': 2}

    def __init__(self, root, split, transform=None, download=False, ontology='default',
                 augment=False):
        assert (transform is None)
        
        super().__init__('agnews', root, split, transform=transform, download=download,
                         ontology=ontology)

    @property
    def processed_folder(self):
        return os.path.join(self.root, '20newsgroup')

    def _check_exists(self):
        
        return (
            os.path.isfile(os.path.join(self.processed_folder, 'train_data.csv')) and
            os.path.isfile(os.path.join(self.processed_folder, 'test_data.csv')) and
            os.path.isfile(os.path.join(self.processed_folder, 'val_data.csv'))
                 )
                

    def _download(self):
        """Raises an error if the raw dataset has not yet been downloaded."""
        raise ValueError('Download the 20newsgroup Dataset. It should contain train_data.csv, test_data.csv and val_data.csv')

    def _load_samples(self):
        """Loads the AGnews dataset"""

        file_path = os.path.join(self.processed_folder,f'{self.split}_data.csv')
        data = pd.read_csv(file_path)
        texts = np.array(data['text'],dtype=str)
        superclass = np.array(data['superclass'])
        subclass = np.array(data['subclass'])
        assert(texts.shape[0]==superclass.shape[0])
        assert(texts.shape[0]==subclass.shape[0])
        
        X = texts.tolist()
        Y_dict = {
            'superclass': torch.from_numpy(superclass),
            'true_subclass': torch.from_numpy(subclass)
        }
        return X, Y_dict
 
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
