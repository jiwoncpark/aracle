import os, sys
import numpy as np
import pandas as pd
from scipy import ndimage
from torch.utils.data import Dataset
import torchvision.transforms
import torch

#from PIL import Image

def get_X_normalizer(normalize_pixels, mean_pixels, std_pixels):
    """Instantiate a normalizer for the pixels in images X
    
    Parameters
    ----------
    normalize_pixels : bool
        whether to normalize the pixels
    mean_pixels : array-like
        each element is the mean of pixel values for that filter
    std_pixels : array-like
        each element is the std of pixel values for that filter
    Returns
    -------
    torchvision.transforms.Compose object
        composition of transforms for X normalization

    """
    if normalize_pixels:
        normalize = torchvision.transforms.Normalize(mean=mean_pixels, std=std_pixels)
        X_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    else:
        X_transform = torch.Tensor
    return X_transform

class ResNetDataset(Dataset): # torch.utils.data.Dataset
    """Represents the training and validation data for the network

    """
    def __init__(self, dataset_dir, t_offset, normalize_X, X_mean=None, X_std=None):
        """
        Parameters
        ----------
        dataset_dir : str or os.path object
            path to the directory containing the images and metadata
        t_offset : float
            the time offset between X and Y

        
        """
        self.dataset_dir = dataset_dir
        self.t_offset = int(t_offset) #FIXME: only for toydata
        self.normalize_X = normalize_X
        self.X_mean = X_mean
        self.X_std = X_std
        self.X_transform = get_X_normalizer(self.normalize_X, self.X_mean, self.X_std)
        # Y metadata
        metadata_path = os.path.join(self.dataset_dir, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_path, index_col=None)
        self.Y_transform = torch.Tensor
        self.img_filenames = sorted(self.metadata['img_filename'].values)
        self.n_img = len(self.img_filenames)
        self.n_data = self.n_img - self.t_offset

    def __getitem__(self, index):
        X_filename = self.img_filenames[index]
        Y_filename = self.img_filenames[index + self.t_offset] #FIXME: only works for toydata
        X_img = np.load(os.path.join(self.dataset_dir, X_filename))
        X_img = np.stack([X_img]*3, axis=2).astype(int)
        Y_img = np.load(os.path.join(self.dataset_dir, Y_filename))
        X_img = self.X_transform(X_img)
        Y_img = self.Y_transform(Y_img).type(torch.int8)
        return X_img, Y_img

    def __len__(self):
        return self.n_data
