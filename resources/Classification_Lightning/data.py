import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


def read_labels (lbls_dir, img_col_name = 'patientId', target_col_name = 'Target'):
    """ read labels from csv file and filter it """
    labels = pd.read_csv (lbls_dir)
    columns = [img_col_name, target_col_name]

    return labels.filter(columns)


def shuffle_split_path (imgs_dir, labels, val_pct = 0.1, seed = 42):
    """ shuffle all data and split to train and valid """
    n_val = int (len (labels) * val_pct)
    np.random.seed (seed)
    idx = np.random.permutation (len (labels))
    labels = labels.values [idx,:]
    
    train_paths = [os.path.join (imgs_dir, image[0]) for image in labels [n_val:]]
    valid_paths = [os.path.join (imgs_dir, image[0]) for image in labels [:n_val]]
    
    return train_paths, valid_paths, labels [n_val:], labels [:n_val]


class RSNADataset (Dataset):
    """ create dataset by using image path and their labels """
    def __init__ (self, paths, labels, transform = None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        
    def __len__ (self):
        return len (self.paths)
    
    def __getitem__ (self, index):
        image_path = self.paths [index] + '.png'
        image = Image.open (image_path).convert ('RGB')
        
        label = self.labels [index][1]

        if self.transform is not None:
            image = self.transform (image)
            
        return image, label    
    
    
def get_loaders (train_paths, valid_paths, train_lbls, valid_lbls, batch_size, train_tfms, valid_tfms):
    """ create train and valid loaders """
    train_ds = RSNADataset (train_paths, train_lbls, transform = train_tfms)
    valid_ds = RSNADataset (valid_paths, valid_lbls, transform = valid_tfms)
    
    train_dl = DataLoader (train_ds, batch_size = batch_size, shuffle = True)
    valid_dl = DataLoader (valid_ds, batch_size = batch_size, shuffle = False)
    
    return train_dl, valid_dl