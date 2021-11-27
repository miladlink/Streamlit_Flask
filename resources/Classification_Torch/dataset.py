import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


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

#         image = cv2.imread (image_path)
#         image = (image).clip(0, 255).astype(np.uint8)
#         image = Image.fromarray(image).convert('RGB')

        if self.transform is not None:
            image = self.transform (image)
            
        return image, label