import sys
sys.path.append('../src')

# import os
# os.chdir('../src')

import numpy as np
import torch

import unittest

from tf_data import preprocess
from torch_data import *

img_dir = '../images/05fe70c1-4cc1-42fb-9303-083fd29dac0f.png'

class TestData (unittest.TestCase):
    """ testing all function use to changing data """
    def test_tf_data (self):
        image = preprocess (img_dir)
        self.assertEqual (image.shape, (1, 128, 128, 1))
        self.assertTrue (image.dtype == np.float32)
        
    def test_torch_data (self):
        torch_img = Image_to_torch (img_dir)
        self.assertEqual (torch_img.shape, (3, 224, 224))
        self.assertTrue (torch_img.dtype == torch.float32)
        
        norm_img = normalize (torch_img)
        self.assertEqual (norm_img.shape, (1, 3, 224, 224))
        
        np_img = Tensor_to_array (norm_img [0])
        self.assertEqual (np_img.shape, (224, 224, 3))
        self.assertTrue (np_img.dtype == np.float32)
        
        
if __name__ == "__main__":
    unittest.main()