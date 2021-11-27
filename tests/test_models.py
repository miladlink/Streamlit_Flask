import sys
sys.path.append('../src')

# import os
# os.chdir('../src')

import numpy as np
import torch

import unittest

from tf_model import create_network
from torch_model import get_model

tf_dummy_image = np.random.randn (1, 128, 128, 1)
torch_dummy_image = torch.randn ((1, 3, 224, 224))

device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')


class TestModels (unittest.TestCase):
    """ testing all models that use in app """
    def test_tf_model (self):
        OD_model = create_network (weight_path = '../weights/RSNA_OD.h5')
        output = OD_model.predict (tf_dummy_image)
        self.assertEqual (output.shape, (1, 128, 128, 1))
        
    def test_res50 (self):
        model_name = 'resnet50'
        res_model = get_model (model_name, device, '../weights/' + model_name + '.pth')
        output = res_model (torch_dummy_image)
        self.assertEqual (output.shape, (1, 2))
        
    def test_dense201 (self):
        model_name = 'densenet201'
        res_model = get_model (model_name, device, '../weights/' + model_name + '.pth')
        output = res_model (torch_dummy_image)
        self.assertEqual (output.shape, (1, 2))
        
    def test_mobilev2 (self):
        model_name = 'mobilenet_v2'
        res_model = get_model (model_name, device, '../weights/' + model_name + '.pth')
        output = res_model (torch_dummy_image)
        self.assertEqual (output.shape, (1, 2))
        
if __name__ == '__main__':
    unittest.main ()
    