import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from skimage.transform import resize


def Bytes_to_torch (encode_img):
    """ convert image from disk to torch tensor """

    nparr = np.frombuffer (encode_img, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = img.transpose (2, 0, 1)
    img = torch.tensor (img).float ().div (255.0)
    img = transforms.Resize ((224, 224))(img)

    return img

def normalize (torch_img):
    """ normalize tensor with imagenet mean and std """
    tfms1 = transforms.Normalize (
                       mean = [0.485, 0.456, 0.406], 
                       std = [0.229, 0.224, 0.225])
    norm_img = tfms1 (torch_img)[None]
    
    return norm_img


def Tensor_to_array (img):
    """ gpu tensor => cpu array """
    return img.cpu ().numpy ().transpose (1, 2, 0)


def Bytes_to_array (encode_img):
    """ convert bytes to array and preprocess """

    nparr = np.frombuffer (encode_img, np.uint8)
    # decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    # resize both image
    image = resize (image, (128, 128), mode = 'symmetric')
    # add trailing channel dimension
    image = np.expand_dims (image, -1).astype (np.float32)
    # add batch size of 1
    image = np.expand_dims (image, 0)
    
    return image