import io
import base64
import numpy as np
from PIL import Image
from torchvision import transforms

from skimage.transform import resize

def b64_to_pil (str_img):
    encoded_img = base64.b64decode(str_img)
    img = Image.open (io.BytesIO (encoded_img))
    return img

def file_to_b64 (file):
    with open (file, 'rb') as f:
        bytes = f.read ()
        str_img = base64.b64encode (bytes).decode("utf-8")

    return str_img


def Img_to_torch (img):
    """ convert PIL Image to tensor """
    tfms = transforms.Compose ([
    transforms.Resize ((224,224)),
    transforms.ToTensor ()
    ])

    img = img.convert ('RGB')
    torch_img = tfms (img)

    return torch_img


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


def Img_to_array (img):
    image = np.array (img.convert ('L'))
    # resize both image
    image = resize (image, (128, 128), mode = 'symmetric')
    # add trailing channel dimension
    image = np.expand_dims (image, -1).astype (np.float32)
    # add batch size of 1
    image = np.expand_dims (image, 0)

    return image

