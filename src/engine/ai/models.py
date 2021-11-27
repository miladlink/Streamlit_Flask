import torch
from torch import nn
from torchvision import models

'===============================pytorch==========================='

def load_checkpoint (checkpoint_path, model, device):
    """ loading model's weights """
    model.load_state_dict (torch.load (checkpoint_path, map_location = device) ['state_dict'])


def torch_model (model_name, device, checkpoint_path = None):
    """ select imagenet models by their name and loading weights """
    if checkpoint_path:
        pretrained = False
    else:
        pretrained = True
    
    model = models.__dict__ [model_name](pretrained)

    if hasattr (model, 'classifier'):
        if model_name == 'mobilenet_v2':
            model.classifier = nn.Sequential(
                nn.Dropout (0.2),
                nn.Linear (model.classifier [-1].in_features, 2))
            
        else:
            model.classifier = nn.Sequential(
                nn.Linear (model.classifier.in_features, 2))
    
    elif hasattr (model, 'fc'):
        model.fc = nn.Linear (model.fc.in_features, 2)
        
    model.to(device)
    
    if checkpoint_path:
        load_checkpoint (checkpoint_path, model, device)
    
    return model

'=========================tensorflow============================'

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D, concatenate, add


def create_downsample (channels, inputs):
    """ bn + conv + leaky relu + maxpool """
    x = BatchNormalization (momentum = 0.9)(inputs)
    x = LeakyReLU (0)(x)
    x = Conv2D (channels, 1, padding = 'same', use_bias = False)(x)
    x = MaxPool2D (2)(x)

    return x


def create_resblock (channels, inputs):
    """ a convelution block with residual connection """
    x = BatchNormalization (momentum = 0.9)(inputs)
    x = LeakyReLU (0)(x)
    x = Conv2D (channels, 3, padding='same', use_bias = False)(x)
    x = BatchNormalization (momentum = 0.9)(x)
    x = LeakyReLU (0)(x)
    x = Conv2D (channels, 3, padding = 'same', use_bias = False)(x)

    #Added Start
    x = BatchNormalization (momentum = 0.9)(x)
    x = LeakyReLU (0)(x)
    x = Conv2D (channels, 3, padding = 'same', use_bias = False)(x)
    #Added End
    
    addInput = x;
    resBlockOut = add ([addInput, inputs])
    out = concatenate([resBlockOut, addInput], axis = 3)
    out = Conv2D (channels, 1, padding = 'same', use_bias = False)(out)
    return out

def tf_model (weight_path, input_size = 128, channels = 16, n_blocks = 2, depth = 3):
    """ create final network like unet architecture """
    # input
    inputs = Input (shape = (input_size, input_size, 1))
    x = Conv2D (channels, 3, padding = 'same', use_bias = False)(inputs)
    # residual blocks
    for d in range (depth):
        channels = channels * 2
        x = create_downsample (channels, x)
        for b in range (n_blocks):
            x = create_resblock (channels, x)
    # output
    x = BatchNormalization (momentum = 0.9)(x)
    x = LeakyReLU (0)(x)
    x = Conv2D (1, 1, activation = 'sigmoid')(x)
    outputs = UpSampling2D (2**depth)(x)
    model = Model (inputs = inputs, outputs = outputs)
    
    model.load_weights (weight_path)

    
    return model