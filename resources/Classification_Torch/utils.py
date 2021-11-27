import torch
import numpy as np
import matplotlib.pyplot as plt
import torch


def shuffle_split (labels, val_pct = 0.1, seed = 42):
    """ shuffle all data and split to train, valid and test """
    n_val = int (len (labels) * val_pct)
    np.random.seed (seed)
    idx = np.random.permutation (len (labels))
    labels = labels.values [idx,:]
    return labels [2*n_val:], labels [:n_val], labels [n_val:2*n_val]


def imshow (img, title = None):
    """ visualize torch tensor image """
    img = img.numpy ().transpose (1, 2, 0)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = img * std + mean
    img = np.clip (img, 0, 1)
    
    plt.figure (figsize = (10, 8))
    plt.axis ('off')
    plt.imshow (img)
    if title:
        plt.title (title)
        

def plot_acc_loss (loss, val_loss, acc, val_acc):
    """ plot training and validation loss and accuracy """
    plt.figure (figsize = (12, 4))
    plt.subplot (1, 2, 1)
    plt.plot (range (len (loss)), loss, 'b-', label = 'Training')
    plt.plot (range (len (loss)), val_loss, 'bo-', label = 'Validation')
    plt.xlabel ('Epochs')
    plt.ylabel ('Loss')
    plt.title ('Loss')
    plt.legend ()

    plt.subplot (1, 2, 2)
    plt.plot (range (len (acc)), acc, 'b-', label = 'Training')
    plt.plot (range (len (acc)), val_acc, 'bo-', label = 'Validation')
    plt.xlabel ('Epochs')
    plt.ylabel ('accuracy')
    plt.title ('Accuracy')
    plt.legend ()

    plt.show ()


def predict_image (model, test_dl, device):
    """ predict random image and its label and prediction """
    cls2idx = ['penumia', 'normal']
    imgs, lbls = next (iter (test_dl))
    output = model (imgs.to(device))
    _, preds = torch.max (output, dim = 1)
    idx = np.random.choice (len (imgs))
    imshow (imgs [idx], 'Acctual: {}, Predicted: {}'.format (cls2idx [lbls [idx]], cls2idx [preds [idx]]))