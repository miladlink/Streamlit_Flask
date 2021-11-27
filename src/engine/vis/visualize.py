import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

from skimage import measure
import torch

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

from engine.io.b64 import normalize, Tensor_to_array


def predict_and_gradcam (model, torch_img, device):
    """ predict class and plot image with their gradcam results """
    img = normalize (torch_img)
    out = model (img.to(device))
    # predict class
    _, pred = torch.max (out, dim = 1)
    cls2idx = ['Normal', 'Pneumomia']
    # ploting GradCam
    fig, ax = plt.subplots (nrows = 1, ncols = 5, figsize = (15, 5))
    
    # usage layer for gradcam
    if hasattr (model, 'fc'):
        target_layer = model.layer4 [2].bn3
        
    else:
        target_layer = model.features
   
    
    gradcam = GradCAM (model, target_layer)
    gradcam_pp = GradCAMpp (model, target_layer)
    
    mask, _ = gradcam (img)
    heatmap, result = visualize_cam (mask, torch_img)

    mask_pp, _ = gradcam_pp (img)
    heatmap_pp, result_pp = visualize_cam (mask_pp, torch_img)

    ax [0].imshow (Tensor_to_array (img [0])[:, :, 0], cmap = 'gray')
    ax [0].set_title ('Predict: {}'.format (cls2idx [pred]))
    ax [0].axis ('off')

    ax [1].imshow (Tensor_to_array (heatmap))
    ax [1].set_title ('Grad Cam')
    ax [1].axis ('off')

    ax [2].imshow (Tensor_to_array (heatmap_pp))
    ax [2].set_title ('Grad Cam ++')
    ax [2].axis ('off')

    ax [3].imshow (Tensor_to_array (result))
    ax [3].set_title ('Result')
    ax [3].axis ('off')

    ax [4].imshow (Tensor_to_array (result_pp))
    ax [4].set_title ('Result ++')
    ax [4].axis ('off')

    return fig


def show_preds_with_bboxes (model, image):
    """ showing selected picture with bounding box """
    # predict batch of images
    pred = model.predict (image)
    # create figure
    fig, ax = plt.subplots ()
    # plot image
    ax.imshow (image [0][:, :, 0])
    ax.axis ('off')
    # threshold predicted mask
    comp = pred [0][:, :, 0] > 0.5
    comp = comp.astype (np.float32)
    # apply connected components
    comp = measure.label (comp)
    # apply bounding boxes
    predictionString = ''
    for region in measure.regionprops (comp):
        if region:
            predictionString = 'Pneumonia Detected!'
            # retrieve x, y, height and width
            y, x, y2, x2 = region.bbox
            height = y2 - y
            width = x2 - x
            ax.add_patch (patches.Rectangle((x,y), width,height,linewidth = 2, edgecolor = 'b',facecolor = 'none'))
        else:
            predictionString = 'No Pneumonia Detected'
    
    ax.set_title (predictionString)

    return fig
