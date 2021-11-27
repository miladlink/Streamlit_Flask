import os
import cv2
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

from skimage import measure


def show_images (img_dir, data):
    img_data = list (data.T.to_dict ().values ())
    f, ax = plt.subplots (3,3, figsize = (16,18))
    for i, data_row in enumerate (img_data):
        patientImage = data_row ['patientId'] + '.png'
        imagePath = os.path.join (img_dir, patientImage)
        image = cv2.imread (imagePath, cv2.IMREAD_GRAYSCALE)
        ax [i//3, i%3].imshow (image, cmap = plt.cm.bone)
        ax [i//3, i%3].axis ('off')
        ax [i//3, i%3].set_title('ID: {}\nTarget: {}\nClass: {}\nWindow: {}:{}:{}:{}'.format(
                data_row ['patientId'], data_row ['Target'], data_row ['class'], 
                data_row ['x'], data_row ['y'], data_row ['width'], data_row ['height']))
        
    plt.show ()
    
    
def show_images_with_bboxes (img_dir, data, train_class):
    img_data = list (data.T.to_dict ().values ())
    f, ax = plt.subplots (3,3, figsize = (16,18))
    for i, data_row in enumerate (img_data):
        patientImage = data_row ['patientId'] + '.png'
        imagePath = os.path.join (img_dir, patientImage)
        image = cv2.imread (imagePath, cv2.IMREAD_GRAYSCALE)
        ax [i//3, i%3].imshow (image, cmap = plt.cm.bone)
        ax [i//3, i%3].axis ('off')
        ax [i//3, i%3].set_title('ID: {}\nTarget: {}\nClass: {}'.format(
                data_row ['patientId'], data_row ['Target'], data_row ['class']))
        
        rows = train_class [train_class ['patientId'] == data_row ['patientId']]
        boxes = list (rows.T.to_dict ().values ())
        for j, row in enumerate (boxes):
            ax [i//3, i%3].add_patch (Rectangle (xy = (row ['x'], row ['y']),
                        width = row ['width'], height = row ['height'], 
                        color = "blue", alpha = 0.1))
    plt.show ()
    

def show_preds_with_bboxes (model, valid_gen):
    imgs, msks = next (iter (valid_gen))
    # predict batch of images
    preds = model.predict (imgs)
    # create figure
    f, axarr = plt.subplots (4, 8, figsize = (20,15))
    axarr = axarr.ravel()
    axidx = 0
    # loop through batch
    for img, msk, pred in zip(imgs, msks, preds):
        # plot image
        axarr[axidx].imshow (img [:, :, 0])
        # threshold true mask
        comp = msk [:, :, 0] > 0.5
        comp = comp.astype (np.float32)
        # apply connected components
        comp = measure.label(comp)
        # apply bounding boxes
        predictionString = ''
        for region in measure.regionprops (comp):
            # retrieve x, y, height and width
            y, x, y2, x2 = region.bbox
            height = y2 - y
            width = x2 - x
            axarr[axidx].add_patch (patches.Rectangle((x,y),width,height,linewidth = 2,edgecolor = 'b',facecolor = 'none'))
        # threshold predicted mask
        comp = pred[:, :, 0] > 0.5
        # apply connected components
        comp = measure.label(comp)
        # apply bounding boxes
        predictionString = ''
        for region in measure.regionprops(comp):
            # retrieve x, y, height and width
            y, x, y2, x2 = region.bbox
            height = y2 - y
            width = x2 - x
            axarr[axidx].add_patch(patches.Rectangle((x,y),width,height,linewidth = 2,edgecolor = 'r',facecolor = 'none'))
        axidx += 1
    plt.show()