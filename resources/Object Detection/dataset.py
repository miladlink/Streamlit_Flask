import os
import csv
import cv2
import random
import numpy as np

from skimage.transform import resize
from tensorflow.keras.utils import Sequence


def pneumonia_bboxes (labels_dir):
    # empty dictionary
    pneumonia_locations = {}
    # load table
    with open (labels_dir, mode = 'r') as infile:
        # open reader
        reader = csv.reader (infile)
        # skip header
        next (reader, None)
        # loop through rows
        for rows in reader:
            # retrieve information
            filename = rows [1]
            location = rows [2:6]
            pneumonia = rows [6]
            # if row contains pneumonia add label to dictionary
            # which contains a list of pneumonia locations per filename
            if pneumonia == '1':
                # convert string to float to int
                location = [int (float (i)) for i in location]
                # save pneumonia location in dictionary
                if filename in pneumonia_locations:
                    pneumonia_locations [filename].append (location)
                else:
                    pneumonia_locations [filename] = [location]
                    
    return pneumonia_locations


class generator (Sequence):
    
    def __init__ (self, folder, filenames, pneumonia_locations = None, batch_size = 32,
                  image_size = 256, shuffle = True, augment = False, predict = False):
        
        self.folder = folder
        self.filenames = filenames
        self.pneumonia_locations = pneumonia_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end ()
        
    def __load__(self, filename):
        # load file as numpy array
        image = cv2.imread (os.path.join (self.folder, filename), cv2.IMREAD_GRAYSCALE)
        # create empty mask
        mask = np.zeros ((1024, 1024))
        # get filename without extension
        filename = filename.split ('.')[0]

        # if image contains pneumonia
        if filename in pneumonia_locations:
            # loop through pneumonia
            for location in pneumonia_locations [filename]:
                # add 1's at the location of the pneumonia
                x, y, w, h = location
                mask [y:y+h, x:x+w] = 1
        # if augment then horizontal flip half the time
        if self.augment and random.random() > 0.5:
            image = np.fliplr (image)
            mask = np.fliplr (mask)
        # resize both image and mask
        image = resize (image, (self.image_size, self.image_size), mode = 'symmetric')
        mask = resize (mask, (self.image_size, self.image_size), mode = 'symmetric') > 0.5
        mask = mask.astype (np.float32)
        
        # add trailing channel dimension
        image = np.expand_dims (image, -1)
        mask = np.expand_dims (mask, -1)
        return image, mask
    
    def __loadpredict__(self, filename):
        # load file as numpy array
        image = cv2.imread (os.path.join (self.folder, filename), cv2.IMREAD_GRAYSCALE)
        # resize image
        image = resize (image, (self.image_size, self.image_size), mode = 'symmetric')
        # add trailing channel dimension
        image = np.expand_dims (image, -1)
        return image
        
    def __getitem__(self, index):
        # select batch
        filenames = self.filenames [index * self.batch_size: (index+1) * self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            images = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            images = np.array (images)
            return images, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__ (filename) for filename in filenames]
            # unzip images and masks
            images, masks = zip (*items)
            # create numpy batch
            images = np.array (images)
            masks = np.array (masks)
            return images, masks
        
    def on_epoch_end (self):
        if self.shuffle:
            random.shuffle (self.filenames)
        
    def __len__(self):
        if self.predict:
            # return everything
            return int (np.ceil (len (self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int (len (self.filenames) / self.batch_size)