# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:36:58 2019

@author: strai
"""


from PIL import Image
import numpy as np
from skimage import transform

import sys

def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (60, 60, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def preprocess(image):
    np_image = image
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (60, 60, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


colors = ('teal', 'navy', 'darkgreen', 'sienna', 'plum', 'aqua', 'darkmagenta', 'palegreen', 'mediumslateblue'  )

batch_size = 16
n_classes = 8
sample_shape = 60 

train_directory = 'dataset2/train'

file = 'dataset2/530m_4.png'
