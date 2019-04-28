# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:53:52 2019

@author: strai
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from PIL import Image
import numpy as np
from keras import optimizers
from skimage import transform


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (120, 120, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def preprocess(image):
    np_image = image
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (120, 120, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


json_file=open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)  
loaded_model.load_weights("model.h5")
image = load('dataset/1/5.png')

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#loaded_model.compile(loss='mean_squared_error', optimizer=sgd)
#print(loaded_model.get_weights())



test_file = 'dataset/2800m_2.png'
cpsize = 120
test_image = Image.open(test_file)
#width, height = test_image.size
width = 200
height = 200

box = (0, 0, 120, 120)
sample = test_image.crop(box)
sample1 = preprocess(sample)
print(loaded_model.predict(sample1))
sample1 = np.multiply(sample1, loaded_model.get_weights())


"""for x1 in range(0, width, 10):
    for y1 in range(0, height, 10):
        box = (x1, y1, x1+cpsize if x1+cpsize < width else width - 1,
               y1+cpsize if y1+cpsize < height else height - 1)
        sample = test_image.crop(box)
        sample.show()
        sample = preprocess(sample)
        
        print(loaded_model.predict(sample))"""
