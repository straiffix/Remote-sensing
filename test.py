# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:53:52 2019

@author: strai
"""

from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator,  load_img, img_to_array
from keras.models import model_from_json
from PIL import Image
import numpy as np
from keras import optimizers
from skimage import transform
from scipy import ndimage, misc
from keras.optimizers import rmsprop


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


loaded_model.compile(optimizer = rmsprop(lr=0.0001, decay=1e-6),
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])


#img = image.load_img('dataset/patterns/houses_6.png', target_size=(120, 120))
#img1 = image.img_to_array(img)
#img1 = np.expand_dims(img, axis=0)
#
#res = loaded_model.predict(img1)
#if res[0][0] == 1:
#    point = Image.new('RGB', (5, 5), color = 'red')
#    img.paste(point, (58, 58))
#    plt.imsave('point.png', img)
#    
#elif res[0][1] == 1:
#    point = Image.new('RGB', (5, 5), color = 'green')
#    img.paste(point, (58, 58))
#    plt.imsave('point.png', img)
#    
#else: 
#    point = Image.new('RGB', (5, 5), color = 'blue')
#    img.paste(point, (58, 58))
#    plt.imsave('point.png', img)
#    


test_file = 'dataset/3113m_1.png'
cpsize = 120
test_image = Image.open(test_file)
#width, height = test_image.size
box = (0, 0, 120, 120)
width = test_image.size[0]
height = test_image.size[1]
sample2 = test_image.crop(box)

result = Image.new('RGBA', (width, height))




for x in range(0, width, 10):
    for y in range(0, height, 10):
        if x+cpsize < width:
            x2 = x + cpsize
        else:
            break
        if y + cpsize < height:
            y2 = y + cpsize
        else:
            break
        
        box = (x, y, x2, y2)
        sample = test_image.crop(box)
        img = sample
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
#       
#        layer_outputs = [layer.output for layer in loaded_model.layers[:16]] 
## Extracts the outputs of the top 12 layers
#        activation_model = models.Model(inputs=loaded_model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
#        activations = activation_model.predict(img_tensor) 
#
#        layer_names = []
#        for layer in loaded_model.layers[:16]:
#            layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
#        lactiv = activations[4]
#        channel_image = lactiv[0, :, :, 5]
#        channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
#        channel_image /= channel_image.std()
#        channel_image *= 64
#        channel_image += 128
#        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#        plt.imsave('ci.png', channel_image, cmap='inferno' )
#        ci = Image.open('ci.png')
#        ci = ci.resize((120, 120))
#        result.paste(ci, (x, y))
#        ci.close()
        
        res = loaded_model.predict(img_tensor)
        class_a = res[0][0]
        class_b = res[0][1]
        class_c = res[0][2]
        class_x = max(class_a, class_b, class_c)
        xp = ((x2 - x) // 2) + x - 5
        yp =  ((y2 -y) // 2) + y  - 5 

        if class_x == class_a:
            point = Image.new('RGB', (10, 10), color = 'red')
            result.paste(point, (xp, yp))
    
        elif class_x == class_b:
            point = Image.new('RGB', (10, 10), color = 'green')
            result.paste(point, (xp, yp))
    
        else: 
            point = Image.new('RGB', (10, 10), color = 'blue')
            result.paste(point, (xp, yp))
        
        
plt.imshow(result )
plt.imsave('result2.png', result)
