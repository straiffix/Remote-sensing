# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:53:52 2019

@author: strai
"""

from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.models import model_from_json
from PIL import Image
import numpy as np
from keras.optimizers import rmsprop
import time
import os
from manage import file, colors, update_progress, sample_shape


json_file=open('model3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)  
loaded_model.load_weights("model3.h5")


loaded_model.compile(optimizer = rmsprop(lr=0.0001, decay=1e-6),
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])


test_file = file

test_image = Image.open(test_file)
#width, height = test_image.size
box = (0, 0, sample_shape, sample_shape)
width = test_image.size[0]
height = test_image.size[1]
#sample2 = test_image.crop(box)

result = Image.new('RGBA', (width, height))
result2 = test_image.copy()


for x in range(0, width, 3):
    for y in range(0, height, 3):
        if x+sample_shape < width:
            x2 = x + sample_shape
        else:
            break
        if y + sample_shape < height:
            y2 = y + sample_shape
        else:
            break
        
        box = (x, y, x2, y2)
        sample = test_image.crop(box)
        img = sample
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
#       
        res = loaded_model.predict(img_tensor)
        classx = np.argmax(res)
        xp = ((x2 - x) // 2) + x - 1
        yp = ((y2 - y) // 2) + y  - 1 
        point = Image.new('RGB', (5, 5), color = colors[classx])
        result.paste(point, (xp, yp))
    update_progress(x/width)
    time.sleep(0.5)
    
#        elif class_x == class_b:
#            point = Image.new('RGB', (10, 10), color = 'green')
#            result.paste(point, (xp, yp))
#    
result = result.convert("RGBA")
result2 = result2.convert("RGBA")
result2 = Image.blend(result, result2, 0.55)        
plt.imshow(result2 )
plt.imsave('result8.png', result2)


count = 1

for directory in os.listdir('dataset2/train'):
    point = Image.new('RGB', (30, 30), color = colors[count-1])
    plt.subplot(2, 4, count)
    plt.title(directory)
    plt.imshow(point)
    count +=1
    
    
