# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:25:51 2019

@author: strai
"""
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('dataset/patterns/houses_13.png')
x = img_to_array(img)
x = x.reshape((1,) + x.shape) 

i = 0
for batch in train_gen.flow(x, batch_size=1, save_to_dir='dataset/preview', save_prefix='houses', save_format='jpeg'):
    i +=1
    if i > 20:
        break 
    
"""file = '1.png'
cpsize = 150;
imgg = Image.open(file)
width, height = imgg.size

for i in range(0, width):
    for j in range(0, height):
        box = (i, j, i + cpsize if i + cpsize < width else width - 1, j + cpsize if j + cpsize < height else
               height - 1 )
        print('%s %s' % (file, box))
        imgg.crop(box).save('dataset/test_data/zc.%s.x%03d.y%03d.jpg' % (file.replace('.png', ''), i, j))"""


'''imgg = Image.open(file)
width, height = imgg.size

for x1 in range(0, width, cpsize):
    for y1 in range(0, height, cpsize):
        box = (x1, y1, x1+cpsize if x1+cpsize < width else width - 1,
               y1+cpsize if y1+cpsize < height else height - 1)
        print('%s %s' % (file, box))
        imgg.crop(box).save('dataset/test_data/zc.%s.x%03d.y%03d.jpg' % (file.replace('.png', ''), x1, y1))'''