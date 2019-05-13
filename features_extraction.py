# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:44:11 2019

@author: strai
"""


from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras import models
from keras.models import model_from_json
import numpy as np


json_file=open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)  
loaded_model.load_weights("model.h5")

img_path = 'dataset/patterns/houses_1.png'
img = image.load_img(img_path, target_size=(120, 120))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
plt.imshow(img_tensor[0])
plt.show()
print(img_tensor.shape)

layer_outputs = [layer.output for layer in loaded_model.layers[:16]] 
# Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=loaded_model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
activations = activation_model.predict(img_tensor) 

layer_names = []
for layer in loaded_model.layers[:16]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

#images_per_row = 16
#for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
#    n_features = layer_activation.shape[-1] # Number of features in the feature map
#    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
#    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
#    display_grid = np.zeros((size * n_cols, images_per_row * size))
#    for col in range(n_cols): # Tiles each filter into a big horizontal grid
#        for row in range(images_per_row):
#            channel_image = layer_activation[0,
#                                             :, :,
#                                             col * images_per_row + row]
#            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
#            channel_image /= channel_image.std()
#            channel_image *= 64
#            channel_image += 128
#            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#            display_grid[col * size : (col + 1) * size, # Displays the grid
#                         row * size : (row + 1) * size] = channel_image
#    scale = 1. / size
#    plt.figure(figsize=(scale * display_grid.shape[1],
#                        scale * display_grid.shape[0]))
#    plt.title(layer_name)
#    plt.grid(False)
#    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    lactiv = activations[4]
channel_image = lactiv[0, :, :, 5]
channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
channel_image /= channel_image.std()
channel_image *= 64
channel_image += 128
channel_image = np.clip(channel_image, 0, 255).astype('uint8')
plt.imshow(channel_image, cmap='inferno')
