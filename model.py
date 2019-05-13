# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:18:26 2019

@author: strai
"""

from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from functools import partial


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape = (112, 112, 3), activation = 'relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5)) # antes era 0.25

# Adding a second convolutional layer
model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5)) # antes era 0.25

# Adding a third convolutional layer
model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5)) # antes era 0.25

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dropout(0.5)) 
model.add(Dense(units = 3, activation = 'relu'))

#model = Sequential()
#model.add(Conv2D(32, (3, 3), input_shape=(120, 120, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
#model.add(Dense(64))
#model.add(Activation('relu'))
#
#model.add(Dropout(0.5))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))
#
model.compile(optimizer = 'rmsprop',
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

batch_size = 16

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(112, 112),
        batch_size = 16,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'dataset/validation',
        target_size=(112, 112),
        batch_size=16,
        class_mode = 'categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // 16,
        epochs = 50,
        validation_data=validation_generator,
        validation_steps= 800 // 16)

model_json=model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
