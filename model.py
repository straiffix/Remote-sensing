# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:18:26 2019

@author: strai
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import rmsprop



n_classes = 3
#model = Sequential()

# Layer 1
#
#model.add(Conv2D(96, (11, 11), input_shape=(224, 224, 3), 
#                 padding='same', kernel_regularizer=l2(0.)))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#
## Layer 2
#model.add(Conv2D(256, (5, 5), padding='same'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#	# Layer 3
#model.add(ZeroPadding2D((1, 1)))
#model.add(Conv2D(512, (3, 3), padding='same'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#	# Layer 4
#model.add(ZeroPadding2D((1, 1)))
#model.add(Conv2D(1024, (3, 3), padding='same'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#
#	# Layer 5
#model.add(ZeroPadding2D((1, 1)))
#model.add(Conv2D(1024, (3, 3), padding='same'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#	# Layer 6
#model.add(Flatten())
#model.add(Dense(3072))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#
#	# Layer 7
#model.add(Dense(4096))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#
#	# Layer 8
#model.add(Dense(n_classes))
#model.add(BatchNormalization())
#model.add(Activation('softmax'))

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
#model.add(Dense(3))
#model.add(Activation('softmax'))

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(120, 120, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer = rmsprop(lr=0.0001, decay=1e-6),
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

batch_size = 16
#batch_size = 128

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(120, 120),
        batch_size = 16,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'dataset/validation',
        target_size=(120, 120),
        batch_size=16,
        class_mode = 'categorical')
#
model.fit_generator(
        train_generator,
        steps_per_epoch=701 // 16,
        epochs = 15,
        validation_data=validation_generator,
        validation_steps= 79 // 16
        )
#validation_steps= 800 // 128

model_json=model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
