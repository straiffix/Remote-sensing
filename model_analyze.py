# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:45:09 2019

@author: strai
"""

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import rmsprop
import os
import pandas as pd
import numpy as np
import seaborn as sns

from manage import sample_shape, n_classes, batch_size, load

json_file=open('model3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)  
loaded_model.load_weights("model3.h5")


loaded_model.compile(optimizer = rmsprop(lr=0.0001, decay=1e-6),
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])




test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'dataset2/test',
        target_size=(sample_shape, sample_shape),
        batch_size = batch_size,
        class_mode='categorical')

#scores = loaded_model.evaluate_generator(test_generator, 117 )
#print("Accuracy = ", scores[1])

n_classes1 = 0
count1 = 0
original_class = pd.DataFrame(columns = ['sample', 'classx'])
for directory in os.listdir('dataset2/test'):
    for file in os.listdir('dataset2/test/%s' % directory):
        original_class.loc[count1] = (directory, n_classes1)
        count1 +=1
    n_classes1 +=1
    
n_classes2 = 0
count2 = 0

predicted_class = pd.DataFrame(columns = ['sample', 'classy'])
for directory in os.listdir('dataset2/test'):
    for file in os.listdir('dataset2/test/%s' % directory):
        img = load('dataset2/test/%s/%s'% (directory, file))
        classx = np.argmax(loaded_model.predict(img))
        predicted_class.loc[count2] = (directory, classx)
        count2 +=1

errors = pd.crosstab(predicted_class.classy, original_class.classx)
sns.heatmap(errors, annot = errors)
