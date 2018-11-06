# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:35:29 2018

@author: xyjin213
"""

import os
from keras.utils import np_utils
import numpy as np
from data_generator import *
from model import *

# Parameters
params = {'dim': (240,240,1),
          'batch_size': 16,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}

# Datasets
'''
train_image=np.load('e:train_image_list.npy')
train_label=np.load('e:train_label_list.npy')
validation_image=np.load('e:validation_image_list.npy')
validation_label=np.load('e:validation_label_list.npy')
'''
train_image=np.load('/home/xjin/brats_dataset/train_image_list.npy')
train_label=np.load('/home/xjin/brats_dataset/train_label_list.npy')
validation_image=np.load('/home/xjin/brats_dataset/validation_image_list.npy')
validation_label=np.load('/home/xjin/brats_dataset/validation_label_list.npy')
# Generators
training_generator = DataGenerator(train_image, train_label, **params)
validation_generator = DataGenerator(validation_image, validation_label, **params)





model=AtrousFCN_Resnet50_16s(input_shape=(240,240,1),weight_decay=0.00005,batch_momentum=0.95,batch_shape=None,classes=2)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit_generator(generator=training_generator, validation_data=validation_generator,epochs=5,workers=4,verbose=1)

model.save_weights('/home/xjin/brats_dataset/weights.hdf5')