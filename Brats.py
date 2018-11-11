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
import keras.backend as K

# Parameters
params = {'dim': (240,240,1),
          'batch_size': 8,
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
train_image=np.load('/home/xjin/brats_dataset_slices/train_image_list.npy')
train_label=np.load('/home/xjin/brats_dataset_slices/train_label_list.npy')
validation_image=np.load('/home/xjin/brats_dataset_slices/validation_image_list.npy')
validation_label=np.load('/home/xjin/brats_dataset_slices/validation_label_list.npy')

# Generators
training_generator = DataGenerator(train_image, train_label, **params)
validation_generator = DataGenerator(validation_image, validation_label, **params)

smooth=1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


model=AtrousFCN_Resnet50_16s(input_shape=(240,240,1),weight_decay=0.0005,batch_momentum=0.95,batch_shape=None,classes=2)
model.compile(loss=dice_coef_loss,optimizer='adam',metrics=['accuracy'])
K.set_value(model.optimizer.lr,K.get_value(model.optimizer.lr)*1)
model.fit_generator(generator=training_generator, validation_data=validation_generator,epochs=5,workers=4,verbose=1)

model.save_weights('/home/xjin/brats_dataset/weights_11_10_3.hdf5')
