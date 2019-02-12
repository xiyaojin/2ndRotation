# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:35:29 2018

@author: xyjin213
"""

import os
from keras.utils import np_utils
import numpy as np
from data_generator_3D import *
from model_3D import *
import keras.backend as K
import tensorflow as tf
from loss_function import softmax_sparse_crossentropy_ignoring_last_label
from keras.optimizers import (Nadam, Adam, SGD)
from keras.callbacks import ModelCheckpoint
# Parameters
params = {'dim': (240,240,155),
          'batch_size': 1,
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
train_image=np.load('/home/xjin/brats_dataset_3d/train_image_list.npy')
train_label=np.load('/home/xjin/brats_dataset_3d/train_label_list.npy')
validation_image=np.load('/home/xjin/brats_dataset_3d/validation_image_list.npy')
validation_label=np.load('/home/xjin/brats_dataset_3d/validation_label_list.npy')

# Generators
training_generator = DataGenerator(train_image, train_label, **params)
validation_generator = DataGenerator(validation_image, validation_label, **params)


def dice_coef(y_true, y_pred):
    smooth=1.
    y_pred=K.reshape(y_pred,(-1,K.int_shape(y_pred)[-1]))
    softmax=tf.nn.softmax(y_pred)[:,1]
 #   y_true=K.one_hot(tf.to_int32(K.flatten(y_true)),K.int_shape(y_pred)[-1])[:,1]
    y_true=K.reshape(y_true,(-1,K.int_shape(y_pred)[-1]))[:,1]
    intersection = K.sum(y_true * softmax)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(softmax) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)
learning_rate_base = 0.01 * (float(params['batch_size']) / 16)
#opt = Adam(lr=learning_rate_base, beta_1=0.9, beta_2=0.999, epsilon=1e-08)s
opt = SGD(lr=learning_rate_base, momentum=0.9)
loss=dice_coef_loss
filepath='/home/xjin/Brats_output/checkpoint_3d_12_2_{epoch:02d}.hdf5'
checkpointer=ModelCheckpoint(filepath,verbose=1,period=20)
callbacks_list=[checkpointer]

#loss = softmax_sparse_crossentropy_ignoring_last_label
model=AtrousFCN_Resnet50_16s(input_shape=(240,240,155,1),weight_decay=0,batch_momentum=0.95,batch_shape=None,classes=2)
model.summary()
model.compile(loss=loss,optimizer=opt,metrics=['accuracy'])
#K.set_value(model.optimizer.lr,K.get_value(model.optimizer.lr)*1)
model.fit_generator(generator=training_generator, validation_data=validation_generator,callbacks=callbacks_list,epochs=1000,workers=4,verbose=1)

model.save_weights('/home/xjin/Brats_output/weights_3D_12_2.hdf5')
