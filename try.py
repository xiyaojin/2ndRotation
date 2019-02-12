# -*- coding: utf-8 -*-
from model_3D_Z import *
import numpy as np
import matplotlib as plt
from keras.preprocessing.image import *
#import keras
from tensorflow.python.keras import *
import tensorflow as tf
import keras.backend as K
import os

x=np.load('/home/xjin/brats_dataset_3d/Brats18_CBICA_AQD_1/input.npy')

y=np.load('/home/xjin/brats_dataset_3d/Brats18_CBICA_AQD_1/truth.npy')
y=img_to_array(y,'channels_last')

#plt.pyplot.imshow(x)
model=AtrousFCN_Resnet50_16s(input_shape=(240,240,155,1),weight_decay=0.000,batch_momentum=0.95,batch_shape=None,classes=2)
model.load_weights('/home/xjin/Brats_output/checkpoint_3d_12_13_2_20.hdf5',by_name=True)
X=x[np.newaxis,:,:,:,np.newaxis]
Y_pred=model.predict(X)

result = np.argmax(np.squeeze(Y_pred), axis=-1)

for i in range(155):
    image=array_to_img(x[:,:,i:i+1])
    truth=array_to_img(y[:,:,i:i+1])
    predicted=array_to_img(result[:,:,i:i+1])
    path='/home/xjin/brats_3d_test/Brats18_CBICA_AQD_1/12_13_2_20'
    if os.path.exists(path)==False:
        os.mkdir(path)
    image.save(os.path.join(path,'input_'+str(i+1)+'.png'))
    truth.save(os.path.join(path,'truth_'+str(i+1)+'.png'))
    predicted.save(os.path.join(path,'predicted_'+str(i+1)+'.png'))


'''

y_true=keras.utils.to_categorical(y, num_classes=2)
#plt.pyplot.imshow(y_true[:,:,0])
smooth=1.

y_pred=K.reshape(Y_pred,(-1,np.shape(Y_pred)[-1]))
#log_softmax=tf.nn.log_softmax(y_pred)
softmax=tf.nn.softmax(y_pred)
softmax=K.argmax(softmax,axis=-1)
softmax=K.cast(softmax,'float32')
y_true=K.reshape(y_true,(-1,K.int_shape(y_pred)[-1]))[:,1]
#y_true=K.reshape(y_true,(-1,np.shape(y_true)[-1]))
#unpacked = tf.unstack(y_true, axis=-1)
#y_true1 = tf.stack(unpacked[:-1], axis=-1)

intersection = K.sum(y_true * softmax)
dice=(2. * intersection + smooth) / (K.sum(softmax) + K.sum(y_true) + smooth)

with tf.Session():
    Y_true=y_true.eval()
    Sum_pred=K.sum(softmax).eval()
    Sum_true=K.sum(y_true).eval()
    Intersection=intersection.eval()
#    Mult=(y_true * softmax).eval()
    Dice=dice.eval()
'''
