# -*- coding: utf-8 -*-




from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
import keras.backend as K
import tensorflow as tf
from resnet_helpers import *
from bilinear_upsampling_3D import *


def AtrousFCN_Resnet50_16s(input_shape = None, weight_decay=0, batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:4]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:3]

    bn_axis = 4
    nb_filters=[16, 16, 16]
    x = Conv3D(64, (7, 7, 7), strides=(2, 2, 1), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 1),padding='same')(x)

    x = conv_block3D(3, nb_filters, stage=2, block='a', weight_decay=weight_decay, strides=(1, 1, 1), batch_momentum=batch_momentum)(x)
    x = identity_block3D(3, nb_filters, stage=2, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block3D(3, nb_filters, stage=2, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block3D(3, nb_filters, stage=3, block='a', weight_decay=weight_decay, strides=(2, 2, 1),batch_momentum=batch_momentum)(x)
    x = identity_block3D(3, nb_filters, stage=3, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block3D(3, nb_filters, stage=3, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block3D(3, nb_filters, stage=3, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block3D(3, nb_filters, stage=4, block='a', weight_decay=weight_decay, strides=(2, 2, 1),batch_momentum=batch_momentum)(x)
    x = identity_block3D(3, nb_filters, stage=4, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block3D(3, nb_filters, stage=4, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block3D(3, nb_filters, stage=4, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block3D(3, nb_filters, stage=4, block='e', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block3D(3, nb_filters, stage=4, block='f', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    
    x = atrous_conv_block3D(3, nb_filters, stage=5, block='a', weight_decay=weight_decay, atrous_rate=(2, 2, 2), batch_momentum=batch_momentum)(x)
    #x = UpSampling3D(size=(4,4,1))(x)
    x = atrous_identity_block3D(3, [8, 8, 16], stage=5, block='b', weight_decay=weight_decay, atrous_rate=(2, 2, 2), batch_momentum=batch_momentum)(x)
   # x = UpSampling3D(size=(4,4,1))(x)
    x = atrous_identity_block3D(3, [8, 8, 16], stage=5, block='c', weight_decay=weight_decay, atrous_rate=(2, 2, 2), batch_momentum=batch_momentum)(x)
    #classifying layer
    #x = Conv2D(classes, (3, 3), dilation_rate=(2, 2), kernel_initializer='normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(classes, (1, 1, 1), kernel_initializer='he_normal', activation='linear', padding='same', strides=(1, 1, 1), kernel_regularizer=l2(weight_decay))(x)
    #x = Conv3DTranspose(classes,(1,1,1),strides=(1,1,1),padding='same',dilation_rate=(16,16,1))(x)
    x = BilinearUpSampling3D(target_size=tuple(image_size))(x)
    model = Model(img_input, x)
    #weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    #model.load_weights(weights_path, by_name=True)
    return model