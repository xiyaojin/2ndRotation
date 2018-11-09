# -*- coding: utf-8 -*-

import numpy as np
import keras
import SimpleITK as sitk
from PIL import Image
from keras.preprocessing.image import *


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(240,240,1), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        labels_temp   = [self.labels[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp,labels_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:

    def __data_generation(self, list_IDs_temp,labels_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.dim), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image=load_img(ID)
            I=img_to_array(image,'channels_last')
            label=load_img(labels_temp[i])
            L=img_to_array(label,'channels_last')
            X[i,]=I
            y[i,]=L
            # Store class

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
