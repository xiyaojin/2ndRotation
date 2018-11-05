# -*- coding: utf-8 -*-

import numpy as np
import keras
import SimpleITK as sitk

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
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp,labels_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        temp_file=sitk.ReadImage(list_IDs_temp[0])
        temp_array=sitk.GetArrayFromImage(temp_file)
        slices=np.shape(temp_array)[0]
        z=self.batch_size*slices
        X = np.empty((self.batch_size*z, *self.dim))
        y = np.empty((self.batch_size*z, *self.dim), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image=sitk.ReadImage(ID)
            I=sitk.GetArrayFromImage(image)
           
            label=sitk.ReadImage(labels_temp[i])
            L=sitk.GetArrayFromImage(label)
            L[L!=1]=0
            # Store class
            for j in range(slices):
                X[i*slices+j,:,:,0]=I[j,:,:]
                y[i*slices+j,:,:,0]=L[j,:,:]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
