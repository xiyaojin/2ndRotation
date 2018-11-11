# -*- coding: utf-8 -*-


from model import *
import numpy as np
import matplotlib as plt
from keras.preprocessing.image import *

img=load_img(r'e:test2\Brats18_2013_0_1\image_61.png')
x=img_to_array(img,'channels_last')[:,:,1]
(a,b)=np.shape(x)
l=np.max(x)
for i in range(a):
    for j in range(b):
        x[i][j]=x[i][j]/l
model=AtrousFCN_Resnet50_16s(input_shape=(240,240,1),weight_decay=0.00005,batch_momentum=0.95,batch_shape=None,classes=2)
model.load_weights(r'e:weights_11_10_2.hdf5')
x=x[np.newaxis,:,:,np.newaxis]
pred=model.predict(x)

Pred=np.squeeze(pred)
result = np.expand_dims(np.argmax(Pred, axis=-1), axis=-1)
#plt.pyplot.imshow(Pred)

