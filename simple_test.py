# -*- coding: utf-8 -*-


from model import *
import numpy as np
import matplotlib as plt
'''
x=np.load(r'e:test\Brats18_2013_0_1\image_80.npy')
model=AtrousFCN_Resnet50_16s(input_shape=(240,240,1),weight_decay=0.00005,batch_momentum=0.95,batch_shape=None,classes=2)
model.load_weights(r'e:weights.hdf5')
x=x[np.newaxis,:,:,np.newaxis]
pred=model.predict(x)
'''
Pred=np.squeeze(pred)
slice1=Pred[:,:,1]
plt.pyplot.imshow(slice1)

