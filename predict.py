# -*- coding: utf-8 -*-
from model import *
import time
from data_generator import *
import os
from PIL import Image
from keras.preprocessing.image import *
import numpy as np
import xlwt

params = {'dim': (240,240,1),
          'batch_size': 1,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}




print('\n > Predicting the model...')
model=AtrousFCN_Resnet50_16s(input_shape=(240,240,1),weight_decay=0,batch_momentum=0.95,batch_shape=None,classes=2)
model.load_weights('/home/xjin/Brats_output/checkpoint_1_4_label-40.hdf5')


test_image=np.load('/home/xjin/brats_dataset_1_4_label/test_image_list.npy')
test_label=np.load('/home/xjin/brats_dataset_1_4_label/test_label_list.npy')
#test_gen=DataGenerator(test_image,test_label,**params)



nb_test_samples = len(test_image)



start_time = time.time()
#y_predictions = model.predict_generator(test_gen, steps=nb_test_samples)

path0='/home/xjin/Brats_output/output_11_29_2'
if os.path.exists(path0)==False:
    os.mkdir(path0)
result_path = os.path.join(path0,'predicted_labels')
if os.path.exists(result_path) == False:
    os.mkdir(result_path)
    
results = []
name=[]
Dice=[]

book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('test', cell_overwrite_ok=True)
for (idx, path) in enumerate(test_label):
    
    directory=os.path.split(path)[0]
    patient_id=os.path.split(directory)[1]
    if os.path.exists(os.path.join(result_path,patient_id))==False:
        os.mkdir(os.path.join(result_path,patient_id))
    label_id=os.path.splitext(os.path.split(path)[1])[0]
    if idx > nb_test_samples-1:
        continue
    
    x=load_img(test_image[idx])
    x=img_to_array(x,'channels_last')[:,:,1]
    (a,b)=np.shape(x)
    l=np.max(x)
    for i in range(a):
        for j in range(b):
            x[i][j]=x[i][j]/l

    X=x[np.newaxis,:,:,np.newaxis]
    Y_pred=model.predict(X)

    y_sample_prediction = np.squeeze(Y_pred)  
    y_pred=np.array(np.argmax(y_sample_prediction, axis=-1))
    y_truth=load_img(path)
    y_truth=img_to_array(y_truth,'channels_last')[:,:,1]
    (a,b)=np.shape(y_truth)
    for i in range(a):
        for j in range(b):
            y_truth[i][j]=y_truth[i][j]/255
    y_truth=np.array(y_truth)
    dice=2*np.sum(y_truth*y_pred)/(np.sum(y_truth)+np.sum(y_pred))
    Dice.append(dice)
    name.append(path)
    sheet.write(idx,0,path)
    sheet.write(idx,1,dice)
    # save as image
    result = np.expand_dims(np.argmax(y_sample_prediction, axis=-1), axis=-1)    
    result_img = array_to_img(result)
    
    result_img.save(os.path.join(result_path, patient_id, label_id + '.png'))

print('mean dice coef= ',np.mean(Dice))
book.save(os.path.join(result_path,'dice.xls'))
total_time = time.time() - start_time
fps = float(nb_test_samples) / total_time
s_p_f = total_time / float(nb_test_samples)
f=open(os.path.join(path0,'info.txt'),'w')
text="use 'predict' method instead of generator"
f.write(text)
f.close()

print (' /n  Testing time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time, fps, s_p_f))
print ('/n Number of slices: '+str(nb_test_samples))

#np.save(os.path.join(result_path, 'y_pred'),y_predictions)
