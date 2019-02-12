# -*- coding: utf-8 -*-
from model_3D import *
import time
from data_generator_3D import *
import os
from PIL import Image
from keras.preprocessing.image import *
import numpy as np
import xlwt

params = {'dim': (240,240,155),
          'batch_size': 1,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}




print('\n > Predicting the model...')
model=AtrousFCN_Resnet50_16s(input_shape=(240,240,155,1),weight_decay=0,batch_momentum=0.95,batch_shape=None,classes=2)
model.load_weights('/home/xjin/Brats_output/checkpoint_3d_12_16_175.hdf5')


test_image=np.load('/home/xjin/brats_dataset_3d/test_image_list.npy')
test_label=np.load('/home/xjin/brats_dataset_3d/test_label_list.npy')
#test_gen=DataGenerator(test_image,test_label,**params)



nb_test_samples = len(test_image)



start_time = time.time()
#y_predictions = model.predict_generator(test_gen, steps=nb_test_samples)

path0='/home/xjin/Brats_output/output_3d_12_16_175'
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
    print('current patient: '+patient_id)
    if os.path.exists(os.path.join(result_path,patient_id))==False:
        os.mkdir(os.path.join(result_path,patient_id))
    #label_id=os.path.splitext(os.path.split(path)[1])[0]
    if idx > nb_test_samples-1:
        continue
    
    '''
    x=load_img(test_image[idx])
    x=img_to_array(x,'channels_last')[:,:,1]
    (a,b)=np.shape(x)
    l=np.max(x)
    for i in range(a):
        for j in range(b):
            x[i][j]=x[i][j]/l

    X=x[np.newaxis,:,:,np.newaxis]
    '''
    x=np.load(test_image[idx])
    X=x[np.newaxis,:,:,:,np.newaxis]
    Y_pred=model.predict(X)


    y_pred=np.array(np.argmax(Y_pred, axis=-1))
    y_pred=np.squeeze(y_pred)
    y_truth=np.load(path)



    dice=2*np.sum(y_truth*y_pred)/(np.sum(y_truth)+np.sum(y_pred))
    Dice.append(dice)
    name.append(path)
    sheet.write(idx,0,path)
    sheet.write(idx,1,dice)
    # save as image
  
    
    for i in range(155):
        image=array_to_img(x[:,:,i:i+1])
        truth=array_to_img(y_truth[:,:,i:i+1])
        predicted=array_to_img(y_pred[:,:,i:i+1])
        image.save(os.path.join(result_path,patient_id, 'input_'+str(i+1)+'.png'))
        truth.save(os.path.join(result_path,patient_id,'truth_'+str(i+1)+'.png'))
        predicted.save(os.path.join(result_path,patient_id,'predicted_'+str(i+1)+'.png'))

print('mean dice coef= ',np.mean(Dice))
book.save(os.path.join(result_path,'dice.xls'))
total_time = time.time() - start_time
fps = float(nb_test_samples) / total_time
s_p_f = total_time / float(nb_test_samples)

f=open(os.path.join(path0,'info.txt'),'w')
text="down sampled along Z-axis"
f.write(text)
f.close()

print (' /n  Testing time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time, fps, s_p_f))
print ('/n Number of slices: '+str(nb_test_samples))

#np.save(os.path.join(result_path, 'y_pred'),y_predictions)