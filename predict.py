# -*- coding: utf-8 -*-
from model import *
import time
from data_generator import *
import os
from PIL import Image
from keras.preprocessing.image import *

params = {'dim': (240,240,1),
          'batch_size': 16,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}




print('\n > Predicting the model...')
model=AtrousFCN_Resnet50_16s(input_shape=(240,240,1),weight_decay=0.00005,batch_momentum=0.95,batch_shape=None,classes=2)
model.load_weights('/home/xjin/brats_dataset/weights.hdf5')


test_image=np.load('/home/xjin/brats_dataset/test_image_list.npy')
test_label=np.load('/home/xjin/brats_dataset/test_label_list.npy')
test_gen=DateGenerator(test_image,test_label,**params)



nb_test_samples = len(test_image)



start_time = time.time()
y_predictions = self.model.predict_generator(test_gen, steps=nb_test_samples)
total_time = time.time() - start_time
fps = float(nb_test_samples) / total_time
s_p_f = total_time / float(nb_test_samples)
print ('   Testing time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time, fps, s_p_f))

result_path = os.path.join('/home/xjin/brat_dataset/output/','predicted_labels')
if os.path.exists(result_path) == False:
    os.mkdir(result_path)
    
results = []
for (idx, path) in enumerate(test_label):
    directory=os.path.split(path)[0]
    patient_id=os.path.split(directory)[1]
    if os.path.exists(os.path.join(result_path,patient_id))==False:
        os.mkdir(os.path.joint(result_path,patient_id))
    label_id=os.path.split(os.path.split(path)[1])[0]
    if idx > nb_test_samples-1:
        continue
    y_sample_prediction = y_predictions[idx,:,:,:]

    # print('sample: ' + img_num + ', idx: ' + str(idx))
    # print('min: ' + str(np.min(y_sample_prediction)))
    # print('max: ' + str(np.max(y_sample_prediction)))

    # compress to top probabilty
    result = np.expand_dims(np.argmax(y_sample_prediction, axis=-1), axis=-1)

    # save as image
    result_img = array_to_img(result)
    
    result_img.save(os.path.join(result_path, patient_id, label_id + '.png'))

np.save(os.path.join(result_path, 'y_pred'),y_predictions)