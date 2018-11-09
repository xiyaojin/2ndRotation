# -*- coding: utf-8 -*-



import numpy as np
import os
import SimpleITK as sitk
from PIL import Image
from keras.preprocessing.image import *


#image=sitk.ReadImage(path)
#I_array=sitk.GetArrayFromImage(image)
#I_array=I_array/np.max(I_array)
'''
def list_all_files(rootdir):
    files=[]
    list = os.listdir(rootdir) 
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              files.extend(list_all_files(path))
           if os.path.isfile(path):
               if 'tice_hist.nii.gz' in list and 'truth.nii.gz' in list:
                   if list[i]=='tice_hist.nii.gz':
                       files.append(path)
                       I=sitk.ReadImage(path)
                       I=sitk.RescaleIntensity(I,0,1)
                       I_array=sitk.GetArrayFromImage(I)
                       
                       
                       
                       z=np.shape(I_array)[0]
                       
                       
                       for j in range(z):
                           write_path=os.path.join(rootdir,'image_'+str(j)+'.npy')
                           if os.path.exists(write_path) == False:
                               np.save(write_path,I_array[j,:,:])
                           elif np.max(I_array[j:,:])==0:
                               os.remove(write_path)
                               continue
                           slices_image.append(write_path)
                       del I_array,I
                    
                       #files_image.append(path)
                   if list[i]=='truth.nii.gz':
                       files.append(path)
                       L=sitk.ReadImage(path)
                       L=sitk.RescaleIntensity(L,0,1)
                       L_array=sitk.GetArrayFromImage(L)
                       z=np.shape(L_array)[0]
                       for j in range(z):
                           write_path=os.path.join(rootdir,'label_'+str(j)+'.npy')
                           if os.path.exists (write_path) == False:
                               np.save(write_path,L_array[j,:,:])
                           slices_label.append(write_path)
                       del L_array,L    
                      #files_label.append(path)
    del list
    return files
'''

    
slices_image=[]
slices_label=[]

root_dir=r'E:test'
#files=list_all_files(root_dir)
list=os.listdir(root_dir)
output_dir=r'E:test2'
for i in list:
    new_root=os.path.join(root_dir,i)
    if os.path.isdir(new_root):
        sub_list=os.listdir(new_root)
        if 'tice_hist.nii.gz' in sub_list and 'truth.nii.gz' in sub_list:
            for j in sub_list:
                if j=='tice_hist.nii.gz':
                    I=sitk.ReadImage(os.path.join(new_root,'tice_hist.nii.gz'))
                    I=sitk.RescaleIntensity(I,0,1)
                    I_array=sitk.GetArrayFromImage(I)
                    L=sitk.ReadImage(os.path.join(new_root,'truth.nii.gz'))
                    L_array=sitk.GetArrayFromImage(L)
                    L_array[L_array!=1]=0
                    z=np.shape(L_array)[0]
                    for k in range(z):
                        if np.max(L_array[k,:,:])!=0:
                            image=array_to_img(I_array[k,:,:][:,:,np.newaxis],'channels_last')
                            label=array_to_img(L_array[k,:,:][:,:,np.newaxis],'channels_last')
                            image_path=os.path.join(output_dir,i,'image_'+str(k)+'.png')
                            if os.path.exists(os.path.join(output_dir,i))==False:
                                os.mkdir(os.path.join(output_dir,i))                        
                            image.save(image_path)
                            label_path=os.path.join(output_dir,i,'label_'+str(k)+'.png')
                            label.save(label_path)
                            slices_image.append(image_path)
                            slices_label.append(label_path)


train_percentage=0.9
train_image_list=slices_image[0:int(len(slices_image)*train_percentage)]
train_label_list=slices_label[0:int(len(slices_image)*train_percentage)]
validation_image_list=slices_image[int(len(slices_image)*train_percentage):int(len(slices_image)*0.97)]
validation_label_list=slices_label[int(len(slices_image)*train_percentage):int(len(slices_image)*0.97)]
test_image_list=slices_image[int(len(slices_image)*0.97):]
test_label_list=slices_label[int(len(slices_image)*0.97):]

'''
np.save('e:train_image_list.npy',train_image_list)
np.save('e:train_label_list.npy',train_label_list)
np.save('e:validation_image_list.npy',validation_image_list)
np.save('e:validation_label_list.npy',validation_label_list)
np.save('e:test_image_list.npy',test_image_list)
np.save('e:test_label_list.npy',test_label_list)

np.save('/home/xjin/brats_dataset/train_image_list.npy',train_image_list)
np.save('/home/xjin/brats_dataset/train_label_list.npy',train_label_list)
np.save('/home/xjin/brats_dataset/validation_image_list.npy',validation_image_list)
np.save('/home/xjin/brats_dataset/validation_label_list.npy',validation_label_list)
np.save('/home/xjin/brats_dataset/test_image_list.npy',test_image_list)
np.save('/home/xjin/brats_dataset/test_label_list.npy',test_label_list)
'''
