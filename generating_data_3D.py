# -*- coding: utf-8 -*-



import numpy as np
import os
import SimpleITK as sitk
  
image_list=[]
label_list=[]

root_dir='/home/xjin/brats_dataset/histogram_matching_results'
#files=list_all_files(root_dir)
list=os.listdir(root_dir)
output_dir='/home/xjin/brats_dataset_3d'
if os.path.exists(output_dir)==False:
    os.mkdir(output_dir)
for i in list:
    new_root=os.path.join(root_dir,i)
    if os.path.isdir(new_root):
        sub_list=os.listdir(new_root)
        if 'tice_hist.nii.gz' in sub_list and 'truth.nii.gz' in sub_list:
            if os.path.exists(os.path.join(output_dir,i))==False:
                os.mkdir(os.path.join(output_dir,i))
            print (i)
            I=sitk.ReadImage(os.path.join(new_root,'tice_hist.nii.gz'))
            I=sitk.RescaleIntensity(I,0,1)
            I_array=sitk.GetArrayFromImage(I)
            I_array=np.moveaxis(I_array,0,-1)
            np.save(os.path.join(output_dir,i,'input.npy'),I_array)
            image_list.append(os.path.join(output_dir,i,'input.npy'))
            del I,I_array
            
            L=sitk.ReadImage(os.path.join(new_root,'truth.nii.gz'))
            L_array=sitk.GetArrayFromImage(L)
            L_array=np.moveaxis(L_array,0,-1)
            L_array[L_array==4]=1
            L_array[L_array!=1]=0
            np.save(os.path.join(output_dir,i,'truth.npy'),L_array)
            label_list.append(os.path.join(output_dir,i,'truth.npy'))
            del L,L_array


train_percentage=0.9
train_image_list=image_list[0:int(len(image_list)*train_percentage)]
train_label_list=label_list[0:int(len(image_list)*train_percentage)]
validation_image_list=image_list[int(len(image_list)*train_percentage):int(len(image_list)*0.97)]
validation_label_list=label_list[int(len(image_list)*train_percentage):int(len(image_list)*0.97)]
test_image_list=image_list[int(len(image_list)*0.97):]
test_label_list=label_list[int(len(image_list)*0.97):]


np.save('/home/xjin/brats_dataset_3d/train_image_list.npy',train_image_list)
np.save('/home/xjin/brats_dataset_3d/train_label_list.npy',train_label_list)
np.save('/home/xjin/brats_dataset_3d/validation_image_list.npy',validation_image_list)
np.save('/home/xjin/brats_dataset_3d/validation_label_list.npy',validation_label_list)
np.save('/home/xjin/brats_dataset_3d/test_image_list.npy',test_image_list)
np.save('/home/xjin/brats_dataset_3d/test_label_list.npy',test_label_list)

