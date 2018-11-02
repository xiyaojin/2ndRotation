# -*- coding: utf-8 -*-



import numpy as np
import os
import SimpleITK as sitk

X_train=[]
Y_train=[]

def list_all_files(rootdir):
    global X_train
    global Y_train
    files = []
    list = os.listdir(rootdir) 
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              files.extend(list_all_files(path))
           if os.path.isfile(path):
              if list[i]=='t1ce_hist.nii.gz':
                  image=sitk.ReadImage(path)
                  I_array=sitk.GetArrayFromImage(image)
                  if X_train==[]:
                      X_train=I_array
                  else:
                      X_train=np.concatenate((X_train,I_array),axis=0)
              if list[i]=='truth.nii.gz':
                  label=sitk.ReadImage(path)
                  L_array=sitk.GetArrayFromImage(label)
                  L_array[L_array!=1]=0
                  if Y_train==[]:
                      Y_train=L_array
                  else:
                      Y_train=np.concatenate((Y_train,L_array),axis=0)
                  
              files.append(path)
    return files


output_image_path=r'E:\WashU-Lab\2ndRotation\x_train.npy'
output_label_path=r'E:\WashU-Lab\2ndRotation\y_train.npy'
#np.save(output_image_path,I_array)
#np.save(output_label_path,L_array) 



root_dir='E:\histogram_matching_results\histogram_matching_results'
files=list_all_files(root_dir)
