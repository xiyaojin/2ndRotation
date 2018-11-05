# -*- coding: utf-8 -*-



import numpy as np
import os
import SimpleITK as sitk



#image=sitk.ReadImage(path)
#I_array=sitk.GetArrayFromImage(image)
#I_array=I_array/np.max(I_array)

def list_all_files(rootdir):
    files=[]
    list = os.listdir(rootdir) 
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              files.extend(list_all_files(path))
           if os.path.isfile(path):
               if len(list)==1:
                   break
               if list[i]=='tice_hist.nii.gz':
                   files.append(path)
                   files_image.append(path)
               if list[i]=='truth.nii.gz':
                   files.append(path)
                   files_label.append(path)
                   
    return files

files_image=[]
files_label=[]

root_dir=r'E:histogram_matching_results'
files=list_all_files(root_dir)

train_percentage=0.9
train_image_list=files_image[0:int(len(files_image)*train_percentage)]
train_label_list=files_label[0:int(len(files_image)*train_percentage)]
test_image_list=files_image[int(len(files_image)*train_percentage):]
test_label_list=files_label[int(len(files_image)*train_percentage):]

np.save('e:train_image_list.npy',train_image_list)
np.save('e:train_label_list.npy',train_label_list)
np.save('e:test_image_list.npy',test_image_list)
np.save('e:test_label_list.npy',test_label_list)
