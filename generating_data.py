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
               if 'tice_hist.nii.gz' in list and 'truth.nii.gz' in list:
                   if list[i]=='tice_hist.nii.gz':
                       files.append(path)
                       I=sitk.ReadImage(path)
                       I_array=sitk.GetArrayFromImage(I)
                       z=np.shape(I_array)[0] 
                       for j in range(z):
                           write_path=os.path.join(rootdir,'image_'+str(j)+'.npy')
                           if os.path.exists(write_path) == False:
                               np.save(write_path,I_array[j,:,:])
                           slices_image.append(write_path)
                       del I_array,I
                    
                       #files_image.append(path)
                   if list[i]=='truth.nii.gz':
                       files.append(path)
                       L=sitk.ReadImage(path)
                       L_array=sitk.GetArrayFromImage(L)
                       z=np.shape(L_array)[0]
                       for j in range(z):
                           write_path=os.path.join(rootdir,'label_'+str(j)+'.npy')
                           if os.path.exists (write_path) == False:
                               np.save(write_path,L_array[j,:,:])
                           slices_label.append(write_path)
                       del L_array,L    
                   #files_label.append(path)
                   
    return files

#files_image=[]
#files_label=[]
slices_image=[]
slices_label=[]

root_dir=r'E:test'
files=list_all_files(root_dir)

train_percentage=0.9
train_image_list=slices_image[0:int(len(slices_image)*train_percentage)]
train_label_list=slices_label[0:int(len(slices_image)*train_percentage)]
validation_image_list=slices_image[int(len(slices_image)*train_percentage):int(len(slices_image)*0.95)]
validation_label_list=slices_label[int(len(slices_image)*train_percentage):int(len(slices_image)*0.95)]
test_image_list=slices_image[int(len(slices_image)*0.95):]
test_label_list=slices_label[int(len(slices_image)*0.95):]

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
