# -*- coding: utf-8 -*-



import numpy as np
from os import path
import SimpleITK as sitk

img_path=r'E:\preprocessed\HGG\Brats18_2013_2_1\t1ce.nii'
label_path=r'E:\preprocessed\HGG\Brats18_2013_2_1\truth.nii'

"""
PixelType = itk.ctype('unsigned char')
ImageType = itk.Image[PixelType, 3]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(img_path)

I=reader.GetOutput()
J=I.GetPixel
"""
image=sitk.ReadImage(img_path)
label=sitk.ReadImage(label_path)
I_array=sitk.GetArrayFromImage(image)
L_array=sitk.GetArrayFromImage(label)
L_array[L_array!=1]=0

output_image_path=r'E:\WashU-Lab\2ndRotation\x_train.npy'
output_label_path=r'E:\WashU-Lab\2ndRotation\y_train.npy'
np.save(output_image_path,I_array)
np.save(output_label_path,L_array)
