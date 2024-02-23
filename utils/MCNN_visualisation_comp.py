import os
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import math
MCNN_res_path = 'D:\\LIDC_LDRI\\MCNN_result'
UNet_res_path = 'D:\\LIDC_LDRI\\UNet_result'
img_filename = 'D:\\LIDC_LDRI\\OUT_LIDC_300_nodule_small\\val\\ct\\LIDC-IDRI-0315_4.nii'
seg_filename = 'D:\\LIDC_LDRI\\OUT_LIDC_300_nodule_small\\val\\label\\LIDC-IDRI-0315_4_segmentation.nii'
out_path = 'D:\\LIDC_LDRI\\Res_comp_0011_gg'
if not os.path.exists(out_path):
    os.makedirs(out_path)

mask = nib.load(seg_filename)
img = nib.load(img_filename)
mask = np.array(mask.dataobj)
img = np.array(img.dataobj)
MCNN_set = []
UNet_set = []
pos_index = 2
for index in range(5):
    file_index = pow(2,index)
    file_name = str(file_index) + '-result-0315_4.nii.gz'
    MCNN_file_path = os.path.join(MCNN_res_path, file_name) 
    UNet_file_path = os.path.join(UNet_res_path, file_name) 
    MCNN_tmp = nib.load(MCNN_file_path)
    UNet_tmp = nib.load(UNet_file_path)
    MCNN_tmp = np.array(MCNN_tmp.dataobj)
    UNet_tmp = np.array(UNet_tmp.dataobj)
    MCNN_set.append(MCNN_tmp)
    UNet_set.append(UNet_tmp)
for index in range(1,13):
    plt.subplot(2,6,index)
    if index == 1:
        plt.title("Input")
        plt.imshow(img[:,:,24], interpolation='none')
    elif (index > 1 and index < 7):
        plt.title("MCNN I/" + str(pow(2, 4-(index-2))))
        plt.imshow(MCNN_set[index-2][:,:,pos_index], interpolation='none')
        print(MCNN_set[index-2][:,:,pos_index])
        pos_index = 2 * pos_index            
    elif index == 7:
        plt.title("Annotation")
        plt.imshow(mask[:,:,24], interpolation='none')
        pos_index = 2
    elif index > 7:
        plt.title("UNet I/" + str(pow(2, 4-(index-8))))
        plt.imshow(UNet_set[index-8][:,:,pos_index], interpolation='none')
        print(UNet_set[index-8][:,:,pos_index])
        pos_index = 2 * pos_index
    img_path = os.path.join(out_path, '0315_4.png')
    plt.savefig(img_path)  