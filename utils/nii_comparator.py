import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from nibabel.viewers import OrthoSlicer3D
import os
import cv2

number_of_case = "749"
img_filename = 'D:\\LIDC_LDRI\\LIDC_TEST\\image\\LIDC-IDRI-0' + number_of_case + '.nii'
seg_filename = 'D:\\LIDC_LDRI\\LIDC_TEST\\label\\LIDC-IDRI-0' + number_of_case + '_segmentation.nii'
res_filename = 'D:\\LIDC_LDRI\\LIDC_TEST\\result\\result-0' + number_of_case + '.nii.gz'

mask = nib.load(seg_filename)
img = nib.load(img_filename)
mask_res = nib.load(res_filename)
mask = np.array(mask.dataobj)
img = np.array(img.dataobj)
mask_res = np.array(mask_res.dataobj)

print(mask.shape)
print(img.shape)
print(mask_res.shape)
#mask = np.transpose(mask, (2, 0, 1))
#OrthoSlicer3D(img.dataobj).show()
label_count = 0
true_count = 0
intersection = 0
img  = (img - np.min(img))/ (np.max(img) - np.min(img)) * 200
img = img.astype(np.uint8)
for i in tqdm.trange(len(img[0][0])):
    #i += 184
    plt.figure()
    #mask_out = cv2.resize(mask_res[:,:,i], (512, 512))
    #mask_out[mask_out < 0.5] = 0
    #mask_out[mask_out >= 0.5] = 1 
    mask_out = mask_res[:,:,i]
    mask = mask.astype(np.uint8)
    original_image = cv2.cvtColor(img[:,:,i], cv2.COLOR_GRAY2BGR)
    original_mask = cv2.cvtColor(mask[:,:,i], cv2.COLOR_GRAY2BGR)
    original_mask = original_mask.astype(np.uint8)
    #print(original_mask.shape)
    original_mask = cv2.cvtColor(original_mask, cv2.COLOR_BGR2GRAY)
    #print(original_image.shape)
    #print(original_mask.shape)
    #plt.imshow(original_image)
    #plt.show()
    #target_img = img[:,:,i] * (mask[:,:,i]*0.95+0.05)
    #result_img = img[:,:,i] * (mask_out*0.95+0.05)
    prediction_mask = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)
    prediction_mask = cv2.cvtColor(prediction_mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(original_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    target_img = original_image.copy()
    cv2.drawContours(target_img, contours, -1, (255,0,0), 1)

    contours2, hierarchy = cv2.findContours(prediction_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    result_img = original_image.copy()
    cv2.drawContours(result_img, contours2, -1, (255,0,0), 1)
    plt.figure(dpi=128,figsize=(6,6))
    for index in range(1,5):
        plt.subplot(2,2,index)
        if(index == 1):
            plt.title("(a)")
            plt.imshow(original_image, interpolation='none')
        if(index == 2):
            plt.title("(b)")
            plt.imshow(original_mask, interpolation='none')
        if(index == 3):
            plt.title("(c)")
            plt.imshow(target_img, interpolation='none')
        if(index == 4):
            plt.title("(d)")
            plt.imshow(result_img, interpolation='none')
        plt.xticks([])
        plt.yticks([])
    img_parent_path = 'D:\\LIDC_LDRI\\LIDC_TEST\\vis\\' + number_of_case + '\\'
    if not os.path.exists(img_parent_path):
        os.makedirs(img_parent_path)
    img_path = img_parent_path + str(i) + '.png'
    plt.savefig(img_path)
    plt.close('all')