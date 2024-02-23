import SimpleITK as nib
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from nibabel.viewers import OrthoSlicer3D
import os
import cv2
import math
img_filename = 'D:\\LIDC_LDRI\\LIDC_TEST\\image\\LIDC-IDRI-0838.nii'
example_filename = 'D:\\LIDC_LDRI\\LIDC_TEST\\result1.nii'

mask = nib.load(example_filename)
img = nib.load(img_filename)
mask = np.array(mask.dataobj)
img = np.array(img.dataobj)
print(img.shape)
print(mask.shape)
#mask = np.transpose(mask, (2, 0, 1))
#OrthoSlicer3D(img.dataobj).show()
img_parent_path = 'C:\\Users\\cheny\\Desktop\\FYP\\LungSegmentation-master\\res_NSCLC_img_list_final\\'
if not os.path.exists(img_parent_path):
    os.makedirs(img_parent_path)
i = 0
pos = 238
image = img[:,:,238]
mask = mask[:,:,238]
mask = mask * 255
mask = mask.astype(np.uint8)
image = np.float32(image)
print(image.shape)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

image  = (image - np.min(image))/ (np.max(image) - np.min(image)) * 250
image = image.astype(np.uint8)
original_size = 512
for size in range(4):
    div =  math.pow(0.5, size+1)
    print(div)
    image = cv2.resize(image, (int(512*div),int(512*div)))
    mask = cv2.resize(mask, (int(512*div),int(512*div)))
    #plt.imshow(image)
    
    img_path = img_parent_path + str(size) + '_img.png'
    cv2.imwrite(img_path, image)
    #plt.savefig(img_path)
    #plt.close('all')
    #plt.imshow(mask)
    mask_path = img_parent_path + str(size) + '_mask.png'
    cv2.imwrite(mask_path, mask)
    #plt.savefig(mask_path)
    #plt.close('all') 
    
#for i in tqdm.trange(len(img[0][0])):
#    result_img = img[:,:,i] * (mask[:,:,i]*0.9+0.1)
#    plt.imshow(result_img, interpolation='none')
#    img_parent_path = 'C:\\Users\\cheny\\Desktop\\FYP\\LungSegmentation-master\\res_NSCLC3\\'
#    if not os.path.exists(img_parent_path):
#        os.makedirs(img_parent_path)
#    img_path = img_parent_path + str(i) + '.png'
#    plt.savefig(img_path)
    