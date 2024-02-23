import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from os.path import join
import config
import pylidc as pl
from skimage.measure import marching_cubes, mesh_surface_area
import matplotlib.pyplot as plt
from pylidc.utils import consensus
import tqdm
import nibabel as nib
from skimage import exposure, util
import imageio as io
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, median_filter
import cv2
'''

path = '.\LIDC\LIDC-IDRI\LIDC-IDRI-0001\01-01-2000-30178\3000566-03192\'
reader = sitk.ImageSeriesReader()
dicom = reader.GetGDCMSeriesFileNames(path)
# print(dicom)
reader.SetFileNames(dicom)
image = reader.Execute()
img_array = sitk.GetArrayFromImage(image)

'''
'''
save_dir = 'D:\\LIDC LDRI\\LIDC_Augmented'

scan = pl.query(pl.Annotation).filter(pl.Annotation.spiculation == 5,
                                      pl.Annotation.malignancy == 5)
print(anns.count())
ann = anns[0]
#for ann in anns:
print(ann.scan.patient_id)


mask = ann.boolean_mask(pad=[(1,1), (1,1), (1,1)]) 

rij  = ann.scan.pixel_spacing
rk   = ann.scan.slice_thickness
verts, faces, _, _= marching_cubes(mask.astype(float), 0.5,
                                        spacing=(rij, rij, rk),
                                        step_size=1)

print(verts)
print(faces)
'''
'''
# View all the resampled image volume slices.
for i in range(n+1):
    img.set_data(vol[:,:,i] * (mask[:,:,i]*0.6+0.2))

    plt.title("%02d / %02d" % (i+1, n))
    plt.pause(0.1)
'''

default_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
dataset_out_path = 'D:\LIDC_LDRI\OUT_LIDC_200_full_horizontal_slice_4_lastfiltered'

if not os.path.exists(dataset_out_path):    # 创建保存目录
    os.makedirs(dataset_out_path)
    os.makedirs(join(dataset_out_path,'ct'))
    os.makedirs(join(dataset_out_path, 'label'))

scans = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 20,
                                 pl.Scan.pixel_spacing <= 5)
count = 0
res_list = [0,0,0,0,0]

for scan in scans:
    try:
        patient_id = scan.patient_id
        nods = scan.cluster_annotations()

        rij   = scan.pixel_spacing
        rk    = scan.slice_spacing

        vol = scan.to_volume()

        #initialise image and mask maps
        img = scan.load_all_dicom_images(verbose=True)
        
        img_array = np.zeros((512,512,len(img)))
        print(img_array.shape)
        for i in range(len(img)):
            img_array[:,:,i] = img[i].pixel_array

        mask = np.zeros((512,512,len(img)))
        for i in range(len(img)):
            mask[:,:,i] = False

        #Image pre-processing
        img_array[img_array > 600] = 600
        img_array[img_array < -1000] = -1000


        #for z in range(len(img)):
        #    plt.imshow(img_array[:,:,z], interpolation='none')
        #    plt.show()
        #    plt.waitforbuttonpress()
        # img_array = 255 * (img_array + 1000) / 1600
        img_array.astype(int)
        #plt.imshow(img_array[:,:,100], interpolation='none')
        #plt.show()
        count += 1
        if count <= 600:
            continue
        elif count <= 800:
        #print("%s has %d nodules." % (scan, len(nods)))
        #Add annotations to the mask
            for ann_group in nods:
                if(len(ann_group) < 4):
                    continue

                avg_diam = 0
                for index in range(len(ann_group)):
                    avg_diam += float(ann_group[index].diameter)
                    #print(avg_diam)
                avg_diam = avg_diam / len(ann_group)
                print(avg_diam)
                if(avg_diam < 3):
                    continue 
                
                #if(avg_subtlety < 3):
                #    continue

                #print(avg_subtlety) 
                #res_list[avg_subtlety-1] += 1
                #print(res_list)
                cmask,cbbox,_ = consensus(ann_group, clevel=0.5)
                imin = cbbox[0].start
                imax = cbbox[0].stop
                jmin = cbbox[1].start
                jmax = cbbox[1].stop
                kmin = cbbox[2].start
                kmax = cbbox[2].stop
                #print(imin, imax, jmin, jmax, kmin, kmax)
                for x in range(imin, imax):
                    for y in range(jmin, jmax):
                        for z in range(kmin, kmax):
                            mask[x, y, z] = cmask[x-imin, y-jmin, z-kmin]
            #for i in tqdm.trange(len(img)):
            #    result_img = img_array[:,:,i] * (mask[:,:,i]*0.6+0.2)
            #    plt.imshow(result_img, interpolation='none')
            #    img_parent_path = 'D:\\LIDC LDRI\\LIDC_Demo\\'
            #    img_path = img_parent_path + str(i) + '.png'
            #    plt.savefig(img_path)#保存图片
            depth = img_array.shape[2]
            zoomed_images = np.zeros((256,256,depth))
            zoomed_masks = np.zeros((256,256,depth))
            
            for pos in range(depth):
                img = cv2.resize(img_array[:,:,pos], (256, 256))
                mask_out = cv2.resize(mask[:,:,pos], (256, 256))
                mask_out[mask_out < 0.5] = 0
                mask_out[mask_out >= 0.5] = 1 
                zoomed_images[:,:,pos] = img
                zoomed_masks[:,:,pos] = mask_out

            #zoomed_img = ndimage.zoom(img_array, (0.5, 0.5, 0.5*rk))
            #zoomed_img = exposure.equalize_hist(zoomed_img)
            #zoomed_img = median_filter(zoomed_img, size=3)
            #zoomed_mask = ndimage.zoom(mask, (0.5, 0.5, 0.5*rk), order = 2)
            #zoomed_mask[zoomed_mask < 0.5] = 0
            #zoomed_mask[zoomed_mask >= 0.5] = 1
            zoomed_images = np.transpose(zoomed_images, (2, 0, 1))
            zoomed_masks = np.transpose(zoomed_masks, (2, 0, 1))                
            img_out = sitk.GetImageFromArray(zoomed_images)
            img_out.SetDirection(default_direction)
            #img_out.SetOrigin(ct.GetOrigin())
            img_out.SetSpacing((2 * rij, 2*rij , rk))
            print(zoomed_images.shape)

            #img_out = sitk.GetImageFromArray(zoomed_img)
            #img_out.SetDirection(default_direction)
            #img_out.SetOrigin(ct.GetOrigin())
            #img_out.SetSpacing((2 * rij, 2*rij , 2))

            mask_out = sitk.GetImageFromArray(zoomed_masks)
            mask_out.SetDirection(default_direction)
            mask_out.SetSpacing((2 * rij, 2*rij , rk))
            print(zoomed_masks.shape)
            img_out_path = os.path.join(dataset_out_path, 'ct', patient_id + '.nii')
            mask_out_path = os.path.join(dataset_out_path, 'label', patient_id + '_segmentation.nii')
            sitk.WriteImage(img_out, img_out_path)  
            sitk.WriteImage(mask_out, mask_out_path)
            txt_path = os.path.join(dataset_out_path, 'train_path_list.txt')
            data_written = img_out_path + ' ' + mask_out_path + '\n'
            with open(txt_path,"a") as f:
                f.writelines(data_written)
            print(patient_id)
        else:
            break
    except:
        pass
#for i in tqdm.trange(len(zoomed_img[0][0])):
#    result_img = zoomed_img[:,:,i] * (zoomed_mask[:,:,i]*0.6+0.2)
#    plt.imshow(result_img, interpolation='none')
#    img_parent_path = 'D:\\LIDC LDRI\\LIDC_Demo\\'
#    img_path = img_parent_path + str(i) + '.png'
#    plt.savefig(img_path)#保存图片
'''
n = 140
vol,mask = ann.uniform_cubic_resample(n)
print(vol)
print(mask)
rij  = ann.scan.pixel_spacing
rk   = ann.scan.slice_thickness
print(rij, rk)
# Setup the plot.
img = plt.imshow(np.zeros((n+1, n+1)), 
                    vmin=vol.min(), vmax=vol.max(),
                    cmap=plt.cm.gray)

'''
#print(len(cmask), len(cmask[0]), len(cmask[0][0])) -> 14 14 8
#print(cbbox) -> (slice(278, 292, None), slice(67, 81, None), slice(72, 80, None))
#print(len(masks), len(masks[0]), len(masks[0][0]), len(masks[0][0][0]))# -> 4 14 14 8
