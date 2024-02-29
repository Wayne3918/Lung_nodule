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
import json
import matplotlib.patches as patches

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
dataset_out_path = './LIDC_normalised_detection_tmp'

if not os.path.exists(dataset_out_path):    # 创建保存目录
    os.makedirs(dataset_out_path)
    os.makedirs(join(dataset_out_path,'ct'))

scans = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 20,
                                 pl.Scan.pixel_spacing <= 5)
count = 0
res_list = [0,0,0,0,0]

def visualize_bboxes_2d_slices(image_array, bboxes):
    """
    Visualize bounding boxes on 2D slices of a 3D image array.

    Parameters:
    - image_array: A numpy array of shape [x, 256, 256] representing the 3D image.
    - bboxes: A list of bounding boxes, each defined as [xmin, xmax, ymin, ymax, zmin, zmax].
    """
    # Determine the number of slices
    num_slices = image_array.shape[0]

    # Loop through each slice
    for z in range(num_slices):
        fig, ax = plt.subplots()
        ax.imshow(image_array[z], cmap='gray')

        # Check each bbox to see if it intersects with this z slice
        for bbox in bboxes:
            zmin, zmax, xmin, xmax, ymin, ymax = bbox
            # print(bbox)
            if zmin <= z <= zmax:
                # Calculate the rectangle dimensions
                rect = patches.Rectangle((ymin, xmin), ymax-ymin, xmax-xmin, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        ax.set_title(f'Slice {z}')
        plt.show()

def compute_labels(anno_group):

    result_dict = {} # [subtlety(avg), internalStructure(mode), 
        #calcification(mode), sphericity(avg),
        #margin(avg), lobulation(avg), spiculation(avg), 
        #texture(avg), malignancy(avg)]
    avg_subtlety = 0
    internalStructure_list = []
    calcification_list = []
    avg_sphericity = 0
    avg_margin = 0
    avg_lobulation = 0
    avg_spiculation = 0
    avg_texture = 0
    avg_malignancy = 0
    for index in range(len(ann_group)):
        avg_subtlety += float(ann_group[index].subtlety)
        internalStructure_list.append(ann_group[index].internalStructure)
        calcification_list.append(ann_group[index].calcification)
        avg_sphericity += float(ann_group[index].sphericity)
        avg_margin += float(ann_group[index].margin)
        avg_lobulation += float(ann_group[index].lobulation)
        avg_spiculation += float(ann_group[index].spiculation)
        avg_texture += float(ann_group[index].texture)
        avg_malignancy += float(ann_group[index].malignancy)
    avg_subtlety = avg_subtlety / len(ann_group) - 0.01
    #print(avg_subtlety) 
    result_dict['subtlety'] = round(avg_subtlety / len(ann_group))
    result_dict['internalStructure'] = max(internalStructure_list, key=internalStructure_list.count)
    result_dict['calcification'] = max(calcification_list, key=internalStructure_list.count)
    result_dict['sphericity'] = round(avg_sphericity / len(ann_group))
    result_dict['margin'] = round(avg_margin / len(ann_group))
    result_dict['lobulation'] = round(avg_lobulation / len(ann_group))
    result_dict['spiculation'] = round(avg_spiculation / len(ann_group))
    result_dict['texture'] = round(avg_texture / len(ann_group))
    result_dict['malignancy'] = round(avg_malignancy / len(ann_group))
    return result_dict


formated_data = []
for scan in tqdm.tqdm(scans):
    # try:
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
    # if count <= 201:
    #     continue
    if count <= 5000:
    #print("%s has %d nodules." % (scan, len(nods)))
    #Add annotations to the mask
        anno_dict_list = []
        bbox_list = []
        for ann_group in nods:
            anno_dict = compute_labels(ann_group)
            # print(anno_dict)
            anno_dict_list.append(anno_dict)
            # avg_subtlety = 0
            # for index in range(len(ann_group)):
            #     avg_subtlety += float(ann_group[index].subtlety)
            #     print(float(ann_group[index].malignancy))
            # avg_subtlety = avg_subtlety / len(ann_group) - 0.01
            # #print(avg_subtlety) 
            # avg_subtlety = round(avg_subtlety)
            # if(avg_subtlety < 3):
            #     continue
            # print('aaa')

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

            bbox = [int(0.5*rk*kmin), int(0.5*rk*kmax),int(0.5*imin), int(0.5*imax), int(0.5*jmin), int(0.5*jmax)]
            
            print(bbox)
            bbox_list.append(bbox)
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

        zoomed_img = ndimage.zoom(img_array, (0.5, 0.5, 0.5*rk))
        #zoomed_img = exposure.equalize_hist(zoomed_img)
        #zoomed_img = median_filter(zoomed_img, size=3)
        
        # zoomed_mask = ndimage.zoom(mask, (0.5, 0.5, 0.5*rk), order = 2)
        # zoomed_mask[zoomed_mask < 0.5] = 0
        # zoomed_mask[zoomed_mask >= 0.5] = 1
        zoomed_img = np.transpose(zoomed_img, (2, 0, 1))
        # zoomed_mask = np.transpose(zoomed_mask, (2, 0, 1))                
        img_out = sitk.GetImageFromArray(zoomed_img)
        img_out.SetDirection(default_direction)
        #img_out.SetOrigin(ct.GetOrigin())
        img_out.SetSpacing((2 * rij, 2*rij , 2))
        print(zoomed_img.shape)
        # visualize_bboxes_2d_slices(zoomed_img, bbox_list)
        #處理Annotation
        # {
        #     "image_id": "image1.jpg",
        #     "boxes": [[10, 20, 110, 120], [30, 40, 130, 140]],
        #     "categories": [1, 2],
        #     "colors": [2, 3],
        #     "shapes": [3, 1]
        # }
        # colors = [item['color'] for item in images_info]
        predict_dict = {
            "patient_id": patient_id,
            "boxes": bbox_list,
            "subtlety" : [item['subtlety'] for item in anno_dict_list],
            "internalStructure" : [item['internalStructure'] for item in anno_dict_list],
            "calcification" : [item['calcification'] for item in anno_dict_list],
            "sphericity" : [item['sphericity'] for item in anno_dict_list],
            "margin" : [item['margin'] for item in anno_dict_list],
            "lobulation" : [item['lobulation'] for item in anno_dict_list],
            "spiculation" : [item['spiculation'] for item in anno_dict_list],
            "texture" : [item['texture'] for item in anno_dict_list],
            "malignancy" : [item['malignancy'] for item in anno_dict_list],
        }
        print(predict_dict)
        
        formated_data.append(predict_dict)
        #img_out = sitk.GetImageFromArray(zoomed_img)
        #img_out.SetDirection(default_direction)
        #img_out.SetOrigin(ct.GetOrigin())
        #img_out.SetSpacing((2 * rij, 2*rij , 2))

        # # mask_out = sitk.GetImageFromArray(zoomed_mask)
        # mask_out.SetDirection(default_direction)
        # mask_out.SetSpacing((2 * rij, 2*rij , 2))
        # print(zoomed_mask.shape)
        img_out_path = os.path.join(dataset_out_path, 'ct', patient_id + '.nii')
        # mask_out_path = os.path.join(dataset_out_path, 'label', patient_id + '_segmentation.nii')
        sitk.WriteImage(img_out, img_out_path)  
        # sitk.WriteImage(mask_out, mask_out_path)
        txt_path = os.path.join(dataset_out_path, 'train_path_list.txt')
        data_written = img_out_path + '\n'
        with open(txt_path,"a") as f:
            f.writelines(data_written)
        print(patient_id)
    else:
        break
with open(os.path.join(dataset_out_path, 'annotation.json'), 'w') as file:
    # 将字典写入文件，使用indent参数美化输出
    json.dump(formated_data, file, indent=4)
    # except:
    #     pass

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
