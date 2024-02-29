from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize, Random_Crop_Z
import json
from torch.utils.data._utils.collate import default_collate

def custom_collate_fn(batch):
    ct_arrays = [item[0] for item in batch]
    bboxes = [item[1] for item in batch]
    malignancies = [item[2] for item in batch]
    
    # 对ct_arrays使用default_collate来堆叠成Tensor
    ct_arrays_collated = default_collate(ct_arrays)
    
    # 由于bboxes和malignancies可能长度不同，直接作为列表返回
    # 注意：这里我们不对bboxes和malignancies做进一步处理，如填充或堆叠，因为它们可能被用于不同的目的
    return ct_arrays_collated, bboxes, malignancies

class LungDataset(dataset):
    def __init__(self, args, ratio, trainSet = True):
        # self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))
        super().__init__()
        self.args = args
        self.anno_path = os.path.join(args.dataset_path, 'annotation.json')
        self.ct_path = os.path.join(args.dataset_path, 'ct')
        self.ratio = ratio
        self.transforms = Compose([
                Random_Crop_Z(self.args.crop_size),
                # RandomFlip_LR(prob=0.5),
                # RandomFlip_UD(prob=0.5),
                # RandomRotate()
            ])

        with open(self.anno_path, 'r') as file:
            annotation_list = json.load(file)
            if(trainSet):
                self.annotation = annotation_list[:int(len(annotation_list) * ratio)]
            else:
                self.annotation = annotation_list[int(len(annotation_list) * ratio):]

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        annotation = self.annotation[index]
        
        patient_id = annotation["patient_id"]
        
        ct = sitk.ReadImage(os.path.join(self.ct_path, patient_id + '.nii'), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        ct_array = ct_array / self.args.norm_factor
        ct_array = ct_array.astype(np.float32)
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)

        bboxes = annotation["boxes"]
        malignancy = annotation["malignancy"]
        print(malignancy)
        #print(ct_array.shape)
        #ct_array = np.transpose(ct_array, (2, 0, 1))
        #seg_array = np.transpose(seg_array, (2, 0, 1))
        #print(ct_array.shape)

        if self.transforms:
            ct_array, bboxes, malignancy = self.transforms(ct_array, bboxes, malignancy)     
        #while torch.max(seg_array_update) != 1 and torch.max(seg_array_update) == 1:
        #    ct_array_update,seg_array_update = self.transforms(ct_array, seg_array)  
        #print(ct_array_update)   
        return ct_array, bboxes, malignancy

    # def load_file_name_list(self, file_path):
    #     file_name_list = []
    #     with open(file_path, 'r') as file_to_read:
    #         while True:
    #             lines = file_to_read.readline().strip()  # 整行读取数据
    #             if not lines:
    #                 break
    #             file_name_list.append(lines.split())
    #     return file_name_list

if __name__ == "__main__":
    sys.path.append('D:\\Chenyang\\developer\\Lung_nodule')
    from config import args
    train_ds = LungDataset(args, 0.8, True)
    val_ds = LungDataset(args, 0.8, True)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 1, False, num_workers=1, collate_fn=custom_collate_fn)
    val_dl = DataLoader(val_ds, 1, False, num_workers=1, collate_fn=custom_collate_fn)

    for i, (ct, bboxes, malignancy) in enumerate(train_dl):
        print(i,",",ct.size(),",",bboxes,",", malignancy)

    for i, (ct, bboxes, malignancy) in enumerate(val_dl):
        print(i,",",ct.size(),",",bboxes,",", malignancy)