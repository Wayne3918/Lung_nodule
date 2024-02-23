import os

dataset_out_path = 'D:\\LIDC_LDRI\\OUT_LIDC_300_nodule_adjusted\\'

train_path = os.path.join(dataset_out_path, 'train')
val_path = os.path.join(dataset_out_path, 'val')
train_path_ct = os.path.join(train_path, 'ct')
val_path_ct = os.path.join(val_path, 'ct')
train_path_label = os.path.join(train_path, 'label')
val_path_label = os.path.join(val_path, 'label')

train_txt_path = os.path.join(dataset_out_path, 'train_path_list.txt')
val_txt_path = os.path.join(dataset_out_path, 'val_path_list.txt')

for ct_img in os.listdir(train_path_ct):
    ct_path_name = os.path.join(train_path_ct,ct_img)
    label_name = ct_img.split('.')[0] + '_segmentation.nii'
    ct_label_name = os.path.join(train_path_label,label_name)
    data_written = ct_path_name + ' ' + ct_label_name + '\n'
    with open(train_txt_path,"a") as f:
        f.writelines(data_written)

for ct_img in os.listdir(val_path_ct):
    ct_path_name = os.path.join(val_path_ct,ct_img)
    label_name = ct_img.split('.')[0] + '_segmentation.nii'
    ct_label_name = os.path.join(val_path_label,label_name)
    data_written = ct_path_name + ' ' + ct_label_name + '\n'
    with open(val_txt_path,"a") as f:
        f.writelines(data_written)
