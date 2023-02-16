import numpy as np
import os,sys
import torch
sys.path.append("..")

from joint_diffusion_structural_seg.predict import predict
from joint_diffusion_structural_seg.metrics import DiceLossLabels
from joint_diffusion_structural_seg import utils
from joint_diffusion_structural_seg.generators import encode_onehot
from sklearn import preprocessing

def dice(pred,gt,level=0):
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    pred_flat[pred_flat != level] = 0
    pred_flat[pred_flat == level] = 1
    gt_flat[gt_flat != level] = 0
    gt_flat[gt_flat == level] =	1
    
    inter = np.sum(gt_flat * pred_flat)
    epsilon = 1e-6
    return (2. * inter + epsilon)/(np.sum(gt_flat) + np.sum(pred_flat) + epsilon)


def label_dice(pred,gt,label_list):
    label_dice_list = []
    for label in label_list:
        label_dice_list.append(dice(pred,gt,level=label)) 
    return np.array(label_dice_list)


def group_dice(pred,gt,label_list,group_list):
    group_dice_list = []
    for group in list(set(group_list)):
        gt_copy = np.zeros_like(gt)
        pred_copy = np.zeros_like(pred)
        label_indices = [i for i,x in enumerate(group_list) if x==group]
        raw_label_indices = [label_list[i] for i in label_indices]
        for label in raw_label_indices:
            gt_copy[gt == label] = 1
            pred_copy[pred == label] = 1    
        group_dice_list.append(dice(pred_copy,gt_copy,level=1))
    return group_dice_list


def thalamus_dice(pred,gt):
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    pred_flat[pred_flat != 0] = 1
    gt_flat[gt_flat != 0] = 1
    inter = np.sum(gt_flat * pred_flat)
    epsilon = 1e-6
    return (2. * inter + epsilon)/(np.sum(gt_flat) + np.sum(pred_flat) + epsilon)

def onehot(vol,label_list):
  num_classes = len(label_list)
  return np.squeeze(np.eye(num_classes)[vol.reshape(-1)])


subject_list = ['subject_141119']
fs_subject_dir = '/autofs/space/nicc_003/users/olchanyi/scripts/tmp/files4mark/data/training_reduced/validate/'
# for now...must be
dataset = 'validate'
path_label_list = '/autofs/space/nicc_003/users/olchanyi/scripts/tmp/files4mark/data/proc_training_data_label_list_reduced.npy'
path_group_list = '/autofs/space/nicc_003/users/olchanyi/scripts/tmp/files4mark/data/proc_training_group_seg_reduced.npy'
model_file = '/autofs/space/nicc_003/users/olchanyi/scripts/tmp/files4mark/models/dice_200.h5'
# model file resolution
resolution_model_file=0.7
# generator mode for prediction data (make sure same as training!)
generator_mode='rgb'
# U-net: number of features in base level (make sure same as training!)
unet_feat_count=24
# U-net: number of levels (make sure same as training!)
n_levels = 5
# U-net: convolution kernel size (make sure same as training!)
conv_size = 3
# U-net: number of features per level multiplier (make sure same as training!)
feat_multiplier = 2
# U-net: number of convolutions per level (make sure same as training!)
nb_conv_per_level = 2
# U-net: activation function (make sure same as training!)
activation='elu'
# (isotropic) dimensions of bounding box to take around thalamus
bounding_box_width = 128
# reference affine
aff_ref = np.eye(4)


## load params including label and group lists
shell_mag = ['_1k','_2k']
seg_list = ['_1k_DSWbeta','_1k_LogGauss','_1k_Wishart','_2k_DSWbeta','_2k_LogGauss','_2k_Wishart']

label_list = np.sort(np.load(path_label_list)).astype(int)
#mapping = np.zeros(1 + label_list[-1], dtype='int')
#mapping[label_list] = np.arange(len(label_list))

group_list = np.load(path_group_list)
#grp_list = np.load(path_group_list)
#grp_mat = torch.zeros(grp_list.shape[0], grp_list.max() + 1, dtype=torch.float64)
#for il in range(0, grp_list.shape[0]):
#    grp_mat[il, grp_list[il]] = 1


for subject in subject_list:
    for shell in shell_mag:
        unet_seg_file = os.path.join(fs_subject_dir)

        predict([subject + shell],
                    fs_subject_dir,
                    dataset,
                    path_label_list,
                    model_file,
                    resolution_model_file,
                    generator_mode,
                    unet_feat_count,
                    n_levels,
                    conv_size,
                    feat_multiplier,
                    nb_conv_per_level,
                    activation,
                    bounding_box_width,
                    aff_ref,
                    shell_flag=shell)

        unet_seg_path = os.path.join(fs_subject_dir,subject + shell,'results','thalNet_reduced_randV1_e050.seg.mgz')
        t1_path = os.path.join(fs_subject_dir,subject + shell, subject + '.t1.nii.gz')
        unet_seg, aff_unet, hd_unet = utils.load_volume(unet_seg_path,im_only=False)

       
  
        for seg_name in seg_list:
            seg_path = os.path.join(fs_subject_dir,subject + shell,'segs',subject + seg_name + ".nii.gz")
            seg, aff_seg, hd_seg = utils.load_volume(seg_path,im_only=False)
            seg_reshaped = utils.resample_like(unet_seg, aff_unet, seg, aff_seg, method='nearest')

            #seg_reshaped = torch.tensor(seg_reshaped.astype(int), device='cpu').long()
            #unet_seg = torch.tensor(unet_seg.astype(int), device='cpu').long()
            #mapping = torch.tensor(mapping.astype(int), device='cpu').long()
            #unet_seg_encoded_labels = onehot(unet_seg_transformed_labels.astype(int),label_list)
            #unet_seg_encoded_grouped = encode_onehot(mapping, unet_seg, label_list, 'grouped', grp_mat)
            #seg_encoded_labels = onehot(seg_transformed_labels.astype(int), label_list)
            #seg_encoded_grouped = encode_onehot(mapping, seg_reshaped, label_list, 'grouped', grp_mat)
            
            #dl = DiceLossLabels()
            #tf_label_dice = dl.loss(unet_seg_encoded_labels,seg_encoded_labels)

            label_dice_array = label_dice(unet_seg,seg_reshaped,label_list)
            group_dice_array = group_dice(unet_seg,seg_reshaped,label_list,group_list)
            thal_dice = thalamus_dice(unet_seg,seg_reshaped)
            #print("label dice for " + subject + shell + " with " + seg_name +  " is: ", label_dice_array)
            #print("group dice for " + subject + shell + " with " + seg_name +  " is: ", group_dice_array)
            #print("thalamus dice for " + subject + shell + " with " + seg_name +  " is: ", thal_dice)
            
            avg_loss = 1.0 - (1/3)*(np.mean(label_dice_array) + np.mean(group_dice_array) + thal_dice)
            print("Average loss for " + subject + shell + " with " + seg_name + "  is: ", avg_loss, "\n")
