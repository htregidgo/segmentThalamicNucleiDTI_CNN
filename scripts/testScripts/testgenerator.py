import os
import numpy as np
import keras.callbacks as KC
import torch
import time
from joint_diffusion_structural_seg.generators import image_seg_generator, image_seg_generator_rgb, image_seg_generator_rgb_validation

# Path with training data
#training_dir = '/autofs/space/panamint_005/users/iglesias/data/joint_diffusion_structural_seg/proc_training_data/'
# training_dir = '/home/henry/Documents/Brain/synthDTI/4henry/data/training_withValidation/train/'
# validation_dir = '/home/henry/Documents/Brain/synthDTI/4henry/data/training_withValidation/validate/'
# training_dir = '/home/henry/Documents/Brain/synthDTI/4henry/data/training_reduced/train/'
# validation_dir = '/home/henry/Documents/Brain/synthDTI/4henry/data/training_reduced/validate/'
training_dir = '/media/henry/_localstore/Brain/synthDTI/large_download/training_reduced/train/'
validation_dir = None
# validation_dir = None
# NPY file with list of labels
# path_label_list = '/home/henry/Documents/Brain/synthDTI/4henry/data/proc_training_data_label_list.npy'
# path_label_list = '/home/henry/Documents/Brain/synthDTI/4henry/data/proc_training_data_label_list_reduced.npy'
path_label_list = '/media/henry/_localstore/Brain/synthDTI/large_download/training_reduced/proc_training_data_label_list_reduced.npy'
# Directory where model files will be written
# model_dir = '/home/henry/Documents/Brain/synthDTI/4henry/joint_diffusion_structural_seg/models/diffusion_thalamus_test_LabelLossWithWholeThaldebug/'
model_dir = '/media/henry/_localstore/Brain/synthDTI/models/diffusion_thalamus_new_year_run'
# NPY file with segmentation of channels into groups
# path_group_list = '/home/henry/Documents/Brain/synthDTI/4henry/data/proc_training_group_seg.npy'
path_group_list = '/media/henry/_localstore/Brain/synthDTI/large_download/training_reduced/proc_training_group_seg_reduced.npy'
# Batch size being volumes, it will probably be always 1...
batchsize = 1
# Size to which inputs will be cropped (use None to use whole volume)
crop_size = 128
# Maximum scaling during augmentation, in [0,1]. 0.15 is a good value
scaling_bounds = 0.15
# Maximum rotation during augmentation, in degrees. 15 is a good value
rotation_bounds = 15
# Flag whether we'll individually rotate the DTI vectors
nonlinear_rotation=True
# Maximum standard deviation of noise to add in augmentation.
# Since images are normalized, 0.1 is a good value for t1 (a bit lower for FA, since it's a fit already)
max_noise_std = 0.1
max_noise_std_fa = 0.03
# Standard deviation of log gamma, both for FA and intensity channels
gamma_std = 0.1
# Standard deviation of random contrast / brightness for intensity channels
contrast_std = 0.1
brightness_std = 0.1
# Randomize resolution during training?
randomize_resolution = True
# Fraction of DTI voxels to randomise
speckle_frac_selected=1e-4
# How to encode the segs into a onehot
# single (pick an example), combined (average), mode (majority vote), grouped (majority vote on group then average)
seg_selection='grouped'
# Will we deform the vollumes with piecewise linear displacement fields
flag_deformation = True
# Maximimum piecewise linear displacement in mm (excluding rotation + scaling)
deformation_max = 5.0
# Mode of the generator. Must be fa_v1 (linear interpolation of fa, nearest of v1) or rgb (linear on rgb)
generator_mode = 'rgb'
# generator_mode = 'fa_v1'
# Resolution of diffusion data (only needed if randomizing resolution; we use it to compute width of blurring kernels)
diffusion_resolution = 1.25
# Number of levels in Unet (5 is good)
n_levels = 5
# Number of convolution + nonlinearity blocks per level (2 is good)
nb_conv_per_level = 2
# Size of convoluton kernels (typically 3)
conv_size = 3
# Number of features per layer (eg 24)
unet_feat_count = 24
# Feature multiplier, to have more features deeper in the net. We used to do 2, more recently Benjamin started using 1
feat_multiplier = 2
# Dropout probability (between 0 and 1, we normally disable it by setting it to 0)
dropout = 0
# Type of activation / nonlinearity (elu is good)
activation = 'elu'
# Learning rate: 1e-3 is too muchn, 1e-5 is generally too little, so 1e-4 is good
lr = 1e-4
# Decay in learning rate, if you want to schedule. I normally leave it alone (ie set it to 0)
lr_decay = 0
# Number of "pretraining" epochs where we use the L2 norm on the activations rather than Dice in the softmax (5-10)
wl2_epochs = 5
# Number of epocts with Dice
dice_epochs = 200
# Steps per epoch (1000 is good)
steps_per_epoch = 1000
# Checkpoint file from which training will start (use None to start from scratch)
#checkpoint = None
checkpoint = '/home/henry/Documents/Brain/synthDTI/4henry/joint_diffusion_structural_seg/models/diffusion_thalamus_test_random_resolution3/wl2_001.h5'

# torch.set_num_threads(8)




generator = image_seg_generator_rgb(training_dir,
                                path_label_list,
                                path_group_list,
                                batchsize=batchsize,
                                scaling_bounds=scaling_bounds,
                                rotation_bounds=rotation_bounds,
                                nonlinear_rotation=nonlinear_rotation,
                                max_noise_std=max_noise_std,
                                max_noise_std_fa=max_noise_std_fa,
                                gamma_std=gamma_std,
                                contrast_std=contrast_std,
                                brightness_std=brightness_std,
                                crop_size=crop_size,
                                randomize_resolution=randomize_resolution,
                                diffusion_resolution=diffusion_resolution,
                                speckle_frac_selected=speckle_frac_selected,
                                seg_selection=seg_selection,
                                flag_deformation=flag_deformation,
                                deformation_max=deformation_max)

next(generator)
start = time.time()
next(generator)
next(generator)
next(generator)
next(generator)
next(generator)
end = time.time()
print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")
