import time

from tmp.rotationTest import image_seg_generator_rgb_rotationtest

# Path with training data
#training_dir = '/autofs/space/panamint_005/users/iglesias/data/joint_diffusion_structural_seg/proc_training_data/'
# training_dir = '/home/henry/Documents/Brain/synthDTI/4henry/data/training_withValidation/train/'
# validation_dir = '/home/henry/Documents/Brain/synthDTI/4henry/data/training_withValidation/validate/'
training_dir = '/home/henry/Documents/Brain/synthDTI/4henry/data/training_reduced/train/'
validation_dir = '/home/henry/Documents/Brain/synthDTI/4henry/data/training_reduced/validate/'
# validation_dir = None
# NPY file with list of labels
# path_label_list = '/home/henry/Documents/Brain/synthDTI/4henry/data/proc_training_data_label_list.npy'
path_label_list = '/home/henry/Documents/Brain/synthDTI/4henry/data/proc_training_data_label_list_reduced.npy'
# Directory where model files will be writte
model_dir = '/home/henry/Documents/Brain/synthDTI/4henry/joint_diffusion_structural_seg/models/diffusion_thalamus_test_pytorchmixed/'
# Batch size being volumes, it will probably be always 1...
batchsize = 1
# Size to which inputs will be cropped (use None to use whole volume)
crop_size = 128
# Maximum scaling during augmentation, in [0,1]. 0.15 is a good value
scaling_bounds = 0.15
# Maximum rotation during augmentation, in degrees. 15 is a good value
rotation_bounds = 90
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




generator = image_seg_generator_rgb_rotationtest(training_dir,
                                path_label_list,
                                batchsize=batchsize,
                                scaling_bounds=scaling_bounds,
                                rotation_bounds=rotation_bounds,
                                max_noise_std=max_noise_std,
                                max_noise_std_fa=max_noise_std_fa,
                                gamma_std=gamma_std,
                                contrast_std=contrast_std,
                                brightness_std=brightness_std,
                                crop_size=crop_size,
                                randomize_resolution=randomize_resolution,
                                diffusion_resolution=diffusion_resolution)

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
