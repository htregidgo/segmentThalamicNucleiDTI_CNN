# This script creates a dataset to train the CNN
# It essentially goes around the FreeSurfer subject directory of the HCP datset, and:
# 1. Reorients the T1s so the vox2ras header is approximately diagonal
# 2. Rescales the T1s so that the median of the white matter is 0.75 (and clips to [0,1])
# 3. Resamples the FA to the space of the T1
# 4. Resamples the ground truth segmentation to the space of the T1
# 5. Crops the 3 volumes around the thalami (we don't want to look too far from them)
import os
import numpy as np
import joint_diffusion_structural_seg.utils as utils
from joint_diffusion_structural_seg import models

# TODO: you don't want to create a Unet for every subject... it's be good to write a loop over subjects in the provided
# FreeSurfer directory and push them through the Unet

# TODO: also, it may be a good idea to turn this into an executable so you can process different directories with
# different models etc  all with the same command

subject = 'subject_996782'
fs_subject_dir = '/autofs/space/panamint_005/users/iglesias/data/HCPlinked/'
path_label_list = '/autofs/space/panamint_005/users/iglesias/data/joint_diffusion_structural_seg/proc_training_data_label_list.npy'
model_file = '/cluster/scratch/friday/models/diffusion_thalamus/dice_041.h5'
resolution_model_file = 0.7
output_seg_file = '/tmp/seg.mgz'
output_vol_file = '/tmp/vols.npy'

##########################################################################################

#  Unet parameters (must be the same as in training
unet_feat_count = 24
n_levels = 5
conv_size = 3
feat_multiplier = 2
nb_conv_per_level = 2
activation = 'elu'

##########################################################################################

# Area around the thalamus that we feed to CNN
W = 128

# File names
t1_file = os.path.join(fs_subject_dir, subject, 'mri', 'T1w_hires.masked.norm.mgz')
aseg_file = os.path.join(fs_subject_dir, subject, 'mri', 'aseg.mgz')
fa_file = os.path.join(fs_subject_dir, subject, 'dmri', 'dtifit.1+2+3K_FA.nii.gz')
v1_file = os.path.join(fs_subject_dir, subject, 'dmri', 'dtifit.1+2+3K_V1.nii.gz')

# Read in and reorient T1
aff_ref = np.eye(4)
t1, aff, _ = utils.load_volume(t1_file, im_only=False)
t1, aff2 = utils.align_volume_to_ref(t1, aff, aff_ref=aff_ref, return_aff=True, n_dims=3)

# If the resolution is not the one the model expected, we need to upsample!
if any(abs(np.diag(aff2)[:-1] - resolution_model_file) > 0.1):
    print('Warning: t1 does not have the resolution that the CNN expects; we need to resample')
    t1, aff2 = utils.rescale_voxel_size(t1, aff2, [resolution_model_file, resolution_model_file, resolution_model_file])

# TODO: add option to upsample T1 to resolution of model, which we will need when processing data of lower resolutio
# (e.g., GENFI). Then all other volumes will get resample to this space too when calling resample_like

# Read and resample all the other volumes
aseg, aff, _ = utils.load_volume(aseg_file, im_only=False)
aseg = utils.resample_like(t1, aff2, aseg, aff, method='nearest')
fa, aff, _ = utils.load_volume(fa_file, im_only=False)
fa = utils.resample_like(t1, aff2, fa, aff)
v1, aff, _ = utils.load_volume(v1_file, im_only=False)
# TODO: we'll want to do this in the log-tensor domain
# but for now I simply interpolate with nearest neighbors
v1_copy = v1.copy()
v1 = np.zeros([*t1.shape, 3])
v1[:, :, :, 0] = - utils.resample_like(t1, aff2, v1_copy[:, :, :, 0], aff, method='nearest') # minus as in generators.py
v1[:, :, :, 1] = utils.resample_like(t1, aff2, v1_copy[:, :, :, 1], aff, method='nearest')
v1[:, :, :, 2] = utils.resample_like(t1, aff2, v1_copy[:, :, :, 2], aff, method='nearest')

# Normalize the T1
wm_mask = (aseg == 2) | (aseg == 41)
wm_t1_median = np.median(t1[wm_mask])
t1 = t1 / wm_t1_median * 0.75
t1[t1 < 0] = 0
t1[t1 > 1] = 1

# Find the center of the thalamus and crop a volumes around it
th_mask = (aseg == 10) | (aseg == 49)
idx = np.where(th_mask)
i1 = (np.mean(idx[0]) - np.round(0.5 * W)).astype(int)
j1 = (np.mean(idx[1]) - np.round(0.5 * W)).astype(int)
k1 = (np.mean(idx[2]) - np.round(0.5 * W)).astype(int)
i2 = i1 + W
j2 = j1 + W
k2 = k1 + W

t1 = t1[i1:i2, j1:j2, k1:k2]
fa = fa[i1:i2, j1:j2, k1:k2]
v1 = v1[i1:i2, j1:j2, k1:k2, :]

# Let's be picky and preserve the RAS coordinates
aff2[:-1, -1] = aff2[:-1, -1] + np.matmul(aff2[:-1, :-1], np.array([i1, j1, k1]))

# Put toghether the 5D input
dti = np.abs(v1 * fa[..., np.newaxis])
input = np.concatenate((t1[..., np.newaxis], fa[..., np.newaxis], dti ), axis=-1)[np.newaxis,...]

# Load label list
label_list = np.load(path_label_list)

# Build Unet
unet_input_shape = [W, W, W, 5]
n_labels = len(label_list)

unet_model = models.unet(nb_features=unet_feat_count,
                         input_shape=unet_input_shape,
                         nb_levels=n_levels,
                         conv_size=conv_size,
                         nb_labels=n_labels,
                         feat_mult=feat_multiplier,
                         nb_conv_per_level=nb_conv_per_level,
                         conv_dropout=0,
                         batch_norm=-1,
                         activation=activation,
                         input_model=None)

unet_model.load_weights(model_file, by_name=True)

# Predict
posteriors = np.squeeze(unet_model.predict(input))

# TODO: it'd be good to do some postprocessing here. I would do something like:
# 1. Create a mask for the whole left thalamus, as the largest connected component of the union of left labels
# 2. Dilate the mask eg by 3 voxels.
# 3. Set to zero the probability of all left thalamic nuclei in the voxels outside the mask
# 4. Repeat 1-3 with the right thalamus
# 5. Renormalize posteriors by dividing by their sum (plus epsilon)

# TODO: a possible improvement: compute the posteriors for the left-right flipped inputs, and average them with the
# posteriors of the non-flipped volume (before or after postprocessing, I don't know ... determining which one works
# better is an empirical questions).
# This augmentation at test time has 2 benefits: 1. it fixes some errors here and there (we've seen Dice improvements of
# 1-2 points in SynthSeg or the hypothalamus); and 2. it explicitly makes the method symmetric (so you can be sure that
# any left/right differences you may find in your data are not because of a bias in the deep learning model)

# Compute volumes (skip background)
voxel_vol_mm3 = np.linalg.det(aff2)
vols_in_mm3 = np.sum(posteriors, axis=(0,1,2))[1:] * voxel_vol_mm3

# Compute segmentations
seg = label_list[np.argmax(posteriors, axis=-1)]

# Write to disk and we're done!
utils.save_volume(seg.astype(int), aff2, None, output_seg_file)
np.save(output_vol_file, vols_in_mm3)

print('freeview ' + t1_file + ' ' + fa_file + ' '  + output_seg_file)

print('All done!')




