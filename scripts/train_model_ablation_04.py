import os

from joint_diffusion_structural_seg.ablation_training import ablation_training_function

## machine specific directories
top_level_data_dir = '../data'
top_level_model_dir = '../models'

## Run specific parameters that will change for ablations
# Name of the model - link to ablation spreadsheet
model_name = 'joint_thalamus_ablation_04'
# Fraction of DTI voxels to randomised. Between 0 and 1. Set to 0 to turn off speckle. 1 in 10k sounds right
speckle_frac_selected=1e-4
# Flag whether we'll individually rotate the DTI vectors
nonlinear_rotation=True
# Will we deform the images with piecewise linear displacement fields (this includes re-orientation)
flag_deformation = True
# Maximum piecewise linear displacement in mm (excluding rotation + scaling)
deformation_max = 5.0
# How to encode the segmentations into a onehot
# single (pick an example), combined (average), mode (majority vote), grouped (majority vote on group then average)
seg_selection='combined'
# Dice version - "individual" do standard label-wise Dice, "grouped" also add contribution of groups and whole thalamus
dice_version="grouped"
# directory and file locations based on if we're using the full labels or reduced set
top_level_training_dir = os.path.join(top_level_data_dir,'training_reduced')
label_list_name = 'proc_training_data_label_list_reduced.npy'
group_list_name = 'proc_training_group_seg_reduced.npy'


ablation_training_function(top_level_training_dir, label_list_name, group_list_name,
                               top_level_model_dir, model_name, nonlinear_rotation,
                               speckle_frac_selected, seg_selection, flag_deformation,
                               deformation_max, dice_version)


