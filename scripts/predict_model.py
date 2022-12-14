from joint_diffusion_structural_seg.training import predict




# keep in list form since you can iterate over multiple sunjects explicitely...implicit will be added soon
subject_list = ['subject_101309']
fs_subject_dir = '/home/henry/Documents/Brain/HCPDataset/HCP/'
# for now...must be
dataset = 'HCP'
path_label_list = '/home/henry/Documents/Brain/synthDTI/4henry/data/proc_training_data_label_list_reduced.npy'
model_file = '/media/henry/_localstore/Brain/synthDTI/models/diffusion_thalamus_test_reducedLabels/dice_050.h5'
# model file resolution
model_file_resolution=0.7
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


predict(subject_list,
            fs_subject_dir,
            dataset,
            path_label_list,
            model_file,
            model_file_resolution,
            generator_mode,
            unet_feat_count,
            n_levels,
            conv_size,
            feat_multiplier,
            nb_conv_per_level,
            activation,
            bounding_box_width,
            aff_ref):
