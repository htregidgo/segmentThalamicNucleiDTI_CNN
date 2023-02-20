import glob

import numpy as np
import os,argparse


from joint_diffusion_structural_seg.validate import validate_dti_segs


def main():

    ## parse arguments
    parser = argparse.ArgumentParser(description="Script to validate a version of the joint structural + DTI "
                                                 "thalamus segmentor from a list of ablations", epilog='\n',
                                     add_help=True)

    parser.add_argument("--top_dir", help="top level directory containing both a data and model directory",
                        default="../")

    parser.add_argument("--N_subjects", help="number of subjects to use. Defaults to 0 which will cause the "
                                             "whole dataset to be used.",
                        default=0,type=int)

    args = parser.parse_args()

    validate_ablation_02(top_level_dir=args.top_dir, N_subjects=args.N_subjects)


def validate_ablation_02( top_level_dir, N_subjects ) :

    ## machine specific directories
    top_level_data_dir = os.path.join(top_level_dir,'data')
    top_level_model_dir = os.path.join(top_level_dir,'models')

    assert os.path.exists(top_level_data_dir), \
        "top_dir must contain data directory"

    ##########################################################
    ## Run specific parameters that will change for ablations
    ##########################################################
    # Name of the model - link to ablation spreadsheet
    model_name = 'joint_thalamus_ablation_02'
    # How to encode the segmentations into a onehot
    # single (pick an example), combined (average), mode (majority vote), grouped (majority vote on group then average)
    seg_selection='grouped'
    # Dice version - "individual" do standard label-wise Dice, "grouped" also add contribution of groups and whole thalamus
    dice_version="grouped"
    # directory and file locations based on if we're using the full labels or reduced set
    top_level_training_dir = os.path.join(top_level_data_dir, 'training_full')
    label_list_name = 'proc_training_data_label_list.npy'
    group_list_name = 'proc_training_group_seg.npy'

    ##########################################################
    ## set U-net parameters
    ##########################################################
    # U-net: number of features in base level (make sure same as training!)
    unet_feat_count = 24
    # U-net: number of levels (make sure same as training!)
    n_levels = 5
    # U-net: convolution kernel size (make sure same as training!)
    conv_size = 3
    # U-net: number of features per level multiplier (make sure same as training!)
    feat_multiplier = 2
    # U-net: number of convolutions per level (make sure same as training!)
    nb_conv_per_level = 2
    # U-net: activation function (make sure same as training!)
    activation = 'elu'
    # (isotropic) dimensions of bounding box to take around thalamus
    bounding_box_width = 128

    ##########################################################
    ## get relative filepaths etc.
    ##########################################################

    fs_subject_dir = os.path.join(top_level_training_dir, 'validate/')
    path_label_list = os.path.join(top_level_training_dir, label_list_name)
    path_group_list = os.path.join(top_level_training_dir, group_list_name)

    subject_list = sorted(glob.glob(fs_subject_dir + '/subject*k'))

    if N_subjects<=0:
        N_subjects=len(subject_list)

    subject_list = subject_list[0:N_subjects]

    ablation_model_Dir = os.path.join(top_level_model_dir,model_name)
    model_list = sorted(glob.glob(ablation_model_Dir + '/dice*h5'))
    validation_out_file = os.path.join(ablation_model_Dir,'model_validation.csv')

    ablation_loss_vector = np.zeros((len(model_list),1))

    model_no = 0

    for model_file in model_list:

        model_fn = os.path.split(model_file)[1]
        print('validating model ',model_fn)

        model_loss, _ = validate_dti_segs(subject_list,
                          path_label_list,
                          path_group_list,
                          model_file,
                          unet_feat_count,
                          n_levels,
                          conv_size,
                          feat_multiplier,
                          nb_conv_per_level,
                          activation,
                          bounding_box_width,
                          seg_selection=seg_selection,
                          dice_version=dice_version)

        ablation_loss_vector[model_no] = model_loss

        model_no += 1

    np.savetxt(validation_out_file,ablation_loss_vector,delimiter=",")


# execute script
if __name__ == '__main__':
    main()
