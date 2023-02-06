import os
import argparse

from joint_diffusion_structural_seg.ablation_training import ablation_training_function

def main():

    ## parse arguments
    parser = argparse.ArgumentParser(description="Script to train a version of the joint structural + DTI "
                                                 "thalamus segmentor from a list of ablations", epilog='\n',
                                     add_help=True)

    parser.add_argument("--top_dir", help="top level directory containing both a data and model directory",
                        default="../")

    args = parser.parse_args()

    run_ablation_03(top_level_dir=args.top_dir)


def run_ablation_03( top_level_dir ) :

    ## machine specific directories
    top_level_data_dir = os.path.join(top_level_dir,'data')
    top_level_model_dir = os.path.join(top_level_dir,'models')

    assert os.path.exists(top_level_data_dir), \
        "top_dir must contain data directory"

    ## Run specific parameters that will change for ablations
    # Name of the model - link to ablation spreadsheet
    model_name = 'joint_thalamus_ablation_03'
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
    seg_selection='grouped'
    # Dice version - "individual" do standard label-wise Dice, "grouped" also add contribution of groups and whole thalamus
    dice_version="individual"
    # directory and file locations based on if we're using the full labels or reduced set
    top_level_training_dir = os.path.join(top_level_data_dir,'training_reduced')
    label_list_name = 'proc_training_data_label_list_reduced.npy'
    group_list_name = 'proc_training_group_seg_reduced.npy'


    ablation_training_function(top_level_training_dir, label_list_name, group_list_name,
                                   top_level_model_dir, model_name, nonlinear_rotation,
                                   speckle_frac_selected, seg_selection, flag_deformation,
                                   deformation_max, dice_version)


# execute script
if __name__ == '__main__':
    main()
