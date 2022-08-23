# This script creates a partial dataset to train the CNN
# It essentially goes around the FreeSurfer subject directory of the HCP datset, and:
# 1. Reorients the T1s so the vox2ras header is approximately diagonal
# 2. Rescales the T1s so that the median of the white matter is 0.75 (and clips to [0,1])
# 3. Locates and resamples Bayesian segmentations to the space of the T1
# 4. Crops the volumes around the thalami (we don't want to look too far from them)
#
# This is only a partial training set as we still need to get the dti files resampled using log tensors.
import os
import numpy as np
import joint_diffusion_structural_seg.utils as utils
import glob

henry_seg_dir = '/home/henry/Documents/Brain/ThalamicNuclei/HTtestdata'
fs_dir = '/home/henry/Documents/Brain/HCPDataset/HCP/'
# output_dir = '/autofs/space/panamint_005/users/iglesias/data/joint_diffusion_structural_seg/proc_training_data/'
output_dir = '/home/henry/Documents/Brain/synthDTI/4henry/data/training_new_wrongseg/'
# output_label_list = '/autofs/space/panamint_005/users/iglesias/data/joint_diffusion_structural_seg/proc_training_data_label_list.npy'
output_label_list = '/home/henry/Documents/Brain/synthDTI/4henry/data/proc_training_data_label_list_new.npy'
suffix = '_JointStructDTI_seg_final.nii.gz'
W = 160

# Create output directory if needed
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# We'll keep track of the labels we encounter to compile a list
label_list = None

result_suffix = '/results/EM'
subject_list = glob.glob(henry_seg_dir + '/subject*' + result_suffix + '/DSW*dti1k*')


# Loop over files
for folder in subject_list:
    em_folder = os.path.split(folder)[0]
    list_GT_comp = glob.glob(em_folder + '/*GTcomp*')
    if len(list_GT_comp) > 0:
        continue
    else:
        subject = os.path.split(em_folder[:-len(result_suffix)])[1]
        print(subject)

        subject_out_dir = os.path.join(output_dir, subject)
        if not os.path.exists(subject_out_dir):
            os.mkdir(subject_out_dir)

        if not os.path.exists(os.path.join(subject_out_dir, 'segs')):
            os.mkdir(os.path.join(subject_out_dir, 'segs'))

        seg_file_list_1K = glob.glob(em_folder + '/*dti1k*/*' + suffix)
        seg_file_list_2K = glob.glob(em_folder + '/*dti2k*/*' + suffix)
        t1_file = os.path.join(fs_dir, subject, 'mri','T1w_hires.masked.norm.mgz')
        aseg_file = os.path.join(fs_dir, subject, 'mri','aseg.mgz')

        # log_tensor_file_123k = os.path.join(fs_dir, subject, 'dmri', 'cleaned', subject + '_LogTensorsElems_hr.mgz')

        # fa_file = os.path.join(fs_dir, subject, 'dmri','dtifit.1+2+3K_FA.nii.gz')
        # v1_file = os.path.join(fs_dir, subject, 'dmri', 'dtifit.1+2+3K_V1.nii.gz')

        # Read in and reorient T1
        aff_ref = np.eye(4)
        t1, aff, _ = utils.load_volume(t1_file, im_only=False)
        t1, aff2 = utils.align_volume_to_ref(t1, aff, aff_ref=aff_ref, return_aff=True, n_dims=3)

        # Read and resample all the other volumes
        aseg, aff, _ = utils.load_volume(aseg_file, im_only=False)
        aseg = utils.resample_like(t1, aff2, aseg, aff, method='nearest')
        # fa, aff, _ = utils.load_volume(fa_file, im_only=False)
        # fa = utils.resample_like(t1, aff2, fa, aff)
        # v1, aff, _ = utils.load_volume(v1_file, im_only=False)
        # TODO: we'll want to do this in the log-tensor domain
        # but for now I simply interpolate with nearest neighbors
        # v1_copy = v1.copy()
        # v1 = np.zeros([*t1.shape,3])
        # for c in range(3):
        #     v1[:,:,:,c] = utils.resample_like(t1, aff2, v1_copy[:,:,:,c], aff, method='nearest')


        # Find the center of the thalamus and crop a volumes around it
        th_mask = (aseg==10) | (aseg==49)
        idx = np.where(th_mask)
        i1 = (np.mean(idx[0]) - np.round(0.5*W)).astype(int)
        j1 = (np.mean(idx[1]) - np.round(0.5*W)).astype(int)
        k1 = (np.mean(idx[2]) - np.round(0.5*W)).astype(int)
        i2 = i1 + W
        j2 = j1 + W
        k2 = k1 + W

        # Let's be picky and preserve the RAS coordinates
        aff3 = np.copy(aff2)
        aff3[:-1, -1] = aff2[:-1, -1] + np.matmul(aff2[:-1, :-1], np.array([i1, j1, k1]))

        for seg_file in seg_file_list_1K:
            if (seg_file[len(em_folder)+1:]).startswith('FA'):
                continue
            elif (seg_file[len(em_folder)+1:]).startswith('Trace'):
                continue
            elif (seg_file[len(em_folder)+1:]).startswith('DSWbeta'):
                out_suffix = '_1k_DSWbeta.nii.gz'
            elif (seg_file[len(em_folder)+1:]).startswith('wishart'):
                out_suffix = '_1k_Wishart.nii.gz'
            elif (seg_file[len(em_folder)+1:]).startswith('logFrobenius'):
                out_suffix = '_1k_LogGauss.nii.gz'
            else:
                continue
            # Clean up segmentation: extrathalamic and reticular must go
            seg, aff, _ = utils.load_volume(seg_file, im_only=False)
            seg = utils.resample_like(t1, aff2, seg, aff, method='nearest')
            seg = seg[i1:i2, j1:j2, k1:k2]
            # Clean up segmentation: extrathalamic and reticular must go
            seg[seg < 8000] = 0
            seg[seg == 8125] = 0
            seg[seg == 8225] = 0
            seg[seg > 8250] = 0
            utils.save_volume(seg, aff3, None, os.path.join(subject_out_dir, 'segs', subject + out_suffix))
            # Keep track of encountered labels
            if label_list is None:
                label_list = np.unique(seg)
            else:
                label_list = np.unique(np.concatenate((label_list, np.unique(seg))))


        for seg_file in seg_file_list_2K:
            if (seg_file[len(em_folder)+1:]).startswith('FA'):
                continue
            elif (seg_file[len(em_folder)+1:]).startswith('Trace'):
                continue
            elif (seg_file[len(em_folder)+1:]).startswith('DSWbeta'):
                out_suffix = '_2k_DSWbeta.nii.gz'
            elif (seg_file[len(em_folder)+1:]).startswith('wishart'):
                out_suffix = '_2k_Wishart.nii.gz'
            elif (seg_file[len(em_folder)+1:]).startswith('logFrobenius'):
                out_suffix = '_2k_LogGauss.nii.gz'
            else:
                continue
            # Clean up segmentation: extrathalamic and reticular must go
            seg, aff, _ = utils.load_volume(seg_file, im_only=False)
            seg = utils.resample_like(t1, aff2, seg, aff, method='nearest')
            seg = seg[i1:i2, j1:j2, k1:k2]
            # Clean up segmentation: extrathalamic and reticular must go
            seg[seg < 8000] = 0
            seg[seg == 8125] = 0
            seg[seg == 8225] = 0
            seg[seg > 8250] = 0
            utils.save_volume(seg, aff3, None, os.path.join(subject_out_dir, 'segs', subject + out_suffix))
            # Keep track of encountered labels
            if label_list is None:
                label_list = np.unique(seg)
            else:
                label_list = np.unique(np.concatenate((label_list, np.unique(seg))))

        # Normalize the T1
        wm_mask = (aseg==2) | (aseg==41)
        wm_t1_median = np.median(t1[wm_mask])
        t1 = t1 / wm_t1_median * 0.75
        t1[t1<0] = 0
        t1[t1>1] = 1

        t1 = t1[i1:i2, j1:j2, k1:k2]
        # fa = fa[i1:i2, j1:j2, k1:k2]
        # v1 = v1[i1:i2, j1:j2, k1:k2, :]


        # Write volumes to disk
        utils.save_volume(t1, aff3, None, os.path.join(subject_out_dir, subject + '.t1.nii.gz'))
        # utils.save_volume(seg, aff2, None, os.path.join(output_dir, subject + '.seg.nii.gz'))
        # utils.save_volume(fa, aff2, None, os.path.join(output_dir, subject + '.fa.nii.gz'))
        # utils.save_volume(v1, aff2, None, os.path.join(output_dir, subject + '.v1.nii.gz'))

# Save label list and exit
np.save(output_label_list, label_list)

print('All done!')




