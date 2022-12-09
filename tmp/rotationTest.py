import glob
import numpy as np
import os
from joint_diffusion_structural_seg import utils
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.ndimage import gaussian_filter as gauss_filt
import torch
from joint_diffusion_structural_seg import dtiutils
from joint_diffusion_structural_seg.generators import fast_3D_interp_torch

def image_seg_generator_rgb_rotationtest(training_dir,
                            path_label_list,
                            batchsize=1,
                            scaling_bounds=0.15,
                            rotation_bounds=90,
                            nonlinear_rotation=True,
                            max_noise_std=0.1,
                            max_noise_std_fa=0.03,
                            gamma_std=0.1,
                            contrast_std=0.1,
                            brightness_std=0.1,
                            crop_size=None,
                            randomize_resolution=False,
                            diffusion_resolution=None,
                            randomize_speckle=True,
                            randomize_flip=True,
                            seg_selection='combined'):

    # check type of one-hot encoding
    assert (seg_selection == 'single') or (seg_selection == 'combined'),\
        'seg_selection must be single or combined'

    # Read directory to get list of training cases
    t1_list = glob.glob(training_dir + '/subject*/*.t1.nii.gz')
    n_training = len(t1_list)
    print('Found %d cases for training' % n_training)

    # Get size from first volume
    aux, aff, _ = utils.load_volume(t1_list[0], im_only=False)
    t1_resolution = np.sum(aff,axis=0)[:-1]
    nx, ny, nz = aux.shape
    if crop_size is None:
        crop_size = aux.shape
    if type(crop_size) == int:
        crop_size = [crop_size] *3

    # Create meshgrid (we will reuse a lot)
    xx, yy, zz = np.meshgrid(range(nx), range(ny), range(nz), sparse=False, indexing='ij')
    cx, cy, cz = (np.array(aux.shape) - 1) / 2
    xc = xx - cx
    yc = yy - cy
    zc = zz - cz
    xc = torch.tensor(xc, device='cpu')
    yc = torch.tensor(yc, device='cpu')
    zc = torch.tensor(zc, device='cpu')
    cx = torch.tensor(cx, device='cpu')
    cy = torch.tensor(cy, device='cpu')
    cz = torch.tensor(cz, device='cpu')

    # Some useful precomputations for one-hot encoding
    label_list = np.sort(np.load(path_label_list)).astype(int)
    mapping = np.zeros(1 + label_list[-1], dtype='int')
    mapping[label_list] = np.arange(len(label_list))
    mapping = torch.tensor(mapping, device='cpu').long()



    # indices =0

    # Generate!
    count = 0
    while True:

        # randomly pick as many images as batchsize
        indices = [0]
        # indices = [count]

        # at end of loop indices += 1
        # overflow to 0 once through all examples

        # initialise input lists
        list_images = []
        list_label_maps = []

        for index in indices:

            # read images
            # TODO: this may go wrong with a larger batchsize
            t1_file = t1_list[index]
            subject_path = os.path.split(t1_file)[0]

            seg_list = glob.glob(subject_path + '/segs/*nii.gz')

            # either pick a single seg to train towards or import them all and average the onehot
            if seg_selection == 'single':
                seg_index = np.random.randint(len(seg_list))
                seg_file = seg_list[seg_index]
                seg = utils.load_volume(seg_file)
                seg = torch.tensor(seg, device='cpu').long()
            else:
                seg = utils.load_volume(seg_list[0])
                seg = torch.tensor(seg, device='cpu').long()
                seg = seg[..., None]
                for il in range(1, len(seg_list)):
                    np_seg = utils.load_volume(seg_list[il])
                    seg = torch.concat((seg, torch.tensor(np_seg[..., None], device='cpu')), dim=3)


            fa_list = glob.glob(subject_path + '/dmri/*_fa.nii.gz')
            fa_index = np.random.randint(len(fa_list))

            fa_file = fa_list[fa_index]
            prefix = fa_file[:-10]
            v1_file = prefix + '_v1.nii.gz'

            t1, aff, _ = utils.load_volume(t1_file, im_only=False)
            fa = utils.load_volume(fa_file)
            v1 = utils.load_volume(v1_file)
            t1 = torch.tensor(t1, device='cpu')
            aff = torch.tensor(aff, device='cpu')
            fa = torch.tensor(fa, device='cpu')
            v1 = torch.tensor(v1, device='cpu')

            # Sample augmentation parameters
            rotations = (2 * rotation_bounds * np.random.rand(3) - rotation_bounds) / 180.0 * np.pi
            s = torch.tensor(1 + (2 * scaling_bounds * np.random.rand(1) - scaling_bounds), device='cpu')

            s=1
            rotations[0]=0
            rotations[1]=0
            rotations[2]=np.pi/2

            cropx = np.random.randint(0, nx - crop_size[0] + 1, 1)[0]
            cropy = np.random.randint(0, ny - crop_size[1] + 1, 1)[0]
            cropz = np.random.randint(0, nz - crop_size[2] + 1, 1)[0]

            # Create random rotation matrix and scaling, and apply to v1 and to coordinates,
            R = utils.make_rotation_matrix(rotations)
            Rinv = np.linalg.inv(R)
            R = torch.tensor(R, device='cpu')
            Rinv = torch.tensor(Rinv, device='cpu')

            # TODO: the -1 in the first coordinate (left-right) -1 is crucial
            # I wonder if (/hope!) it's the same for every FreeSurfer processed dataset
            v1[:, :, :, 0] = - v1[:, :, :, 0]

            v1_rot = torch.zeros(v1.shape, device='cpu')
            for row in range(3):
                for col in range(3):
                    v1_rot[:, :, :, row] = v1_rot[:, :, :, row] + Rinv[row, col] * v1[:, :, :, col]

            xx2 = cx + s * (R[0, 0] * xc + R[0, 1] * yc + R[0, 2] * zc)
            yy2 = cy + s * (R[1, 0] * xc + R[1, 1] * yc + R[1, 2] * zc)
            zz2 = cz + s * (R[2, 0] * xc + R[2, 1] * yc + R[2, 2] * zc)

            # We use the rotated v1 to create the RGB (DTI) volume and then forget about V1 / nearest neighbor interpolation
            dti = np.abs(v1_rot * fa[..., np.newaxis])

            # Interpolate!  There is no need to interpolate everywhere; only in the area we will (randomly) crop
            # Essentially, we crop and interpolate at the same time
            xx2 = xx2[cropx:cropx+crop_size[0], cropy:cropy+crop_size[1], cropz:cropz+crop_size[2]]
            yy2 = yy2[cropx:cropx + crop_size[0], cropy:cropy + crop_size[1], cropz:cropz + crop_size[2]]
            zz2 = zz2[cropx:cropx + crop_size[0], cropy:cropy + crop_size[1], cropz:cropz + crop_size[2]]


            combo = torch.concat( (t1[...,None], dti), dim=-1 )
            combo_def = fast_3D_interp_torch(combo, xx2, yy2, zz2, 'linear')
            t1_def = combo_def[:, :, :, 0]
            dti_def = combo_def[:, :, :, 1:]
            fa_def = torch.sqrt(torch.sum(dti_def * dti_def, dim=-1))
            v1_def = dti_def/(fa_def[..., None] + 1e-6)

            dti_def2 = dtiutils.randomly_resample_dti(v1, fa, R, s, xc, yc, zc, cx, cy, cz, crop_size, cropx, cropy, cropz)[0]
            fa_def2 = torch.sqrt(torch.sum(dti_def2 * dti_def2, dim=-1))
            v1_def2 = dti_def2 / (fa_def2[..., None] + 1e-6)

            # If you want to save to disk and open with Freeview during debugging
            from joint_diffusion_structural_seg.utils import save_volume
            utils.save_volume(t1, aff, None, '/tmp/t1.mgz')
            utils.save_volume(t1_def, aff, None, '/tmp/t1_def.mgz')
            utils.save_volume(fa, aff, None, '/tmp/fa.mgz')
            utils.save_volume(fa_def, aff, None, '/tmp/fa_def.mgz')
            utils.save_volume(fa_def2, aff, None, '/tmp/fa_def2.mgz')
            # utils.save_volume(seg, aff, None, '/tmp/seg.mgz')
            # utils.save_volume(seg_def, aff, None, '/tmp/seg_def.mgz')
            utils.save_volume(v1, aff, None, '/tmp/v1.mgz')
            utils.save_volume(v1_def, aff, None, '/tmp/v1_def.mgz')
            utils.save_volume(v1_def2, aff, None, '/tmp/v1_def2.mgz')
            # dti = np.abs(v1 * fa[..., np.newaxis])
            # utils.save_volume(dti * 255, aff, None, '/tmp/dti.mgz')
            # utils.save_volume(dti_def * 255, aff, None, '/tmp/dti_def.mgz')
            # utils.save_volume(dti_def / (fa_def[..., None] + 1e-6), aff, None, '/tmp/v1_def.mgz')
            # utils.save_volume(onehot, aff, None, '/tmp/onehot_def.mgz')

            list_images.append((torch.concat((t1_def[..., None], fa_def[..., None], dti_def), axis=-1)[None, ...]).detach().numpy())
            list_label_maps.append((torch.concat((t1_def[..., None], fa_def[..., None], dti_def), axis=-1)[None, ...]).detach().numpy())

            count += 1
            # if count == len(number of examples in validation set):
            #     count = 0

        # build list of inputs of augmentation model
        list_inputs = [list_images, list_label_maps]
        if batchsize > 1:  # concatenate each input type if batchsize > 1
            list_inputs = [np.concatenate(item, 0) for item in list_inputs]
        else:
            list_inputs = [item[0] for item in list_inputs]

        yield list_inputs