import glob
import numpy as np
from joint_diffusion_structural_seg.utils import load_volume
from scipy.interpolate import RegularGridInterpolator as rgi

# TODO: right now it takes 3.5 seconds on my machine... not great, not terrible
# An alternative may be to train a fatter network, so the GPU is not the bottleneck?
def image_seg_generator(training_dir,
                        path_label_list,
                        batchsize=1,
                        scaling_bounds=0.15,
                        rotation_bounds=15,
                        max_noise_std=0.1,
                        max_noise_std_fa=0.03,
                        gamma_std=0.1,
                        contrast_std=0.1,
                        brightness_std=0.1,
                        crop_size=None):

    # Read directory to get list of training cases
    t1_list = glob.glob(training_dir + '/*.t1.nii.gz')
    n_training = len(t1_list)
    print('Found %d cases for training' % n_training)

    # Get size from first volume
    aux = load_volume(t1_list[0])
    nx = aux.shape[0]
    ny = aux.shape[1]
    nz = aux.shape[2]
    if crop_size is None:
        crop_size = aux.shape
    if type(crop_size) == int:
        crop_size = [crop_size] *3

    # Create meshgrid (we will reuse a lot)
    xx, yy, zz = np.meshgrid(range(nx), range(ny), range(nz), sparse=False, indexing='ij')
    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)
    cz = 0.5 * (nz - 1)
    xc = xx - cx
    yc = yy - cy
    zc = zz - cz

    # Some useful precomputations for one-hot encoding
    label_list = np.sort(np.load(path_label_list)).astype(int)
    mapping = np.zeros(1 + label_list[-1], dtype='int')
    mapping[label_list] = np.arange(len(label_list))
    xxcrop, yycrop, zzcrop = np.meshgrid(range(crop_size[0]), range(crop_size[1]), range(crop_size[2]), sparse=False, indexing='ij')
    xxcrop = xxcrop.flatten()
    yycrop = yycrop.flatten()
    zzcrop = zzcrop.flatten()

    # Generate!
    while True:

        # randomly pick as many images as batchsize
        indices = np.random.randint(n_training, size=batchsize)

        # initialise input lists
        list_images = []
        list_label_maps = []

        for idx in indices:

            # read images
            t1_file = t1_list[idx]
            prefix = t1_list[idx][:-10]
            fa_file = prefix + '.fa.nii.gz'
            v1_file = prefix + '.v1.nii.gz'
            seg_file = prefix + '.seg.nii.gz'

            t1, aff, _ = load_volume(t1_file, im_only=False)
            fa = load_volume(fa_file)
            v1 = load_volume(v1_file)
            seg = load_volume(seg_file)

            # Create random rotation matrix and scaling, and apply to coordinates,
            rot = (2 * rotation_bounds * np.random.rand(3) - rotation_bounds) / 180.0 * np.pi
            Rx = np.array([[1, 0, 0], [0, np.cos(rot[0]), -np.sin(rot[0])], [0, np.sin(rot[0]), np.cos(rot[0])]])
            Ry = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])], [0, 1, 0],[-np.sin(rot[1]), 0, np.cos(rot[1])]])
            Rz = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0], [np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])
            R = np.matmul(np.matmul(Rx, Ry), Rz)
            Rinv = np.linalg.inv(R)
            s = 1 + (2 * scaling_bounds * np.random.rand(1) - scaling_bounds)

            xx2 = cx + s * (R[0, 0] * xc + R[0, 1] * yc + R[0, 2] * zc)
            yy2 = cy + s * (R[1, 0] * xc + R[1, 1] * yc + R[1, 2] * zc)
            zz2 = cz + s * (R[2, 0] * xc + R[2, 1] * yc + R[2, 2] * zc)

            # Interpolate!  TODO: this could be done more efficiently by groupoing t1 and fa into a single 4D array (and
            # interpolating all components of v1 at the same time, in a similar way). But this longer version is clearer
            # Also: there is no need to interpolate everywhere; only in the area we will (randomly) crop
            # Essentially, we crop and interpolate at the same time
            cropx = np.random.randint(0, nx - crop_size[0] + 1, 1)[0]
            cropy = np.random.randint(0, ny - crop_size[1] + 1, 1)[0]
            cropz = np.random.randint(0, nz - crop_size[2] + 1, 1)[0]
            xx2 = xx2[cropx:cropx+crop_size[0], cropy:cropy+crop_size[1], cropz:cropz+crop_size[2]]
            yy2 = yy2[cropx:cropx + crop_size[0], cropy:cropy + crop_size[1], cropz:cropz + crop_size[2]]
            zz2 = zz2[cropx:cropx + crop_size[0], cropy:cropy + crop_size[1], cropz:cropz + crop_size[2]]

            # We do FA and T1 in one shot with complex numbers :-)
            #
            # t1_interpolator = rgi((range(nx), range(ny), range(nz)), t1, method='linear', bounds_error=False, fill_value=0.0)
            # t1_def = t1_interpolator((xx2, yy2, zz2))
            # fa_interpolator = rgi((range(nx), range(ny), range(nz)), fa, method='linear', bounds_error=False, fill_value=0.0)
            # fa_def = fa_interpolator((xx2, yy2, zz2))
            #
            combo = np.array(t1, dtype=complex)
            combo.imag = fa
            combo_interpolator = rgi((range(nx), range(ny), range(nz)), combo, method='linear', bounds_error=False, fill_value=0.0)
            combo_def = combo_interpolator((xx2, yy2, zz2))
            t1_def = np.real(combo_def)
            fa_def = np.imag(combo_def)

            # We also avoid building a bunch of interpolators by doing the nearest neighbor interpolation ourselves
            #
            # v1_def = np.zeros((*crop_size, 3))
            # v1_interpolator_0 = rgi((range(nx), range(ny), range(nz)), v1[:, :, :, 0], method='nearest', bounds_error=False, fill_value=np.sqrt(1.0 / 3.0))
            # v1_interpolator_1 = rgi((range(nx), range(ny), range(nz)), v1[:, :, :, 1], method='nearest', bounds_error=False, fill_value=np.sqrt(1.0 / 3.0))
            # v1_interpolator_2 = rgi((range(nx), range(ny), range(nz)), v1[:, :, :, 2], method='nearest', bounds_error=False, fill_value=np.sqrt(1.0 / 3.0))
            # v1_def[:, :, :, 0] = - v1_interpolator_0((xx2, yy2, zz2))  # TODO: this -1 is crucial... I wonder if it's the same for every FreeSurfer processed dataset
            # v1_def[:, :, :, 1] = v1_interpolator_1((xx2, yy2, zz2))
            # v1_def[:, :, :, 2] = v1_interpolator_2((xx2, yy2, zz2))
            # seg_interpolator = rgi((range(nx), range(ny), range(nz)), seg, method='nearest', bounds_error=False, fill_value=0.0)
            # seg_def = seg_interpolator((xx2, yy2, zz2)).astype(int)
            #
            xx2r = np.round(xx2).astype(int)
            yy2r = np.round(yy2).astype(int)
            zz2r = np.round(zz2).astype(int)
            ok = (xx2r >= 0) & (yy2r >= 0) & (zz2r >= 0) & (xx2r <= nx - 1) & (yy2r <= ny - 1) & (zz2r <= nz - 1)
            idx = xx2r[ok] + nx * yy2r[ok] + nx * ny * zz2r[ok]
            v1_def = np.zeros((*crop_size, 3))
            # TODO: the -1 in the first coordinate (left-right) -1 is crucial
            # I wonder if (/hope!) it's the same for every FreeSurfer processed dataset
            v1_def[:, :, :, 0][ok] = - v1[:, :, :, 0].flatten(order='F')[idx]
            v1_def[:, :, :, 1][ok] = v1[:, :, :, 1].flatten(order='F')[idx]
            v1_def[:, :, :, 2][ok] = v1[:, :, :, 2].flatten(order='F')[idx]
            seg_def = np.zeros(crop_size)
            seg_def[ok] = seg.flatten(order='F')[idx]

            # We also need to rotate v1
            v1_def_rot = np.zeros_like(v1_def)
            for row in range(3):
                for col in range(3):
                    v1_def_rot[:, :, :, row] = v1_def_rot[:, :, :, row] + Rinv[row, col] * v1_def[:, :, :, col]

            # Augment t1 and fa, and compute final DTI (RGB) volume with new FA
            gamma_fa = np.exp(gamma_std * np.random.randn(1)[0])
            noise_std = max_noise_std_fa * np.random.rand(1)[0]
            fa_def = fa_def + noise_std * np.random.randn(*fa_def.shape)
            fa_def[fa_def < 0] = 0
            fa_def[fa_def > 1] = 1
            fa_def = fa_def ** gamma_fa
            dti_def = np.abs(v1_def_rot * fa_def[..., np.newaxis])

            # TODO: maybe add bias field? If we're working with FreeSurfer processed images maybe it's not too important
            gamma_t1 = np.exp(gamma_std * np.random.randn(1)[0]) # TODO: maybe make it spatially variable?
            contrast = np.min((1.4, np.max((0.6, 1.0 + contrast_std * np.random.randn(1)[0]))))
            brightness = np.min((0.4, np.max((-0.4, brightness_std * np.random.randn(1)[0]))))
            noise_std = max_noise_std * np.random.rand(1)[0]
            t1_def = ((t1_def - 0.5) * contrast + (0.5 + brightness)) + noise_std * np.random.randn(*t1_def.shape)
            t1_def[t1_def < 0] = 0
            t1_def[t1_def > 1] = 1
            t1_def = t1_def ** gamma_t1


            # TODO: possible improvement: introduce left right flipping. You need to a) flip all the volumes, b) swap
            # left and right labels in the flipped segmentation, c) change the sign of the flipped v1_def_rot[:, :, :, 0]


            # Efficiently turn label map into one hot encoded array
            seg_def = mapping[seg_def.astype(int)]
            aux = np.zeros(t1_def.size * len(label_list))
            idx = xxcrop + yycrop * t1_def.shape[0] + zzcrop * t1_def.shape[0] * t1_def.shape[1] \
                  + seg_def.flatten() * t1_def.size # This is essentially a Matlab sub2ind
            aux[idx] = 1.0
            onehot = aux.reshape((*t1_def.shape, len(label_list)), order='F')

            # If you want to save to disk and open with Freeview during debugging
            # from joint_diffusion_structural_seg.utils import save_volume
            # save_volume(t1, aff, None, '/tmp/t1.mgz')
            # save_volume(t1_def, aff, None, '/tmp/t1_def.mgz')
            # save_volume(fa, aff, None, '/tmp/fa.mgz')
            # save_volume(fa_def, aff, None, '/tmp/fa_def.mgz')
            # save_volume(seg, aff, None, '/tmp/seg.mgz')
            # save_volume(seg_def, aff, None, '/tmp/seg_def.mgz')
            # save_volume(v1, aff, None, '/tmp/v1.mgz')
            # save_volume(v1_def_rot, aff, None, '/tmp/v1_def.mgz')
            # dti = np.abs(v1 * fa[..., np.newaxis])
            # save_volume(dti * 255, aff, None, '/tmp/dti.mgz')
            # save_volume(dti_def * 255, aff, None, '/tmp/dti_def.mgz')
            # save_volume(onehot, aff, None, '/tmp/onehot_def.mgz')

            list_images.append(np.concatenate((t1_def[..., np.newaxis], fa_def[..., np.newaxis], dti_def ), axis=-1)[np.newaxis,...])
            list_label_maps.append(onehot[np.newaxis,...])

        # build list of inputs of augmentation model
        list_inputs = [list_images, list_label_maps]
        if batchsize > 1:  # concatenate each input type if batchsize > 1
            list_inputs = [np.concatenate(item, 0) for item in list_inputs]
        else:
            list_inputs = [item[0] for item in list_inputs]

        yield list_inputs
