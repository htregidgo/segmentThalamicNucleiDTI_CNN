import numpy as np
import os

import torch

from joint_diffusion_structural_seg import utils

def randomly_resample_dti(v1, fa, R, s, xc, yc, zc, cx, cy, cz, crop_size, cropx, cropy, cropz,
                          flag_deformation=True, deformation_max=5.0, t1_resolution=np.array([0.7]*3)):


    centre = torch.tensor((cx, cy, cz))

    # get rotation displacement
    displacement_full = torch.matmul(R,(torch.cat((xc[..., None], yc[..., None], zc[..., None]), dim=-1))[..., None])[..., 0]
    displacement_full *= s
    displacement_full += centre[None, None, None, :]

    displacement = displacement_full[cropx:cropx+crop_size[0]+1,
                                cropy:cropy+crop_size[1]+1,
                                cropz:cropz+crop_size[2]+1, :]

    if flag_deformation:
        # Add trilinear spline deformation
        n_defNodes = 5
        def_basis_size = np.array([n_defNodes] * 3)
        basis_dist = np.array(s) * (np.array(crop_size) - 1.) / (def_basis_size - 1.)

        deformation_max_vox = deformation_max / t1_resolution
        def_basis_seed = np.random.rand(3, def_basis_size[0], def_basis_size[1], def_basis_size[2])

        replace_def_max = deformation_max_vox > (basis_dist / 3.)
        deformation_max_vox[replace_def_max] = basis_dist[replace_def_max] / 3.

        def_basis = 2 * deformation_max_vox[:, None, None, None] * (def_basis_seed - 0.5)

        def_basis = torch.tensor(def_basis[None, ...], device='cpu')

        def_array = torch.nn.functional.interpolate(def_basis,
                                                    size=(crop_size[0]+1, crop_size[1]+1, crop_size[2]+1),
                                                    mode='trilinear',
                                                    align_corners=True)

        displacement += def_array[0,...].permute((1,2,3,0))

    left = slice(0, -1)
    right = slice(1, None)

    # get jacobian of displacement field using forward differences
    jacobian = torch.empty((crop_size[0], crop_size[1], crop_size[2], 3, 3))

    jacobian[..., 0] = displacement[right, left, left, :] - displacement[left, left, left, :]
    jacobian[..., 1] = displacement[left, right, left, :] - displacement[left, left, left, :]
    jacobian[..., 2] = displacement[left, left, right, :] - displacement[left, left, left, :]

    displacement_out = displacement[left, left, left]

    # rotation is U * Vh where U and V are the singular vector matrices
    U, Vh = torch.linalg.svd(jacobian)[slice(0, None, 2)]

    # use transpose as we're using displacement from target to source
    R_reorient = torch.transpose(torch.matmul(U, Vh), -1, -2)

    dti_def = resmple_dti(fa, v1, displacement_out, R_reorient)

    xx2 = displacement_out[..., 0]
    yy2 = displacement_out[..., 1]
    zz2 = displacement_out[..., 2]


    return dti_def, xx2, yy2, zz2



def randomly_resample_dti_PPD(v1, fa, R, s, xc, yc, zc, cx, cy, cz, crop_size, cropx, cropy, cropz,
                          flag_deformation=True, deformation_max=5.0, t1_resolution=np.array([0.7]*3)):


    centre = torch.tensor((cx, cy, cz))

    # get rotation displacement
    displacement_full = torch.matmul(R,(torch.cat((xc[..., None], yc[..., None], zc[..., None]), dim=-1))[..., None])[..., 0]
    displacement_full *= s
    displacement_full += centre[None, None, None, :]

    displacement = displacement_full[cropx:cropx+crop_size[0]+1,
                                cropy:cropy+crop_size[1]+1,
                                cropz:cropz+crop_size[2]+1, :]

    if flag_deformation:
        # Add trilinear spline deformation
        n_defNodes = 5
        def_basis_size = np.array([n_defNodes] * 3)
        basis_dist = np.array(s) * (np.array(crop_size) - 1.) / (def_basis_size - 1.)

        deformation_max_vox = deformation_max / t1_resolution
        def_basis_seed = np.random.rand(3, def_basis_size[0], def_basis_size[1], def_basis_size[2])

        replace_def_max = deformation_max_vox > (basis_dist / 3.)
        deformation_max_vox[replace_def_max] = basis_dist[replace_def_max] / 3.

        def_basis = 2 * deformation_max_vox[:, None, None, None] * (def_basis_seed - 0.5)

        def_basis = torch.tensor(def_basis[None, ...], device='cpu')

        def_array = torch.nn.functional.interpolate(def_basis,
                                                    size=(crop_size[0]+1, crop_size[1]+1, crop_size[2]+1),
                                                    mode='trilinear',
                                                    align_corners=True)

        displacement += def_array[0,...].permute((1,2,3,0))

    left = slice(0, -1)
    right = slice(1, None)

    # get jacobian of displacement field using forward differences
    jacobian = torch.empty((crop_size[0], crop_size[1], crop_size[2], 3, 3))

    jacobian[..., 0] = displacement[right, left, left, :] - displacement[left, left, left, :]
    jacobian[..., 1] = displacement[left, right, left, :] - displacement[left, left, left, :]
    jacobian[..., 2] = displacement[left, left, right, :] - displacement[left, left, left, :]

    displacement_out = displacement[left, left, left]

    dti_def = resmple_dti_PPD(fa, v1, displacement_out, jacobian)

    xx2 = displacement_out[..., 0]
    yy2 = displacement_out[..., 1]
    zz2 = displacement_out[..., 2]


    return dti_def, xx2, yy2, zz2



def resmple_dti(fa, v1, displacement, R_reorient):

    ok = torch.all((displacement > 0), dim=-1) & (displacement[..., 0] < fa.shape[0]) & (
            displacement[..., 1] < fa.shape[1]) & (displacement[..., 2] < fa.shape[2])

    n_ok = torch.sum(ok)

    Inds_v = torch.masked_select(displacement, ok[..., None])
    Inds_v = torch.reshape(Inds_v, (n_ok, 3))

    R_reorient = torch.reshape(torch.masked_select(R_reorient,ok[..., None, None]), (n_ok,3,3))

    fx = torch.floor(Inds_v[..., 0]).long()
    cx = fx + 1
    cx[cx > (fa.shape[0] - 1)] = (fa.shape[0] - 1)
    wcx = Inds_v[..., 0] - fx
    wfx = 1 - wcx

    fy = torch.floor(Inds_v[..., 1]).long()
    cy = fy + 1
    cy[cy > (fa.shape[1] - 1)] = (fa.shape[1] - 1)
    wcy = Inds_v[..., 1] - fy
    wfy = 1 - wcy

    fz = torch.floor(Inds_v[..., 2]).long()
    cz = fz + 1
    cz[cz > (fa.shape[2] - 1)] = (fa.shape[2] - 1)
    wcz = Inds_v[..., 2] - fz
    wfz = 1 - wcz

    c00 = (wfx[..., None] * (fa[fx, fy, fz])[..., None]) * torch.abs(torch.matmul(R_reorient, v1[fx, fy, fz, :, None]))[..., 0] \
          + (wcx[..., None] * (fa[cx, fy, fz])[..., None]) * torch.abs(torch.matmul(R_reorient, v1[cx, fy, fz, :, None]))[..., 0]
    c01 = (wfx[..., None] * (fa[fx, fy, cz])[..., None]) * torch.abs(torch.matmul(R_reorient, v1[fx, fy, cz, :, None]))[..., 0] \
          + (wcx[..., None] * (fa[cx, fy, cz])[..., None]) * torch.abs(torch.matmul(R_reorient, v1[cx, fy, cz, :, None]))[..., 0]
    c10 = (wfx[..., None] * (fa[fx, cy, fz])[..., None]) * torch.abs(torch.matmul(R_reorient, v1[fx, cy, fz, :, None]))[..., 0] \
          + (wcx[..., None] * (fa[cx, cy, fz])[..., None]) * torch.abs(torch.matmul(R_reorient, v1[cx, cy, fz, :, None]))[..., 0]
    c11 = (wfx[..., None] * (fa[fx, cy, cz])[..., None]) * torch.abs(torch.matmul(R_reorient, v1[fx, cy, cz, :, None]))[..., 0] \
          + (wcx[..., None] * (fa[cx, cy, cz])[..., None]) * torch.abs(torch.matmul(R_reorient, v1[cx, cy, cz, :, None]))[..., 0]


    c0 = c00 * wfy[..., None] + c10 * wcy[..., None]
    c1 = c01 * wfy[..., None] + c11 * wcy[..., None]

    c = c0 * wfz[..., None] + c1 * wcz[..., None]

    dti_def = torch.zeros(*displacement.shape, device='cpu')

    dti_def.masked_scatter_(ok[..., None], c.float())

    return dti_def



def resmple_dti_PPD(fa, v1, displacement, F_reorient):

    ok = torch.all((displacement > 0), dim=-1) & (displacement[..., 0] < fa.shape[0]) & (
            displacement[..., 1] < fa.shape[1]) & (displacement[..., 2] < fa.shape[2])

    n_ok = torch.sum(ok)

    Inds_v = torch.masked_select(displacement, ok[..., None])
    Inds_v = torch.reshape(Inds_v, (n_ok, 3))

    F_reorient = torch.reshape(torch.masked_select(F_reorient,ok[..., None, None]), (n_ok,3,3))

    F_factor = torch.lu(F_reorient)

    fx = torch.floor(Inds_v[..., 0]).long()
    cx = fx + 1
    cx[cx > (fa.shape[0] - 1)] = (fa.shape[0] - 1)
    wcx = Inds_v[..., 0] - fx
    wfx = 1 - wcx

    fy = torch.floor(Inds_v[..., 1]).long()
    cy = fy + 1
    cy[cy > (fa.shape[1] - 1)] = (fa.shape[1] - 1)
    wcy = Inds_v[..., 1] - fy
    wfy = 1 - wcy

    fz = torch.floor(Inds_v[..., 2]).long()
    cz = fz + 1
    cz[cz > (fa.shape[2] - 1)] = (fa.shape[2] - 1)
    wcz = Inds_v[..., 2] - fz
    wfz = 1 - wcz


    left_sample =torch.lu_solve(v1[fx, fy, fz, :, None],*F_factor)[..., 0]
    left_sample /= torch.linalg.norm(left_sample,dim=1)[...,None] +1.0e-10
    right_sample=torch.lu_solve(v1[cx, fy, fz, :, None],*F_factor)[..., 0]
    right_sample /= torch.linalg.norm(right_sample,dim=1)[...,None] +1.0e-10
    c00 = (wfx[..., None] * (fa[fx, fy, fz])[..., None]) * torch.abs(left_sample) \
          + (wcx[..., None] * (fa[cx, fy, fz])[..., None]) * torch.abs(right_sample)

    left_sample = torch.lu_solve(v1[fx, fy, cz, :, None],*F_factor)[..., 0]
    left_sample /= torch.linalg.norm(left_sample, dim=1)[..., None] +1.0e-10
    right_sample = torch.lu_solve(v1[cx, fy, cz, :, None],*F_factor)[..., 0]
    right_sample /= torch.linalg.norm(right_sample, dim=1)[..., None] +1.0e-10
    c01 = (wfx[..., None] * (fa[fx, fy, cz])[..., None]) * torch.abs(left_sample) \
          + (wcx[..., None] * (fa[cx, fy, cz])[..., None]) * torch.abs(right_sample)

    left_sample = torch.lu_solve(v1[fx, cy, fz, :, None],*F_factor)[..., 0]
    left_sample /= torch.linalg.norm(left_sample, dim=1)[..., None] +1.0e-10
    right_sample = torch.lu_solve(v1[cx, cy, fz, :, None],*F_factor)[..., 0]
    right_sample /= torch.linalg.norm(right_sample, dim=1)[..., None] +1.0e-10
    c10 = (wfx[..., None] * (fa[fx, cy, fz])[..., None]) * torch.abs(left_sample) \
          + (wcx[..., None] * (fa[cx, cy, fz])[..., None]) * torch.abs(right_sample)

    left_sample = torch.lu_solve(v1[fx, cy, cz, :, None],*F_factor)[..., 0]
    left_sample /= torch.linalg.norm(left_sample, dim=1)[..., None] +1.0e-10
    right_sample = torch.lu_solve(v1[cx, cy, cz, :, None],*F_factor)[..., 0]
    right_sample /= torch.linalg.norm(right_sample, dim=1)[..., None] +1.0e-10
    c11 = (wfx[..., None] * (fa[fx, cy, cz])[..., None]) * torch.abs(left_sample) \
          + (wcx[..., None] * (fa[cx, cy, cz])[..., None]) * torch.abs(right_sample)



    c0 = c00 * wfy[..., None] + c10 * wcy[..., None]
    c1 = c01 * wfy[..., None] + c11 * wcy[..., None]

    c = c0 * wfz[..., None] + c1 * wcz[..., None]

    dti_def = torch.zeros(*displacement.shape, device='cpu')

    dti_def.masked_scatter_(ok[..., None], c.float())

    return dti_def


