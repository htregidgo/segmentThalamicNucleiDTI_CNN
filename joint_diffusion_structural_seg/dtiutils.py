import numpy as np
import os

import torch

from joint_diffusion_structural_seg import utils

def randomly_resample_dti(v1, fa, R, s, xc, yc, zc, cx, cy, cz, crop_size, nx, ny, nz):


    # get cropping
    cropx = np.random.randint(0, nx - crop_size[0] + 1, 1)[0]
    cropy = np.random.randint(0, ny - crop_size[1] + 1, 1)[0]
    cropz = np.random.randint(0, nz - crop_size[2] + 1, 1)[0]

    centre = torch.tensor((cx, cy, cz))

    # get rotation displacement
    displacement = torch.cat((xc[..., None], yc[..., None], zc[..., None]), dim=-1)
    displacement = s * torch.matmul(R, displacement[..., None])[..., 0]
    displacement = centre[None, None, None, :] + displacement

    displacement = displacement[cropx:cropx+crop_size[0]+1,
                                cropy:cropy+crop_size[1]+1,
                                cropz:cropz+crop_size[2]+1, :]

    left = slice(0, -1)
    right = slice(1, None)

    # get jacobian of displacement field using forward differences
    jacobian = torch.empty((crop_size[0], crop_size[1], crop_size[2], 3, 3))

    jacobian[..., 0] = displacement[right, left, left, :] - displacement[left, left, left, :]
    jacobian[..., 1] = displacement[left, right, left, :] - displacement[left, left, left, :]
    jacobian[..., 2] = displacement[left, left, right, :] - displacement[left, left, left, :]

    displacement = displacement[left, left, left]

    # rotation is U * Vh where U and V are the singular vector matrices
    U, Vh = torch.linalg.svd(jacobian)[slice(0, None, 2)]

    # use transpose as we're using displacement from target to source
    R_reorient = torch.transpose(torch.matmul(U, Vh), -1, -2)

    dti_def = resmple_dti(fa, v1, displacement, R_reorient)

    xx2 = displacement[..., 0]
    yy2 = displacement[..., 1]
    zz2 = displacement[..., 2]


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

    # c = c0 * wfz[..., None] + c1 * wcz[..., None]

    dti_def = torch.zeros(*v1.shape, device='cpu')

    dti_def[ok[..., None]] = c0 * wfz[..., None] + c1 * wcz[..., None]

    return dti_def
