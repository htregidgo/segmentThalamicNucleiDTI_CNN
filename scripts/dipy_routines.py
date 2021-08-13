import dipy
import numpy as np
import os
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.shore import ShoreModel
from dipy.data import get_fnames, get_sphere
from dipy.reconst.dti import fractional_anisotropy, mean_diffusivity, color_fa
from scipy.stats import entropy
import numpy.ma as ma
import dipy.reconst.dki as dki

# TODO: automate extraction of

### numerical functions

def shannon_entropy(odf):
    return entropy(odf, base=2)

def perplexity(odf):
    return 2**(entropy(odf, base=2))

def map_function_odf(odf_vol,f): # assumes odf field from 3d volume
    out = np.empty_like(odf_vol[:,:,:,0])

    for i in range(0,odf_vol.shape[0]):
        for j in range(0,odf_vol.shape[1]):
            for k in range(0,odf_vol.shape[2]):
                out[i,j,k] = f(odf_vol[i,j,k,:])
    return out

###

def fit_odf(data_roi,gtab):
    radial_order = 6
    zeta = 700
    lambdaN = 1e-8
    lambdaL = 1e-8
    asm = ShoreModel(gtab, radial_order=radial_order,
                     zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
    asmfit = asm.fit(data_roi)
    # which sphere - pick one of: * ‘symmetric362’ * ‘symmetric642’ * ‘symmetric724’ * ‘repulsion724’ * ‘repulsion100’ * ‘repulsion200’
    sphere = get_sphere('repulsion200')
    odf = asmfit.odf(sphere)
    print('odf.shape (%d, %d, %d, %d)' % odf.shape)

    return odf, sphere


def import_convert_save_dwi(pathlist,case_list,header,rm_b0,cook_ODF=False,cook_FA=True,cook_Kurt=True,cook_b0=True,b_mask=False):
    for path,case in zip(pathlist,case_list):
        print("beginning %s ..." %case)
        fdwi = path + "diff_preproc.nii.gz"
        data, affine = load_nifti(fdwi, return_img=False)
        if b_mask:
            brain_mask, affine_b = load_nifti(path + "nodif_brain_mask.nii.gz",return_img=False)
        else:
            brain_mask = np.ones_like(data[:,:,:,0])
        ## load in bvec and bval
        fbval = path + "bvals.txt"
        fbvec = path + "bvecs_moco_norm.txt"
        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
        gtab = gradient_table(bvals, bvecs)
        # save everything to bin files
        savepath = header+"/%s/" % case

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        if cook_b0:
            print("saving b0 volume...")
            if b_mask:
                data_b0 = ma.masked_array(data[:,:,:,0], mask=brain_mask)
            else:
                data_b0 = data[:,:,:,0]
            save_nifti(savepath+'b0.nii.gz', data_b0.astype(np.float32), affine)

        if cook_ODF: # wether you want to extract an ODF or not
            if rm_b0:
                # remove all but 1 b_0 image
                where_zeros = list(np.where(bvals==0))
                bvals_upd = np.delete(bvals,where_zeros)
                print(bvals_upd)
                bvecs_upd = np.delete(bvecs,where_zeros,axis=0)
                data_upd = np.delete(data,where_zeros,axis=3)
                gtab = gradient_table(bvals_upd, bvecs_upd)

                extract_ODF(savepath,data_upd,gtab)

            else:
                gtab = gradient_table(bvals, bvecs)
                extract_ODF(savepath,data,gtab,affine,brain_mask)

        if cook_Kurt:
            print("cooking Kurtosis model...")
            extract_Kurtosis(data,affine,gtab,savepath,brain_mask)
            print("done")

        if cook_FA:
            print("cooking FA volumes...")
            extract_MD_FA(data,affine,gtab,savepath,brain_mask)
            print("done")

        print("done cooking and transferring %s" %case)



def convert_nifti_masks(pathlist,case_list,header):
    for path,case in zip(pathlist,case_list):
        savepath = header+"/%s/" % case
        print("beginning converting %s mask..." %case)
        fdwi = path + "mri/pons.nii.gz"
        data, affine = load_nifti(fdwi, return_img=False)
        np.save(savepath+"pons_mask", data, allow_pickle=False, fix_imports=True)


def extract_MD_FA(data,affine,gtab,savepath,brain_mask,save_cfa=False):
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data)
    print('Computing anisotropy measures (FA, MD, RGB)')
    FA = fractional_anisotropy(tenfit.evals)
    MD = mean_diffusivity(tenfit.evals)

    FA = np.clip(FA, 0, 1)
    if save_cfa:
        CFA = color_fa(FA, tenfit.evecs)
        cfa_mask = np.repeat(arr[...,None],3,axis=3) #repeat mask into 4th channel dim
        print("dimensions of CFA image are: ", CFA.shape)

    print("dimensions of FA image are: ", FA.shape)
    print("dimensions of MD image are: ", MD.shape)
    print("saving FA images...")
    save_nifti(savepath+'fa.nii.gz', ma.masked_array(FA, mask=brain_mask).astype(np.float32), affine)
    save_nifti(savepath+'md.nii.gz', ma.masked_array(MD, mask=brain_mask).astype(np.float32), affine)
    if save_cfa:
        save_nifti(savepath+'cfa.nii.gz', ma.masked_array(CFA, mask=brain_mask).astype(np.float32), affine)
    print("done saving FA images")

def extract_Kurtosis(data,affine,gtab,savepath,brain_mask,save_cfa=False):
    dkimodel = dki.DiffusionKurtosisModel(gtab)
    dkifit = dkimodel.fit(data)
    print('Computing anisotropy measures (FA, MD, RGB)')
    FA = dkifit.fa
    MD = dkifit.md
    AD = dkifit.ad
    RD = dkifit.rd

    MK = dkifit.mk(0, 3)
    AK = dkifit.ak(0, 3)
    RK = dkifit.rk(0, 3)

    FA = np.clip(FA, 0, 1)
    if save_cfa:
        CFA = color_fa(FA, tenfit.evecs)
        cfa_mask = np.repeat(arr[...,None],3,axis=3) #repeat mask into 4th channel dim
        print("dimensions of CFA image are: ", CFA.shape)

    print("dimensions of FA image are: ", FA.shape)
    print("dimensions of MD image are: ", MD.shape)
    print("saving FA images...")
    save_nifti(savepath+'fa_k.nii.gz', ma.masked_array(FA, mask=brain_mask).astype(np.float32), affine)
    save_nifti(savepath+'md_k.nii.gz', ma.masked_array(MD, mask=brain_mask).astype(np.float32), affine)
    save_nifti(savepath+'ad_k.nii.gz', ma.masked_array(AD, mask=brain_mask).astype(np.float32), affine)
    save_nifti(savepath+'rd_k.nii.gz', ma.masked_array(RD, mask=brain_mask).astype(np.float32), affine)

    save_nifti(savepath+'mk.nii.gz', ma.masked_array(MK, mask=brain_mask).astype(np.float32), affine)
    save_nifti(savepath+'ak.nii.gz', ma.masked_array(AK, mask=brain_mask).astype(np.float32), affine)
    save_nifti(savepath+'rk.nii.gz', ma.masked_array(RK, mask=brain_mask).astype(np.float32), affine)

    if save_cfa:
        save_nifti(savepath+'cfa.nii.gz', ma.masked_array(CFA, mask=brain_mask).astype(np.float32), affine)
    print("done saving Kurtosis volumes")



def extract_ODF(savepath,data_upd,gtab,affine,brain_mask,save_odf=False,save_odf_map=True):
    print("fitting ODF (repulsion200) ...")
    odf, sphere = fit_odf(data_upd,gtab)
    if save_odf:
        np.save(savepath+"odf", odf, allow_pickle=False, fix_imports=True)
    if save_odf_map:
        save_nifti(savepath+'shannon_entropy.nii.gz', ma.masked_array(map_function_odf(odf,shannon_entropy), mask=brain_mask).astype(np.float32), affine)

    del odf, sphere



def process_single_dwi_nifti():
    base_f =  "../EXC020/diff/"
    base_fdwi = "../EXC020/diff/"
    data, affine = load_nifti(base_fdwi + "data.nii.gz", return_img=False)
    bvals, bvecs = read_bvals_bvecs(base_f + "dwi.bvals", base_f + "dwi.bvecs")
    gtab = gradient_table(bvals, bvecs)

    brain_mask, affine_b = load_nifti(base_f + "lowb_brain_mask.nii.gz",return_img=False)

    data_b0 = data[:,:,:,0]
    print("shape is: ",data_b0.shape)
    save_nifti(base_fdwi+'b0.nii.gz', ma.masked_array(data_b0, mask=brain_mask).astype(np.float32), affine)
    print("beginning fitting...")
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data)
    print('Computing anisotropy measures (FA, MD, RGB)')
    FA = fractional_anisotropy(tenfit.evals)
    MD = mean_diffusivity(tenfit.evals)

    FA = np.clip(FA, 0, 1)
    CFA = color_fa(FA, tenfit.evecs)
    print("dimensions of FA image are: ", FA.shape)
    print("dimensions of MD image are: ", MD.shape)
    print("dimensions of CFA image are: ", CFA.shape)
    print("saving FA images...")
    save_nifti(base_f+'fa.nii.gz', ma.masked_array(FA, mask=brain_mask).astype(np.float32), affine)
    save_nifti(base_f+'md.nii.gz', ma.masked_array(MD, mask=brain_mask).astype(np.float32), affine)
    print("done saving FA images")
