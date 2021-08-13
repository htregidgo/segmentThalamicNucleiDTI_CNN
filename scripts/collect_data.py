import os
import glob
import dipy_routines
import shutil
import h5py
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from distutils.util import strtobool


# TODO: make shell comms verbose
# TODO: clean up .h5 parsing and automate
# TODO: move all dipy stuff to dipy_routines.py

def collect_all_data():
    data_header_dir = "training_data/"
    if not os.path.exists(data_header_dir):
        merge_data(data_header_dir)
    assert len(os.listdir(data_header_dir)) == 0, "no cases in the training folder"

    else:
        query = confirm_change("training directory already exists, do you want to refresh it?")
        if query:
            for filename in os.listdir(data_header_dir):
                print("deleting %s" %filename)
                file_path = os.path.join(data_header_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            shutil.rmtree(data_header_dir)
            merge_data(data_header_dir)

    print("done collecting data...")


def topup_data():
    # assuming that header path already exists
    data_header_dir = "HCP_training_data/"
    if not os.path.exists(data_header_dir):
        raise FileNotFoundError

    print("setting topup paths...")
    path_list, case_list = set_dwi_paths()
    assert len(case_list) == 0, "no cases!"

    print("unwrapping and moving topup dwi files...")
    dipy_routines.import_convert_save_dwi(path_list,case_list,data_header_dir,rm_b0=False)
    print("shutting down")



def merge_data(header_dir):
    print("creating training directory...")
    os.makedirs(header_dir)

    print("setting paths...")
    path_list, case_list = set_dwi_paths()

    print("unwrapping and moving dwi files...")
    dipy_routines.import_convert_save_dwi(path_list,case_list,header_dir,False)



def confirm_change(question):
    while True:
        full_question = "%s [y/n]: " % question
        answer = input(full_question.encode('utf-8')).lower()
        try:
            return strtobool(answer)
        except ValueError:
            print('%s is not a valid answer.' % answer)


def set_dwi_paths(is_training=False):
    base_path = "HCP_training_data/"
    inner_path = "/"
    case_names = ["mgh_1001","mgh_1002","mgh_1003"]
    #"mgh_1007","mgh_1008","mgh_1009","mgh_1010","mgh_1011","mgh_1012",
    #"mgh_1013","mgh_1014","mgh_1015","mgh_1016","mgh_1017","mgh_1018",
    #"mgh_1019","mgh_1020","mgh_1021","mgh_1022","mgh_1023","mgh_1024",
    #"mgh_1025","mgh_1026","mgh_1027","mgh_1028","mgh_1029","mgh_1030",
    #"mgh_1031","mgh_1032","mgh_1033","mgh_1034"]
    #case_names = ["w001","w002","w003","w004","w005","w006"]
    #case_names = ["w001","w002","w003","w004","w005","w006","w007","w008","w009","w010","w011","w012",
    #"w013","w014","w015","w016","w017","w018","w019","w020"]
    #case_names = ["w021","w022"]
    print("extracting ",len(case_names), " cases ")

    if is_training:
        return [base_path + e + inner_path for e in case_names], case_names
    else:
        return [base_path + e + "/" for e in case_names], case_names


def move_masks():
    header_dir = "training_data/"
    if not os.path.exists(header_dir):
        raise FileNotFoundError

    print("setting paths...")
    path_list, case_list = set_dwi_paths()

    print("moving masks...")
    dipy_routines.convert_nifti_masks(path_list,case_list,header_dir)


def labelconvert_call(data_header_dir):
    # assuming that header path already exists
    if not os.path.exists(data_header_dir):
        raise FileNotFoundError

    print("setting label paths...")
    pathlist, case_list = set_dwi_paths()
    label_name = "AAN_overlay"
    for path,case in zip(pathlist,case_list):
        print("beginning converting %s label..." %case)
        tkr_temp = path + "b0.nii.gz"
        flab = path + label_name + ".label"

        @TODO dirty but will fix later
        try:
            os.system("source fs") # assure freesurfer sourcing (mark's desktop)
            os.system("mri_label2vol --label " + flab
            + " --tkr-template " + tkr_temp
            + " --temp " + tkr_temp
            + " --identity"
            + " --o " + path + label_name + ".nii.gz")
            if os.system(os_cmd) != 0:
                raise Exception('shell comm does not exist')
        except:
            print("shell comm does not work")


def convert_all(is_test=True):
    # assuming that header path already exists
    data_header_dir = "HCP_training_data/"
    if not os.path.exists(data_header_dir):
        raise FileNotFoundError
        #os.makedirs(data_header_dir)

    print("converting label to NIFTI")
    labelconvert_call(data_header_dir)
    print("done converting labels!")

    print("setting label paths...")
    pathlist, case_list = set_dwi_paths()

    for path,case in zip(pathlist,case_list):
        print("beginning %s ..." %case)

        path_b0 = path + "b0.nii.gz"
        path_fa = path + "fa.nii.gz"
        #path_cfa = path + "cfa.nii.gz"
        path_md = path + "md.nii.gz"
        path_shannon = path + "shannon_entropy.nii.gz"
        if not is_test:
            path_label = path + "AAN_overlay.nii.gz"

        data_b0, affine_b0 = load_nifti(path_b0, return_img=False)
        data_b0_scaled = np.interp(data_b0, (data_b0.min(), data_b0.max()), (0.0, 255.0))
        data_fa, affine_fa = load_nifti(path_fa, return_img=False)
        data_fa_scaled = np.interp(data_fa, (data_fa.min(), data_fa.max()), (0.0, 255.0))
        data_md, affine_md = load_nifti(path_md, return_img=False)
        data_md_scaled = np.interp(data_md, (data_md.min(), data_md.max()), (0.0, 255.0))
        #data_shannon, affine_shannon = load_nifti(path_shannon, return_img=False)
        #data_shannon_scaled = np.interp(np.nan_to_num(data_shannon), (np.nan_to_num(data_shannon).min(), np.nan_to_num(data_shannon).max()), (0.0, 255.0))
        if not is_test:
            data_label, affine_label = load_nifti(path_label, return_img=False)

        print("done reading")

        cat_data = np.stack((data_fa_scaled,data_md_scaled),axis=0).astype(np.float32)
        #cat_data = data_fa
        print("shape of concatenated data is: ", cat_data.shape)


        hf = h5py.File(data_header_dir + case + '.h5', 'w')
        hf.create_dataset('raw', data=cat_data)
        if not is_test:
            hf.create_dataset('label', data=data_label)
        hf.close()
        print("done with: ", case)


def convert_single_case(is_test=True):
    # assuming that header path already exists
    data_header_dir = "EXC_test_data/"
    if not os.path.exists(data_header_dir):
        raise FileNotFoundError
        #os.makedirs(data_header_dir)

    path_fa = '../FA_registration/EXC020_masked_FA.nii.gz'
    path_md = '../FA_registration/EXC020_masked_MD.nii.gz'
    data_fa, affine_fa = load_nifti(path_fa, return_img=False)
    data_fa_scaled = np.interp(data_fa, (data_fa.min(), data_fa.max()), (0.0, 255.0))
    data_md, affine_md = load_nifti(path_md, return_img=False)
    data_md_scaled = np.interp(data_md, (data_md.min(), data_md.max()), (0.0, 255.0))
    #data_shannon, affine_shannon = load_nifti(path_shannon, return_img=False)
    #data_shannon_scaled = np.interp(np.nan_to_num(data_shannon), (np.nan_to_num(data_shannon).min(), np.nan_to_num(data_shannon).max()), (0.0, 255.0))
    if not is_test:
        data_label, affine_label = load_nifti(path_label, return_img=False)

    print("done reading")

    cat_data = np.stack((data_fa_scaled,data_md_scaled),axis=0).astype(np.float32)
    #cat_data = data_fa
    print("shape of concatenated data is: ", cat_data.shape)


    hf = h5py.File(data_header_dir + 'EXC020.h5', 'w')
    hf.create_dataset('raw', data=cat_data)
    if not is_test:
        hf.create_dataset('label', data=data_label)
    hf.close()
    print("done")

def readhd5(path_raw,path_predict,output_raw,output_predict):
    hf_predict = h5py.File(path_predict, 'r')
    print(list(hf_predict.keys()))
    data_prediction = np.array(hf_predict['predictions'])

    hf_raw = h5py.File(path_raw, 'r')
    data_raw = np.array(hf_raw['raw'])

    save_nifti(output_predict, data_prediction[0,:,:,:].astype(np.float32), affine=np.eye(4))
    save_nifti(output_raw, data_raw[0,:,:,:].astype(np.float32), affine=np.eye(4))
