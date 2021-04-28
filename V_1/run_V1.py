import os
import numpy as np
import json
from datetime import datetime

import Data.load as load
import V_1.ttv_V1 as ttv

params_default = {
    # 1. Mode selection
    "mini_batch": False,  # mini batch

    # test 용도
    'trained_model_TEST': False,  # test mode
    'trained_model_folder_pth': "F:/DH/history/4.09/210412_WLS_net_FG_v1(col_g-buffer)_norm_epoch_1000/04_12_11_03_05",
    # ./results/210407_testtest_6/04_07_15_09_02
    'trained_parameter_name': "latest_parameter",

    'trained_model_RETRAIN': False,

    # 2. Data load
    'color_merge': True,  # diffuse + specular
    'fixing_random_seed': True,

    # 새로 추가 !!
    'img_by_img': True,
    'img_by_img_type': "torch",  # h5 : h5, torch : torch.data

    # 3. Design matrix
    "grid_order": 1,

    # 4. Image and batch size & iterations
    'batch_size': 8,  # 32  배치사이즈가 지나치게 크게 되면 자동으로 잘라준다.
    'epochs': 100,
    'patch_size': 128,  # 200
    'multi_crop': False,

    # 5. Normalization
    'mue_tr': False,

    # 6. Loss configuration
    'loss_type': 'l1',  # l1, l2, custom_loss_v1
    'loss_weights': [0.8, 0.2],  # colorloss + gradientloss
    'SMAP_loss': False,  # for custom_loss_v1
    'reinhard_loss': False,  # for custom_loss_v1

    # 7. Optimization
    'optim': 'adam',
    'lr': 0.0001,  # default : 0.0001

    # 8. Saving period
    "para_saving_epoch": 50,  # 50
    "loss_saving_epoch": 10,  # 이건 지금 쓰이고 있질 않음. epoch마다 저장  10
    "val_patches_saving_epoch": 1,  # 50

    # 9. Index setting for run and network functions
    'run_index': 0,
    'network_name': "WLS_net_FG_v2",  # WLS_net_v1, KPCN_v1, WLS_net_FG_v1
    'network_index': 0,

    # 10. Saving folder setting
    'saving_folder_name': "210427_tttt",
    # 210326_model_stack_v2_epoch_2k_W_half_nonorm_smape
    # 210331_model_stack_v2_patch_size_100_mini_unfolded_no_order

    'saving_sub_folder_name': None,  # if None, it will replaced to time, day string

    'saving_file_name': "210427_tttt",

}


def data_load_and_run(params=None, gpu_id=1):
    if params is None:
        params = params_default

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if params['trained_model_TEST']:
        params['saving_folder_name'] = "TEST_" + params['saving_folder_name']

    """########################  CREATE SAVING FOLDER ########################"""
    # must make the root of saving results if it is not exist
    root_saving_pth = "./results"
    if not os.path.exists(root_saving_pth):
        os.mkdir(root_saving_pth)

    # make the new saving folder as intended
    saving_folder_pth = root_saving_pth + "/" + params["saving_folder_name"]
    if not os.path.exists(saving_folder_pth):
        os.mkdir(saving_folder_pth)

    # for multiple experiments with same params, time_folder is needed.
    if params["saving_sub_folder_name"] == None:
        params["saving_sub_folder_name"] = saving_folder_pth + "/" + str(datetime.today().strftime("%m_%d_%H_%M_%S"))

    if not os.path.exists(params["saving_sub_folder_name"]):
        os.mkdir(params["saving_sub_folder_name"])

    # saving folder -> tile- > tilme foldoer -> setting, img, tensorboard ect.

    """########################  SAVE THE SETTING ###########################"""
    if not params['trained_model_TEST']:
        params['trained_model_folder_pth'] = params["saving_sub_folder_name"]

    # saving folder -> tile- > tilme foldoer -> setting, img, tensorboard ect.
    json_saving_folder_pth = params["saving_sub_folder_name"] + "/settings"
    if not os.path.exists(json_saving_folder_pth):
        os.mkdir(json_saving_folder_pth)

    json_saving_pth = json_saving_folder_pth + "/setting_params.json"
    with open(json_saving_pth, 'w') as fp:
        json.dump(params, fp)

    """####################  GET & NORMALIZE INPUT, DESIGN, GT ####################"""
    if params['img_by_img']:
        print("THIS TRAINING IS DONE BY TORCH DATA (IMG BY IMG) !")

        dataset_dirs = "E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/img_by_img/"

        if params['img_by_img_type'] == "h5":
            dataset_dirs += "h5py_data/"
        else:
            dataset_dirs += "torch_data/"

        train_dirs = dataset_dirs + "1. train/"
        train_scenes = ['bathroom2', 'car2', 'classroom', 'house', 'room2', 'room3', 'spaceship', 'staircase']

        test_dirs = dataset_dirs + "2. test/"
        test_scenes = ['bathroom2', 'car', 'kitchen', 'living-room', 'living-room-3', 'veach-ajar',
                       'curly-hair', 'staircase2', 'glass-of-water', 'teapot-full']

        buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal', 'diffuseVariance',
                                              'specularVariance', 'albedoVariance', 'depthVariance', 'normalVariance']

        train_input_path, train_ref_path = load.get_all_pth_from_tungsten_torch_data(train_dirs, train_scenes,
                                                                                         buffer_list,
                                                                                         "00128spp", "08192spp",
                                                                                         params['mini_batch'],
                                                                                     params['img_by_img_type']
                                                                                     )

        test_input_path, test_ref_path = load.get_all_pth_from_tungsten_torch_data(test_dirs, test_scenes,
                                                                                     buffer_list,
                                                                                     "100spp", "64kspp",
                                                                                     False,
                                                                                   params['img_by_img_type']
                                                                                   )

        ttv.train_test_model_v1(params, train_input_path, train_ref_path, test_input_path, test_ref_path,
                                TEST_MODE=params['trained_model_TEST'], RE_TRAIN=params['trained_model_RETRAIN'],
                                IMG_BY_IMG=params['img_by_img'])


    else:
        print("THIS TRAINING IS DONE BY NUMPY DATASET (WHOLE DATA) !")

        dataset_dirs = "E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/npy/"

        if params["mini_batch"]:
            train_dirs = dataset_dirs + "3. mini_batch/"
        else:
            train_dirs = dataset_dirs + "1. train/"

        test_dirs = dataset_dirs + "2. test/"

        buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal']

        if params['trained_model_TEST']:
            # test mode does not need training datasets
            train_input_buffer, train_ref_buffer = [None, None]
            params['saving_folder_name'] = "TEST_" + params['saving_folder_name']

        else:
            train_input_buffer, train_ref_buffer = load.get_npy_tungsten_and_normalize_v1(train_dirs, buffer_list,
                                                                                          params['color_merge'])

        test_input_buffer, test_ref_buffer = load.get_npy_tungsten_and_normalize_v1(test_dirs, buffer_list,
                                                                                    params['color_merge'])

        """####################  TRAIN OR TEST MODE ####################"""
        ttv.train_test_model_v1(params, train_input_buffer, train_ref_buffer, test_input_buffer, test_ref_buffer,
                                TEST_MODE=params['trained_model_TEST'], RE_TRAIN=params['trained_model_RETRAIN'],
                                IMG_BY_IMG=params['img_by_img'])



def one_exr_load_and_test(test_params, input_pth, ref_pth, BUFFER, gpu_id=1):
    root_saving_pth = "./results"
    if not os.path.exists(root_saving_pth):
        os.mkdir(root_saving_pth)

    # make the new saving folder as intended
    saving_folder_pth = root_saving_pth + "/" + test_params["saving_folder_name"]
    if not os.path.exists(saving_folder_pth):
        os.mkdir(saving_folder_pth)

    if test_params["saving_sub_folder_name"] == None:
        test_params["saving_sub_folder_name"] = saving_folder_pth + "/" + str(
            datetime.today().strftime("%m_%d_%H_%M_%S"))
    else:
        test_params["saving_sub_folder_name"] = saving_folder_pth + "/" + test_params["saving_sub_folder_name"]

    if not os.path.exists(test_params["saving_sub_folder_name"]):
        os.mkdir(test_params["saving_sub_folder_name"])

    input_buffer, ref_buffer = load.load_normalize_one_exr_for_test(input_pth, ref_pth, BUFFER, load_dtype="FLOAT")

    ttv.test_for_one_exr(test_params, input_buffer, ref_buffer)
