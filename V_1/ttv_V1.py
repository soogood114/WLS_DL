import numpy as np
import os, shutil
from datetime import datetime
import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from tqdm import tqdm as tqdm
from torchvision import transforms, utils
from torch.utils import tensorboard as tensorboard
import torch.optim.lr_scheduler as lr_scheduler

import Data.exr as exr
import Data.other_tools as other_tools

import Data.normalization as norm
import Data.design_matrix as design
import Feed_and_Loss.feed_transform as FT
import Feed_and_Loss.dataset as dataset
import Feed_and_Loss.loss as my_loss

import Models.net_op as net_op
import Models.models_v1 as models_v1
import Models.models_v2 as models_v2
import Models.models_KPCN as models_KPCN
import Models.NGPT_models as models_NGPT

import Models.model_pipelines_v1 as model_pipelines_v1


def train_test_model_v1(params, train_input_buffer, train_ref_buffer, test_input_buffer, test_ref_buffer,
                        TEST_MODE, RE_TRAIN, IMG_BY_IMG):
    """
    입력 구성: params, input_buffer = norm. of [color, g-buffer], ref_buffer = norm. of color
    순서: initial setting -> data transformation function setting -> data load -> train and test
    특징: PR 버전의 ttv를 그대로 활용.
    """

    """INITIAL SETTING"""
    # GPU index setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # _, H, W, _ = test_input_buffer.shape

    # channel index
    ch_col_start = 0
    if params["color_merge"]:
        ch_col_end = 3
    else:
        ch_col_end = 3

    if params['fixing_random_seed']:
        np.random.seed(0)

    ch_albedo_start = ch_col_end
    ch_albedo_end = ch_albedo_start + 3
    ch_depth_start = ch_albedo_end
    ch_depth_end = ch_depth_start + 1
    ch_normal_start = ch_depth_end
    ch_normal_end = ch_normal_start + 3

    # for test or retraining
    parameter_pth = params['trained_model_folder_pth'] + "/parameters/" + params['trained_parameter_name']

    if not TEST_MODE:

        """SETTING DATA LOAD AND CORRESPONDING TRANSFORMS FOR TRAINING"""
        if IMG_BY_IMG:
            buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal']

            if params['img_by_img_type'] == "h5":
                # h5py
                transform = transforms.Compose([
                    FT.IBI_RandomCrop_np(params['patch_size'], buffer_list),
                    FT.IBI_RandomFlip_np(params['multi_crop'], buffer_list),
                    FT.IBI_PermuteColor_np(params['multi_crop'], buffer_list),
                    FT.IBI_Normalize_Concat_np_v1(params['multi_crop'], buffer_list),
                    FT.ToTensor(params['multi_crop'])
                ])
                train_data = dataset.Supervised_dataset_img_by_img_np(train_input_buffer, train_ref_buffer,
                                                                          buffer_list,
                                                                          train=True, transform=transform)
            else:
                # torch data
                transform = transforms.Compose([
                    FT.IBI_RandomCrop_tensor(params['patch_size'], buffer_list),
                    FT.IBI_RandomFlip_tensor(params['multi_crop'], buffer_list),
                    FT.IBI_PermuteColor_tensor(params['multi_crop'], buffer_list),
                    FT.IBI_Normalize_Concat_tensor_v1(params['multi_crop'], buffer_list)
                ])
                train_data = dataset.Supervised_dataset_img_by_img_tensor(train_input_buffer, train_ref_buffer,
                                                                          buffer_list,
                                                                          train=True, transform=transform)

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True,
                                                       num_workers=params['num_workers']) # params['batch_size']
        else:
            # define transform op
            transform = transforms.Compose([
                FT.RandomCrop(params['patch_size']),
                FT.RandomFlip(multi_crop=params['multi_crop']),
                FT.PermuteColor(multi_crop=params['multi_crop']),
                FT.ToTensor(multi_crop=False)
            ])
            # train data loader
            train_data = dataset.Supervised_dataset(train_input_buffer, train_ref_buffer,
                                                    train=True, transform=transform)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)

        """SAVING THE TENSORBOARD"""
        out_tensorboard_folder_name = params["saving_sub_folder_name"] + "/tensorboards"
        if not os.path.exists(out_tensorboard_folder_name):
            os.mkdir(out_tensorboard_folder_name)
        writer = tensorboard.SummaryWriter(out_tensorboard_folder_name)

        "PARAMETER LOAD FOR RETRAIN"
        if RE_TRAIN:
            saved_torch_para = torch.load(parameter_pth)
            start_epoch = saved_torch_para['epoch']
        else:
            saved_torch_para = None
            start_epoch = 0

        end_epochs = params["epochs"]

    else:
        "PARAMETER LOAD FOR TEST"
        saved_torch_para = torch.load(parameter_pth)
        start_epoch = 0
        end_epochs = 0


    """LOSS SETTING"""
    if params['loss_type'] == 'l2':
        loss_fn = torch.nn.MSELoss()
    elif params['loss_type'] == 'l1':
        loss_fn = torch.nn.L1Loss()
    elif params['loss_type'] == 'custom_loss_v1':
        loss_fn = my_loss.my_custom_loss_v1(params['loss_weights'], params['SMAP_loss'], params['reinhard_loss'])
    else:
        print("unknown loss!")
        return

    """NETWORK INITIALIZATION"""
    saving_code_folder_name = params["saving_sub_folder_name"] + "/net_code"
    if not os.path.exists(saving_code_folder_name):
        os.mkdir(saving_code_folder_name)

    if params['network_name'] == "WLS_net_v1":
        mynet = models_v1.WLS_net_v1(params, loss_fn, ch_in=10, kernel_size=3, n_layers=50, length_p_kernel=21,
                                     epsilon=0.001,
                                     pad_mode=0, loss_type=0, kernel_accum=True, norm_in_window=False,
                                     is_resnet=True).train().to(device)
        shutil.copy("../Models/models_v1.py", saving_code_folder_name + "/saved_models_v1.py")

    elif params['network_name'] == "KPCN_v1":
        mynet = models_KPCN.KPCN_v1(params, loss_fn, ch_in=10, kernel_size=3, n_layers=50, length_p_kernel=21,
                                  no_soft_max=False,
                                  pad_mode=0, is_resnet=True).train().to(device)
        shutil.copy("../Models/models_KPCN.py", saving_code_folder_name + "/saved_models_KPCN.py")

    elif params['network_name'] == "WLS_net_FG_v1":
        if params['network_index'] == 0:
            print("WLS_net_FG_v1 and FG")
            mynet = models_v2.WLS_net_FG_v1(params, loss_fn, ch_in=10, kernel_size=3, n_layers=50, length_p_kernel=21,
                                            epsilon=0.0001,
                                            pad_mode=0, loss_type=0, kernel_accum=False, norm_in_window=True,
                                            is_resnet=True, FG_mode=1).train().to(device)
        else:
            # g-buffer denoisor
            print("WLS_net_FG_v1 and KPCN for b-buffer denoising")
            mynet = models_v2.WLS_net_FG_v1(params, loss_fn, ch_in=10, kernel_size=3, n_layers=50, length_p_kernel=21,
                                            epsilon=0.0001,
                                            pad_mode=0, loss_type=0, kernel_accum=False, norm_in_window=False,
                                            is_resnet=True, FG_mode=2).train().to(device)
        shutil.copy("../Models/models_v2.py", saving_code_folder_name + "/saved_models_v2.py")

    elif params['network_name'] == "WLS_net_FG_v2":
        print("WLS_net_FG_v2")
        # FG mode > 0 ---> on, FG mode < 0 ---> off
        mynet = models_v2.WLS_net_FG_v2(params, loss_fn, ch_in=10, kernel_size=3, n_layers=50, length_p_kernel=21, epsilon=0.00001,
                 pad_mode=0, loss_type=0, kernel_accum=False, norm_in_window=True, is_resnet=True, FG_mode=2,
                 soft_max_W=True, resi_train=False, g_buff_list=[True, True, True]).train().to(device)


    # re train or test mode
    if RE_TRAIN or TEST_MODE:
        "old and new"
        # mynet.load_state_dict(saved_torch_para['model_state_dict'])
        mynet = saved_torch_para['model']


    """SET LOSS AND OPTIMIZATION"""
    optimizer = optim.Adam(mynet.parameters(), lr=params['lr'])
    if RE_TRAIN or TEST_MODE:
        optimizer.load_state_dict(saved_torch_para['optimizer_state_dict'])


    """TRAIN NETWORK"""
    with tqdm(range(start_epoch, end_epochs), leave=True) as tnr:
        tnr.set_postfix(epoch=0, loss=-1.)

        for epoch in tnr:

            one_epoch_loss = 0.0
            num_iter_for_one_epoch = 0

            for data in train_loader:
                optimizer.zero_grad()

                "주의) 모든 data는 float16임. 그래서 netwrork와 맞추기 위해 float32로 변경"
                x = data['input'].cuda().to(torch.float32)
                y = data['target'].cuda().to(torch.float32)

                if (epoch + 1) % params["val_patches_saving_epoch"] == 0:
                    full_res_out = True
                else:
                    full_res_out = False

                if params['network_name'] == "WLS_net_v1":
                    y_pred, current_loss = mynet(x, y, False, full_res_out)
                    # current_loss = loss_fn(y_pred, y)

                elif params['network_name'] == "KPCN_v1":
                    y_pred, current_loss = mynet(x, y, only_img_out=False)

                elif params['network_name'] == "WLS_net_FG_v1" or params['network_name'] == "WLS_net_FG_v2":
                    y_pred, current_loss = mynet(x, y, False, full_res_out)
                    # current_loss = loss_fn(y_pred, y)

                current_loss.backward()
                optimizer.step()

                # 하나의 배치가 끝날 때 마다의 current loss를 보여줌
                tnr.set_postfix(epoch=epoch, loss=current_loss.item())

                one_epoch_loss += current_loss.data.item()
                num_iter_for_one_epoch += 1

            one_epoch_loss /= num_iter_for_one_epoch
            writer.add_scalar('training loss', one_epoch_loss, epoch)

            "PARAMETER SAVING"
            if (epoch + 1) % params['para_saving_epoch'] == 0:
                out_para_folder_name = params["saving_sub_folder_name"] + "/parameters"
                if not os.path.exists(out_para_folder_name):
                    os.mkdir(out_para_folder_name)
                torch.save(mynet.state_dict(), out_para_folder_name + "/latest_parameter")

                torch.save({
                    'epoch': epoch,
                    # 'model_state_dict': mynet.state_dict(),
                    'model': mynet,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_loss
                }, out_para_folder_name + "/latest_parameter")

                # torch.load(out_para_folder_name + "/latest_parameter")

            "INTERMEDIATE RESULTING PATCH SAVING"
            if (epoch + 1) % params["val_patches_saving_epoch"] == 0:
                inter_patch_folder_name = params["saving_sub_folder_name"] + "/val_patches"
                if not os.path.exists(inter_patch_folder_name):
                    os.mkdir(inter_patch_folder_name)

                x_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(x)
                y_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(y)
                y_pred_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(y_pred)

                for l in range(x_np_saving.shape[0]):
                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_color_in.exr",
                              x_np_saving[l, :, :, ch_col_start:ch_col_end])

                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_albedo.exr",
                              x_np_saving[l, :, :, ch_albedo_start:ch_albedo_end])
                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_depth.exr",
                              x_np_saving[l, :, :, ch_depth_start:ch_depth_end])
                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_normal.exr",
                              x_np_saving[l, :, :, ch_normal_start:ch_normal_end])

                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_color_out.exr",
                              y_pred_np_saving[l, :, :, 0:3])

                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_color_ref.exr",
                              y_np_saving[l, :, :, 0:3])
        # writer.close()


    """SETTING DATA LOAD FOR TEST"""
    if IMG_BY_IMG:
        if params['img_by_img_type'] == "h5":
            # h5py
            transform_img = transforms.Compose([FT.IBI_Normalize_Concat_np_v1(multi_crop=False),
                                                FT.ToTensor(False)])  # targeting for image
            # test data loader
            test_data = dataset.Supervised_dataset_img_by_img_np(test_input_buffer, test_ref_buffer, buffer_list,
                                                   train=False, transform=transform_img)
        else:
            # torch data
            transform_img = transforms.Compose(
                [FT.IBI_Normalize_Concat_tensor_v1(multi_crop=False)])  # targeting for image
            # test data loader
            test_data = dataset.Supervised_dataset_img_by_img_tensor(test_input_buffer, test_ref_buffer, buffer_list,
                                                                     train=False, transform=transform_img)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    else:
        transform_img = transforms.Compose([FT.ToTensor(multi_crop=False)])  # targeting for image

        # test data loader
        test_data = dataset.Supervised_dataset(test_input_buffer, test_ref_buffer,
                                               train=False, transform=transform_img)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    """VALIDATE NETWORK"""
    with torch.no_grad():
        mynet.eval()

        out_folder_name = params["saving_sub_folder_name"] + "/test_imgs"
        if not os.path.exists(out_folder_name):
            os.mkdir(out_folder_name)

        rmse_saving_pth = out_folder_name + "/rmse_list.txt"
        f = open(rmse_saving_pth, 'w')

        image_index = 0

        f.write(str(params["saving_folder_name"]))
        f.write("\n")

        for data in test_loader:
            x = data['input'].cuda()  #.to(torch.float32)
            y = data['target'].cuda()  #.to(torch.float32)

            if params['network_name'] == "WLS_net_v1":
                # y_pred = mynet(x, y, True)
                y_pred = mynet.test_chunkwise(x, chunk_size=200)

            elif params['network_name'] == "KPCN_v1":
                y_pred = mynet(x, y, only_img_out=True)
                # y_pred = mynet.test_chunkwise(x, chunk_size=200)

            elif params['network_name'] == "WLS_net_FG_v1" or params['network_name'] == "WLS_net_FG_v2":
                # y_pred = mynet(x, y, True)
                y_pred = mynet.test_chunkwise(x, chunk_size=200)


            "FROM TORCH TENSOR TO NUMPY TENSOR"
            x_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(x[:, :3, :, :])
            y_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(y)
            y_pred_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(y_pred)

            x_np_saving = x_np_saving[0]
            y_np_saving = y_np_saving[0]
            y_pred_np_saving = y_pred_np_saving[0]

            x_np_saving = norm.denormalization_signed_log(x_np_saving)
            y_np_saving = norm.denormalization_signed_log(y_np_saving)
            y_pred_np_saving = norm.denormalization_signed_log(y_pred_np_saving)

            rmse = other_tools.calcRelMSE(y_pred_np_saving, y_np_saving)
            rmse_str = str(image_index) + " image relMSE : " + str(rmse)
            f.write(rmse_str)
            f.write("\n")
            print(rmse_str)

            "SAVING THE RESULTING IMAGES"
            exr.write(out_folder_name + "/" + params['saving_file_name'] + "_" + str(image_index) + "_input.exr",
                      x_np_saving)
            exr.write(out_folder_name + "/" + params['saving_file_name'] + "_" + str(image_index) + "_gt.exr",
                      y_np_saving)
            exr.write(out_folder_name + "/" + params['saving_file_name'] + "_" + str(image_index) + "_result.exr",
                      y_pred_np_saving)

            image_index += 1
        f.close()



def test_for_one_exr_v1(params, input_buffer, ref_buffer):
    """
    임시로 test 함수를 만듦.
    나중에는 ttv에 이런 기능을 넣어 통합할 예정.
    이름대로 하나의 exr buffer를 갖고 test를 하는 함수임.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    """SETTING DATA LOAD FOR TEST"""
    transform_img = transforms.Compose([FT.ToTensor(multi_crop=False)])  # targeting for image

    # test data loader
    test_data = dataset.Supervised_dataset(input_buffer, ref_buffer,
                                           train=False, transform=transform_img)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)



    """LOSS SETTING"""
    if params['loss_type'] == 'l2':
        loss_fn = torch.nn.MSELoss()
    elif params['loss_type'] == 'l1':
        loss_fn = torch.nn.L1Loss()
    elif params['loss_type'] == 'custom_loss_v1':
        loss_fn = my_loss.my_custom_loss_v1(params['loss_weights'], params['SMAP_loss'], params['reinhard_loss'])
    else:
        print("unknown loss!")
        return

    if params['network_name'] == "WLS_net_v1":
        mynet = models_v1.WLS_net_v1(params, loss_fn, ch_in=10, kernel_size=3, n_layers=50, length_p_kernel=21,
                                     epsilon=0.001,
                                     pad_mode=0, loss_type=0, kernel_accum=True, norm_in_window=False,
                                     is_resnet=True).train().to(device)

    elif params['network_name'] == "KPCN_v1":
        mynet = models_KPCN.KPCN_v1(params, loss_fn, ch_in=10, kernel_size=3, n_layers=50, length_p_kernel=21,
                                  no_soft_max=False,
                                  pad_mode=0, is_resnet=True).train().to(device)

    parameter_pth = params['trained_model_folder_pth'] + "/parameters/" + params['trained_parameter_name']
    saved_torch_para = torch.load(parameter_pth)

    # mynet.load_state_dict(saved_torch_para['model_state_dict'])
    mynet = saved_torch_para['model']

    """VALIDATE NETWORK"""
    with torch.no_grad():
        mynet.eval()

        out_folder_name = params["saving_sub_folder_name"] + "/test_imgs"
        if not os.path.exists(out_folder_name):
            os.mkdir(out_folder_name)

        rmse_saving_pth = out_folder_name + "/rmse_list.txt"
        f = open(rmse_saving_pth, 'w')

        image_index = 0

        f.write(str(params["saving_folder_name"]))
        f.write("\n")

        for data in test_loader:
            x = data['input'].to(device).to(torch.float32)
            y = data['target'].to(device).to(torch.float32)

            if params['network_name'] == "WLS_net_v1":
                # y_pred = mynet(x, y, True)
                y_pred = mynet.test_chunkwise(x, chunk_size=200)

            elif params['network_name'] == "KPCN_v1":
                y_pred = mynet(x, y, only_img_out=True)
                # y_pred = mynet.test_chunkwise(x, chunk_size=200)

            elif params['network_name'] == "WLS_net_FG_v1":
                # y_pred = mynet(x, y, True)
                y_pred = mynet.test_chunkwise(x, chunk_size=200)

            "FROM TORCH TENSOR TO NUMPY TENSOR"
            x_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(x[:, :3, :, :])
            y_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(y)
            y_pred_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(y_pred)

            x_np_saving = x_np_saving[0]
            y_np_saving = y_np_saving[0]
            y_pred_np_saving = y_pred_np_saving[0]

            x_np_saving = norm.denormalization_signed_log(x_np_saving)
            y_np_saving = norm.denormalization_signed_log(y_np_saving)
            y_pred_np_saving = norm.denormalization_signed_log(y_pred_np_saving)

            rmse = other_tools.calcRelMSE(y_pred_np_saving, y_np_saving)
            rmse_str = str(image_index) + " image relMSE : " + str(rmse)
            f.write(rmse_str)
            f.write("\n")
            print(rmse_str)

            "SAVING THE RESULTING IMAGES"
            exr.write(out_folder_name + "/" + params['saving_file_name'] + "_" + str(image_index) + "_input.exr",
                      x_np_saving)
            exr.write(out_folder_name + "/" + params['saving_file_name'] + "_" + str(image_index) + "_gt.exr",
                      y_np_saving)
            exr.write(out_folder_name + "/" + params['saving_file_name'] + "_" + str(image_index) + "_result.exr",
                      y_pred_np_saving)

            image_index += 1
        f.close()




def train_test_model_v_pipeline(params, train_input_buffer, train_ref_buffer, test_input_buffer, test_ref_buffer,
                        TEST_MODE, RE_TRAIN, IMG_BY_IMG):
    """
    개요 : train_test_model_v1의 구조를 "pipeline" 형태의 모델에 맞게 변형을 함.
    특징: PR 버전의 ttv를 그대로 활용.
    """

    """INITIAL SETTING"""
    # GPU index setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # _, H, W, _ = test_input_buffer.shape

    if params['fixing_random_seed']:
        np.random.seed(0)

    # for test or retraining
    parameter_pth = params['trained_model_folder_pth'] + "/parameters/" + params['trained_parameter_name']

    # parameter_pth = "E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/WLS_DL/V_1/results/210504_tttt/05_04_11_01_12/parameters/latest_parameter"
    # RE_TRAIN = True


    buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal', 'albedoVariance', 'depthVariance',
                   'normalVariance', 'colorVariance']

    if not TEST_MODE:

        """SETTING DATA LOAD AND CORRESPONDING TRANSFORMS FOR TRAINING"""
        if IMG_BY_IMG:


            if params['img_by_img_type'] == "h5":
                # h5py
                transform = transforms.Compose([
                    FT.IBI_RandomCrop_np(params['patch_size'], buffer_list),
                    FT.IBI_RandomFlip_np(params['multi_crop'], buffer_list),
                    FT.IBI_PermuteColor_np(params['multi_crop'], buffer_list),
                    FT.IBI_Normalize_Concat_To_GPU_np_v_pipeline(params['multi_crop'], buffer_list)
                ])
                train_data = dataset.Supervised_dataset_img_by_img_np(train_input_buffer, train_ref_buffer,
                                                                          buffer_list,
                                                                          train=True, transform=transform)
            else:
                print("Must be img_by_img and h5 data type !!")
                return

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True,
                                                       num_workers=params['num_workers']) # params['batch_size']
        else:
            print("Must be img_by_img and h5 data type !!")
            return

        """SAVING THE TENSORBOARD"""
        out_tensorboard_folder_name = params["saving_sub_folder_name"] + "/tensorboards"
        if not os.path.exists(out_tensorboard_folder_name):
            os.mkdir(out_tensorboard_folder_name)
        writer = tensorboard.SummaryWriter(out_tensorboard_folder_name)

        end_epochs = params["epochs"]

    else:
        "PARAMETER LOAD FOR TEST"
        start_epoch = 0
        end_epochs = 0

    """LOSS SETTING"""
    if params['loss_type'] == 'l2':
        loss_fn = torch.nn.MSELoss()
    elif params['loss_type'] == 'l1':
        loss_fn = torch.nn.L1Loss()
    elif params['loss_type'] == 'custom_loss_v1':
        loss_fn = my_loss.my_custom_loss_v1(params['loss_weights'], params['SMAP_loss'], params['reinhard_loss'])
    else:
        print("unknown loss!")
        return

    """NETWORK INITIALIZATION"""
    saving_code_folder_name = params["saving_sub_folder_name"] + "/net_code"
    if not os.path.exists(saving_code_folder_name):
        os.mkdir(saving_code_folder_name)

    shutil.copy("./ttv_V1.py", saving_code_folder_name + "/saved_ttv_V1.py.py")

    if params['model_pipeline_name'] == "WLS_net_Gde_FG_Pipeline_v1":
        print("WLS_net_Gde_FG_Pipeline_v1")
        # loss
        g_buff_loss_fn = my_loss.my_custom_loss_v1([0.3, 0.7], params['SMAP_loss'], params['reinhard_loss'])
        # g_buff_loss_fns = {
        #     'albedo_loss':my_loss.my_custom_loss_v1([0.7, 0.3], params['SMAP_loss'], params['reinhard_loss']),
        #     'depth_loss':my_loss.my_custom_loss_v1([1.0, 0.0], params['SMAP_loss'], params['reinhard_loss']),
        #     'normal_loss':my_loss.my_custom_loss_v1([0.7, 0.3], params['SMAP_loss'], params['reinhard_loss']),
        # }
        base_loss_fn = loss_fn

        # sub net
        g_buff_net = models_KPCN.KPCN_for_FG_v1(params, ch_in=(7 + 3), kernel_size=3, n_layers=10, length_p_kernel=9,
                                            no_soft_max=True, pad_mode=1, is_resnet=True, resi_train=False,
                                                use_W_var=False).train().to(device)

        # g_buff_net = models_NGPT.Back_bone_NGPT_v2(params, channels_in=(7 + 3),
        #                                            out_dim=7, padding_mode='reflect').train().to(device)

        base_net = models_v2.WLS_net_FG_v2(params, base_loss_fn, ch_in=11, kernel_size=3, n_layers=50,
                length_p_kernel=21, epsilon=0.00001,
                 pad_mode=1, loss_type=0, kernel_accum=False, norm_in_window=True, is_resnet=True, FG_mode=-1,
                 soft_max_W=True, resi_train=False, g_buff_list=[True, True, True]).train().to(device)

        model_pipeline = model_pipelines_v1.WLS_net_Gde_FG_Pipeline_v1(params, g_buff_net, base_net,
                                                                       g_buff_loss_fn, base_loss_fn, 2000)

        shutil.copy("../Models/models_v2.py", saving_code_folder_name + "/saved_models_v2.py")
        shutil.copy("../Models/models_KPCN.py", saving_code_folder_name + "/saved_models_KPCN.py")
        shutil.copy("../Models/model_pipelines_v1.py", saving_code_folder_name + "/saved_model_pipelines_v1.py")

    elif params['model_pipeline_name'] == "WLS_net_FG_Pipeline_v1":
        G_BUFF_VAR = True
        COLOR_VAR = True
        ch_in = 10

        if G_BUFF_VAR:
            ch_in += 3

        if COLOR_VAR:
            ch_in += 1

        base_net = models_v2.WLS_net_FG_v2(params, loss_fn, ch_in=ch_in, kernel_size=3, n_layers=50,
                                           length_p_kernel=21, epsilon=0.00001,
                                           pad_mode=1, loss_type=0, kernel_accum=False, norm_in_window=True,
                                           is_resnet=True, FG_mode=-1, soft_max_W=True, resi_train=False,
                                           g_buff_list=[True, True, True],
                                           use_g_buff_var=G_BUFF_VAR, use_color_var=COLOR_VAR).train().to(device)

        model_pipeline = model_pipelines_v1.WLS_net_FG_Pipeline_v1(params, base_net, loss_fn, G_BUFF_VAR, COLOR_VAR)

        shutil.copy("../Models/models_v2.py", saving_code_folder_name + "/saved_models_v2.py")
        shutil.copy("../Models/models_KPCN.py", saving_code_folder_name + "/saved_models_KPCN.py")
        shutil.copy("../Models/model_pipelines_v1.py", saving_code_folder_name + "/saved_model_pipelines_v1.py")


    else:
        print("no right name for model pipeline!")
        return


    "MODEL AND OPTIM PARAMETER LOAD FOR RETRAIN"
    if RE_TRAIN:
        # start_epoch = model_pipeline.load_parameter(parameter_pth)

        saved_torch_para = torch.load(parameter_pth)
        model_pipeline = saved_torch_para['pipeline']
        start_epoch = saved_torch_para['epoch']


    else:
        start_epoch = 0

    """TRAIN NETWORK"""
    with tqdm(range(start_epoch, end_epochs), leave=True) as tnr:
        tnr.set_postfix(epoch=0, loss=-1.)

        for epoch in tnr:
            one_epoch_loss_g_buff = 0.0
            one_epoch_loss_base = 0.0

            num_iter_for_one_epoch = 0

            for data in train_loader:
                if (epoch + 1) % params["val_patches_saving_epoch"] == 0:
                    full_res_out = True
                else:
                    full_res_out = False

                model_pipeline.data_feed_and_optimization(epoch, data, False, full_res_out)

                if params['model_pipeline_name'] == "WLS_net_Gde_FG_Pipeline_v1":
                    current_loss_g_buff_net, current_loss_base_net = model_pipeline.get_current_losses()

                elif params['model_pipeline_name'] == "WLS_net_FG_Pipeline_v1":
                    current_loss_g_buff_net = 0
                    current_loss_base_net = model_pipeline.get_current_losses()

                tnr.set_postfix(epoch=epoch, loss_of_g_buff_net=current_loss_g_buff_net,
                                loss_of_base_net=current_loss_base_net)

                one_epoch_loss_g_buff += current_loss_g_buff_net
                one_epoch_loss_base += current_loss_base_net

                num_iter_for_one_epoch += 1

            one_epoch_loss_g_buff /= num_iter_for_one_epoch
            one_epoch_loss_base /= num_iter_for_one_epoch

            writer.add_scalar('training g_buff_loss', one_epoch_loss_g_buff, epoch)
            writer.add_scalar('training_base_loss', one_epoch_loss_base, epoch)


            "PARAMETER SAVING"
            # model_pipeline.save_parameter(epoch)
            if (epoch + 1) % params['para_saving_epoch'] == 0:
                out_para_folder_name = params["saving_sub_folder_name"] + "/parameters"
                if not os.path.exists(out_para_folder_name):
                    os.mkdir(out_para_folder_name)

                torch.save({
                    'epoch': epoch,
                    # 'model_state_dict': mynet.state_dict(),
                    'pipeline': model_pipeline,

                }, out_para_folder_name + "/latest_parameter")


            "INTERMEDIATE RESULTING PATCH SAVING"
            model_pipeline.save_inter_results(epoch, data)

    """SETTING DATA LOAD FOR TEST"""
    if IMG_BY_IMG:
        if params['img_by_img_type'] == "h5":
            # h5py
            transform_img = transforms.Compose([FT.IBI_Normalize_Concat_To_GPU_np_v_pipeline(params['multi_crop'], buffer_list)])  # targeting for image
            # test data loader
            test_data = dataset.Supervised_dataset_img_by_img_np(test_input_buffer, test_ref_buffer, buffer_list,
                                                   train=False, transform=transform_img)
        else:
            print("Must be img_by_img and h5 data type !!")
            return

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    else:
        print("Must be img_by_img and h5 data type !!")
        return


    """VALIDATE NETWORK"""
    with torch.no_grad():
        out_folder_name = params["saving_sub_folder_name"] + "/test_imgs"
        if not os.path.exists(out_folder_name):
            os.mkdir(out_folder_name)

        rmse_saving_pth = out_folder_name + "/rmse_list.txt"
        f = open(rmse_saving_pth, 'w')

        image_index = 0

        f.write(str(params["saving_folder_name"]))
        f.write("\n")

        for data in test_loader:

            model_pipeline.test(data, chunk_size=200)

            "FROM TORCH TENSOR TO NUMPY TENSOR"
            model_pipeline.save_final_results(data, out_folder_name, f, image_index)
            image_index += 1

        f.close()

