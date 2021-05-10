import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim
import os

import Data.other_tools as other_tools
import Data.exr as exr
import Data.normalization as norm


import Feed_and_Loss.loss as my_loss
import Models.resnet as resnet
import Models.NGPT_models as NGPT
import Models.models_KPCN as KPCN
import Models.models_v1 as models_v1
import Models.models_v2 as models_v2


class WLS_net_Gde_FG_Pipeline_v1(nn.Module):
    """
    파이프라인 개요
    - 처음으로 train, test, saving 등의 process 구현.
    - 따로 특수한 네트워크는 존재하지 않음. 다만, 이용할 뿐.
    - 그래서 이름을 Pipeline이라고 지음. (model.py와 차별점을 두기 위해서)

    네트워크 개요
    - WLS_net FG v2 구조를 base net로 채택.
    - KPCN_for_FG_v1 구조를 g_buffer denoiser로 채택.

    입력 개요
    - base net : color, g_buffer, color_var
    - g_buff_net : g_buffer, g_buffer_var


    주요 기능
    - G-buffer denoising network가 있음. 또한, pre_train 기능을 제공하려고 함.
    - 이전의 network와는 다르게 var 정보를 사용 가능하도록 할 예정.
    - Design matrix : buffer list를 통해서 feature 선택 가능하도록 할 것.
    - 큰 특징은 여기서 optimizer와 loss의 back propagation을 할 예정.
    - Gated kerneling의 형태를 구현을 하려고 함. (우선 순위가 낮음)
    - 이전에 사용한 network를 적극적으로 활용을 할 것.

    과정 정리
    - 먼저는 pretrain을 정해진 epoch에 따라서 시킨다.
    - 그런 다음에는 g-buff net과 basenet을 동시에 학습을 시키는 것으로 한다.

    """

    def __init__(self, params, g_buff_net, base_net, g_buff_loss, base_loss, iter_for_g_buff=100):
        super(WLS_net_Gde_FG_Pipeline_v1, self).__init__()


        self.params = params

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.iteration = params['epochs']

        self.iter_for_g_buff_denoiser = iter_for_g_buff

        self.iter_for_whole = self.iteration - self.iter_for_g_buff_denoiser  # 900


        "loss function"
        # loss를 다르게 두어도 되도록 함.
        self.loss_for_g_buff_de = g_buff_loss
        self.loss_for_base = base_loss


        "G-buffer denoiser"
        self.g_buff_net = g_buff_net

        "Base network"
        self.base_net = base_net

        "optimizer"
        self.optimizer_for_base_net = optim.Adam(self.base_net.parameters(), lr=params['lr'])
        self.optimizer_for_g_buff_net = optim.Adam(self.g_buff_net.parameters(), lr=params['lr'])

        "intermediate results"
        self.out_g_buff_net = 0
        self.out_base_net = 0

        self.current_loss_g_buff_net = 0
        self.current_loss_base_net = 0


    def data_feed_and_optimization(self, epoch, input, only_img_out=False, full_res_out=True):
        """
        epoch :
        input : B C_in H W
        design : B C_de H W

        특이하게 gt를 넣어서 loss까지 한방에 계산을 하도록 함.
        output : resulting image and loss

        """
        input['in_g_buff_net'] = input['in_g_buff_net'].to(self.device)
        input['in_color'] = input['in_color'].to(self.device)
        input['in_colorVar'] = input['in_colorVar'].to(self.device)
        input['ref_color'] = input['ref_color'].to(self.device)
        input['ref_g_buff_net'] = input['ref_g_buff_net'].to(self.device)

        "INITIAL SETTING"
        self.optimizer_for_g_buff_net.zero_grad()
        self.optimizer_for_base_net.zero_grad()


        # g_buff_net feed and optimization
        out_g_buff_net = self.g_buff_net(input['in_g_buff_net'])

        # res = out_g_buff_net != out_g_buff_net
        # res = res.type(torch.uint8)
        # res_np = other_tools.from_torch_tensor_img_to_full_res_numpy(res)
        # res_flag = torch.sum(res)
        #
        # aa = input['in_g_buff_net'][:, :7, :, :]
        # res2 = aa != aa
        # res2 = res.type(torch.uint8)
        # res2_np = other_tools.from_torch_tensor_img_to_full_res_numpy(res)
        # res2_flag = torch.sum(res)

        loss_g_buff_net = self.loss_for_g_buff_de(out_g_buff_net, input['ref_g_buff_net'])

        loss_g_buff_net.backward()
        self.optimizer_for_g_buff_net.step()

        self.out_g_buff_net = out_g_buff_net
        self.current_loss_g_buff_net = loss_g_buff_net.item()

        # base_net feed and optimization
        if epoch >= self.iter_for_g_buff_denoiser:
            in_base_net = torch.cat((input['in_color'], out_g_buff_net.detach(), input['in_colorVar']), dim=1)

            out_base_net, loss_base_net = self.base_net(in_base_net, input['ref_color'], only_img_out, full_res_out)

            loss_base_net.backward()
            self.optimizer_for_base_net.step()

            self.out_base_net = out_base_net
            self.current_loss_base_net = loss_base_net.item()


    def test(self, input, chunk_size=200):
        input['in_g_buff_net'] = input['in_g_buff_net'].to(self.device)
        input['in_color'] = input['in_color'].to(self.device)
        input['in_colorVar'] = input['in_colorVar'].to(self.device)


        self.g_buff_net.eval()
        self.base_net.eval()

        out_g_buff_net = self.g_buff_net.test_chunkwise(input['in_g_buff_net'], chunk_size=chunk_size)
        # out_g_buff_net = self.g_buff_net(input['in_g_buff_net'])

        in_base_net = torch.cat((input['in_color'], out_g_buff_net.detach(), input['in_colorVar']), dim=1)
        out_base_net = self.base_net.test_chunkwise(in_base_net, chunk_size=chunk_size)

        self.out_g_buff_net = out_g_buff_net
        self.out_base_net = out_base_net


    def save_parameter(self, epoch):
        "이건 가각의 모델과 optimizer를 저장을 하는 것이라 효율이 떨어짐."
        " ttv에 그냥 pipline 전부를 저장하는 것으로 대체가 됨."

        if (epoch + 1) % self.params['para_saving_epoch'] == 0:
            out_para_folder_name = self.params["saving_sub_folder_name"] + "/parameters"
            if not os.path.exists(out_para_folder_name):
                os.mkdir(out_para_folder_name)
            # torch.save(mynet.state_dict(), out_para_folder_name + "/latest_parameter")

            torch.save({
                'epoch': epoch,
                # 'model_state_dict': mynet.state_dict(),
                'g_buff_net_model': self.g_buff_net,
                'base_net_model': self.base_net,

                'g_buff_net_optimizer': self.optimizer_for_g_buff_net,
                'base_net_optimizer': self.optimizer_for_base_net,

                'g_buff_net_loss': self.current_loss_g_buff_net,
                'base_net_loss': self.current_loss_base_net
            }, out_para_folder_name + "/latest_parameter")


            # torch.load(out_para_folder_name + "/latest_parameter")
            
            
    def load_parameter(self, parameter_pth):
        "이건 가각의 모델과 optimizer를 저장을 하는 것이라 효율이 떨어짐."
        " ttv에 그냥 pipline 전부를 저장하는 것으로 대체가 됨."

        saved_torch_para = torch.load(parameter_pth)
        
        self.g_buff_net = saved_torch_para['g_buff_net_model']
        self.base_net = saved_torch_para['base_net_model']
        
        self.optimizer_for_g_buff_net = saved_torch_para['g_buff_net_optimizer']
        self.optimizer_for_base_net = saved_torch_para['base_net_optimizer']
        
        return saved_torch_para['epoch']


    def save_inter_results(self, epoch, input):
        input['in_g_buff_net'] = input['in_g_buff_net'].to(self.device)
        input['in_color'] = input['in_color'].to(self.device)
        input['in_colorVar'] = input['in_colorVar'].to(self.device)
        input['ref_color'] = input['ref_color'].to(self.device)
        input['ref_g_buff_net'] = input['ref_g_buff_net'].to(self.device)

        if (epoch + 1) % self.params["val_patches_saving_epoch"] == 0:
            inter_patch_folder_name = self.params["saving_sub_folder_name"] + "/val_patches"
            if not os.path.exists(inter_patch_folder_name):
                os.mkdir(inter_patch_folder_name)

            FG_net_out = self.base_net.get_FG_net_out()
            if FG_net_out != None:
                out_FG_net_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(
                    FG_net_out[:, 0:3, :, :])

            in_albedo_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(input['in_g_buff_net'][:, 0:3, :, :])
            in_depth_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(input['in_g_buff_net'][:, 3, :, :].unsqueeze(1))
            in_normal_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(input['in_g_buff_net'][:, 4:7, :, :])


            out_albedo_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(self.out_g_buff_net[:, 0:3, :, :])
            out_depth_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(self.out_g_buff_net[:, 3, :, :].unsqueeze(1))
            out_normal_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(self.out_g_buff_net[:, 4:7, :, :])

            ref_albedo_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(
                input['ref_g_buff_net'][:, 0:3, :, :])
            ref_depth_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(
                input['ref_g_buff_net'][:, 3, :, :].unsqueeze(1))
            ref_normal_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(
                input['ref_g_buff_net'][:, 4:7, :, :])

            color_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(input['in_color'])


            y_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(input['ref_color'])

            if epoch >= self.iter_for_g_buff_denoiser:
                y_pred_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(self.out_base_net)

            for l in range(color_np_saving.shape[0]):
                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_albedo_i.exr",
                          in_albedo_np_saving[l, :, :, :])
                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_depth_i.exr",
                          in_depth_np_saving[l, :, :, :])
                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_normal_i.exr",
                          in_normal_np_saving[l, :, :, :])

                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_albedo_o.exr",
                          out_albedo_np_saving[l, :, :, :])
                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_depth_o.exr",
                          out_depth_np_saving[l, :, :, :])
                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_normal_o.exr",
                          out_normal_np_saving[l, :, :, :])

                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_albedo_r.exr",
                          ref_albedo_np_saving[l, :, :, :])
                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_depth_r.exr",
                          ref_depth_np_saving[l, :, :, :])
                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_normal_r.exr",
                          ref_normal_np_saving[l, :, :, :])

                if epoch >= self.iter_for_g_buff_denoiser:
                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_color_in.exr",
                              color_np_saving[l, :, :, :])

                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_color_out.exr",
                              y_pred_np_saving[l, :, :, 0:3])

                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_color_ref.exr",
                              y_np_saving[l, :, :, 0:3])

                    if FG_net_out != None:
                        exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_FG_out.exr",
                                  out_FG_net_np_saving[l, :, :, :])


    def save_final_results(self, input, out_folder_name, f, image_index):
        input['in_g_buff_net'] = input['in_g_buff_net'].to(self.device)
        input['in_color'] = input['in_color'].to(self.device)
        input['in_colorVar'] = input['in_colorVar'].to(self.device)
        input['ref_color'] = input['ref_color'].to(self.device)
        input['ref_g_buff_net'] = input['ref_g_buff_net'].to(self.device)

        "FROM TORCH TENSOR TO NUMPY TENSOR"
        x_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(input['in_color'])
        y_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(input['ref_color'])
        y_pred_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(self.out_base_net)

        FG_net_out = self.base_net.get_FG_net_out()
        if FG_net_out != None:
            out_FG_net_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(
                FG_net_out[:, 0:3, :, :])


        out_albedo_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(self.out_g_buff_net[:, 0:3, :, :])
        out_depth_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(
            self.out_g_buff_net[:, 3, :, :].unsqueeze(1))
        out_normal_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(self.out_g_buff_net[:, 4:, :, :])

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
        exr.write(out_folder_name + "/" + self.params['saving_file_name'] + "_" + str(image_index) + "_input.exr",
                  x_np_saving)
        exr.write(out_folder_name + "/" + self.params['saving_file_name'] + "_" + str(image_index) + "_gt.exr",
                  y_np_saving)
        exr.write(out_folder_name + "/" + self.params['saving_file_name'] + "_" + str(image_index) + "_result.exr",
                  y_pred_np_saving)

        exr.write(out_folder_name + "/" + self.params['saving_file_name'] + "_" + str(image_index) + "_z_albedo.exr",
                  out_albedo_np_saving[0])
        exr.write(out_folder_name + "/" + self.params['saving_file_name'] + "_" + str(image_index) + "_z_depth.exr",
                  out_depth_np_saving[0])
        exr.write(out_folder_name + "/" + self.params['saving_file_name'] + "_" + str(image_index) + "_z_normal.exr",
                  out_normal_np_saving[0])

        if FG_net_out != None:
            exr.write(out_folder_name + "/" + self.params['saving_file_name'] + "_" + str(image_index) + "_z_FG_out.exr",
                      out_FG_net_np_saving[0])

    def get_current_losses(self):
        return self.current_loss_g_buff_net, self.current_loss_base_net

    def get_images(self):
        return self.out_g_buff_net, self.out_base_net




class WLS_net_FG_Pipeline_v1(nn.Module):
    """
    파이프라인 개요
    - WLS_net_Gde_FG_Pipeline_v1과 거의 똑같음.
    - 하지만, G buffer denoising이 없는 버전임.
    - 기존의 network를 pipeline화를 시킨것으로 보면 됨.
    - 또한, variance 정보를 주는데 최적화되도로 함.
    - 대놓고 WLS_net_FG_v2을 위한 코드.

    """

    def __init__(self, params, base_net, base_loss, use_g_buff_var=True, use_color_var=True):
        super(WLS_net_FG_Pipeline_v1, self).__init__()

        self.params = params

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.iteration = params['epochs']

        self.use_g_buff_var = use_g_buff_var
        self.use_color_Var = use_color_var


        "loss function"
        # loss를 다르게 두어도 되도록 함.
        self.loss_for_base = base_loss


        "Base network"
        self.base_net = base_net

        "optimizer"
        self.optimizer_for_base_net = optim.Adam(self.base_net.parameters(), lr=params['lr'])

        "intermediate results"
        self.out_base_net = 0

        self.current_loss_base_net = 0

    def data_feed_and_optimization(self, epoch, input, only_img_out=False, full_res_out=True):
        """
        epoch :
        input : B C_in H W
        design : B C_de H W

        특이하게 gt를 넣어서 loss까지 한방에 계산을 하도록 함.
        output : resulting image and loss

        """
        input['in_g_buff_net'] = input['in_g_buff_net'].to(self.device)
        input['in_color'] = input['in_color'].to(self.device)
        input['in_colorVar'] = input['in_colorVar'].to(self.device)
        input['ref_color'] = input['ref_color'].to(self.device)


        "INITIAL SETTING"
        self.optimizer_for_base_net.zero_grad()

        if self.use_g_buff_var:
            in_g_buff = input['in_g_buff_net']
        else:
            in_g_buff = input['in_g_buff_net'][:, :7, :, :]

        if self.use_color_Var:
            in_base_net = torch.cat((input['in_color'], in_g_buff, input['in_colorVar']), dim=1)
        else:
            in_base_net = torch.cat((input['in_color'], in_g_buff), dim=1)


        out_base_net, loss_base_net = self.base_net(in_base_net, input['ref_color'], only_img_out, full_res_out)

        loss_base_net.backward()
        self.optimizer_for_base_net.step()

        self.out_base_net = out_base_net
        self.current_loss_base_net = loss_base_net.item()



    def test(self, input, chunk_size=200):
        input['in_g_buff_net'] = input['in_g_buff_net'].to(self.device)
        input['in_color'] = input['in_color'].to(self.device)
        input['in_colorVar'] = input['in_colorVar'].to(self.device)

        self.base_net.eval()

        if self.use_g_buff_var:
            in_g_buff = input['in_g_buff_net']
        else:
            in_g_buff = input['in_g_buff_net'][:, :7, :, :]

        if self.use_color_Var:
            in_base_net = torch.cat((input['in_color'], in_g_buff, input['in_colorVar']), dim=1)
        else:
            in_base_net = torch.cat((input['in_color'], in_g_buff), dim=1)

        out_base_net = self.base_net.test_chunkwise(in_base_net, chunk_size=chunk_size)

        self.out_base_net = out_base_net

    def save_parameter(self, epoch):
        "이건 가각의 모델과 optimizer를 저장을 하는 것이라 효율이 떨어짐."
        " ttv에 그냥 pipline 전부를 저장하는 것으로 대체가 됨."

        if (epoch + 1) % self.params['para_saving_epoch'] == 0:
            out_para_folder_name = self.params["saving_sub_folder_name"] + "/parameters"
            if not os.path.exists(out_para_folder_name):
                os.mkdir(out_para_folder_name)
            # torch.save(mynet.state_dict(), out_para_folder_name + "/latest_parameter")

            torch.save({
                'epoch': epoch,
                # 'model_state_dict': mynet.state_dict(),
                'base_net_model': self.base_net,

                'base_net_optimizer': self.optimizer_for_base_net,

                'base_net_loss': self.current_loss_base_net
            }, out_para_folder_name + "/latest_parameter")

            # torch.load(out_para_folder_name + "/latest_parameter")

    def load_parameter(self, parameter_pth):
        "이건 가각의 모델과 optimizer를 저장을 하는 것이라 효율이 떨어짐."
        " ttv에 그냥 pipline 전부를 저장하는 것으로 대체가 됨."

        saved_torch_para = torch.load(parameter_pth)

        self.base_net = saved_torch_para['base_net_model']

        self.optimizer_for_base_net = saved_torch_para['base_net_optimizer']

        return saved_torch_para['epoch']

    def save_inter_results(self, epoch, input):
        input['in_g_buff_net'] = input['in_g_buff_net'].to(self.device)
        input['in_color'] = input['in_color'].to(self.device)
        input['in_colorVar'] = input['in_colorVar'].to(self.device)
        input['ref_color'] = input['ref_color'].to(self.device)

        if (epoch + 1) % self.params["val_patches_saving_epoch"] == 0:
            inter_patch_folder_name = self.params["saving_sub_folder_name"] + "/val_patches"
            if not os.path.exists(inter_patch_folder_name):
                os.mkdir(inter_patch_folder_name)

            in_albedo_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(
                input['in_g_buff_net'][:, 0:3, :, :])
            in_depth_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(
                input['in_g_buff_net'][:, 3, :, :].unsqueeze(1))
            in_normal_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(
                input['in_g_buff_net'][:, 4:7, :, :])

            FG_net_out = self.base_net.get_FG_net_out()

            if FG_net_out != None:
                out_FG_net_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(
                    FG_net_out[:, 0:3, :, :])

            ref_albedo_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(
                input['ref_g_buff_net'][:, 0:3, :, :])
            ref_depth_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(
                input['ref_g_buff_net'][:, 3, :, :].unsqueeze(1))
            ref_normal_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(
                input['ref_g_buff_net'][:, 4:7, :, :])

            color_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(input['in_color'])

            y_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(input['ref_color'])

            y_pred_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(self.out_base_net)

            for l in range(color_np_saving.shape[0]):
                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_albedo_i.exr",
                          in_albedo_np_saving[l, :, :, :])
                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_depth_i.exr",
                          in_depth_np_saving[l, :, :, :])
                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_normal_i.exr",
                          in_normal_np_saving[l, :, :, :])

                if FG_net_out != None:
                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_FG_out.exr",
                              out_FG_net_np_saving[l, :, :, :])


                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_albedo_r.exr",
                          ref_albedo_np_saving[l, :, :, :])
                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_depth_r.exr",
                          ref_depth_np_saving[l, :, :, :])
                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_normal_r.exr",
                          ref_normal_np_saving[l, :, :, :])

                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_color_in.exr",
                          color_np_saving[l, :, :, :])

                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_color_out.exr",
                          y_pred_np_saving[l, :, :, 0:3])

                exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_color_ref.exr",
                          y_np_saving[l, :, :, 0:3])



    def save_final_results(self, input, out_folder_name, f, image_index):
        input['in_g_buff_net'] = input['in_g_buff_net'].to(self.device)
        input['in_color'] = input['in_color'].to(self.device)
        input['in_colorVar'] = input['in_colorVar'].to(self.device)
        input['ref_color'] = input['ref_color'].to(self.device)

        "FROM TORCH TENSOR TO NUMPY TENSOR"
        x_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(input['in_color'])
        y_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(input['ref_color'])
        y_pred_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(self.out_base_net)

        FG_net_out = self.base_net.get_FG_net_out()

        if FG_net_out != None:
            out_FG_net_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(
                FG_net_out[:, 0:3, :, :])


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
        exr.write(out_folder_name + "/" + self.params['saving_file_name'] + "_" + str(image_index) + "_input.exr",
                  x_np_saving)
        exr.write(out_folder_name + "/" + self.params['saving_file_name'] + "_" + str(image_index) + "_gt.exr",
                  y_np_saving)
        exr.write(out_folder_name + "/" + self.params['saving_file_name'] + "_" + str(image_index) + "_result.exr",
                  y_pred_np_saving)

        if FG_net_out != None:
            exr.write(out_folder_name + "/" + self.params['saving_file_name'] + "_" + str(image_index) + "_z_FG_out.exr",
                      out_FG_net_np_saving[0])


    def get_current_losses(self):
        return self.current_loss_base_net

    def get_images(self):
        return self.out_base_net

