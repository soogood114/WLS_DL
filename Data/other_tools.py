import numpy as np
import torch
import Models.net_op as net_op
import os
import Data.exr as exr

def calcRelMSE(input, ref):
    h, w, _ = np.shape(input)
    num = np.square(np.subtract(input, ref))
    denom = np.mean(ref, axis=2)
    denom = np.reshape(denom, (h, w, 1))
    relMSE = num / (denom * denom + 1e-2)
    relMSEMean = np.mean(relMSE)
    return relMSEMean


def from_torch_tensor_stack_to_full_res_numpy(torch_tensor_stack):
    """
    input : stack version of torch tensor (b tile_size c h_d w_d)
    output : full res version of numpy tensor
    """

    torch_tensor = net_op.make_full_res_img_torch(torch_tensor_stack)  # b c h w
    torch_tensor_np = torch_tensor.cpu().detach().numpy()
    torch_tensor_np = np.transpose(torch_tensor_np, (0, 2, 3, 1))

    return torch_tensor_np

def from_torch_tensor_img_to_full_res_numpy(torch_tensor_stack):
    """
    input : stack version of torch tensor (b tile_size c h_d w_d)
    output : full res version of numpy tensor
    """

    torch_tensor_np = torch_tensor_stack.cpu().detach().numpy()
    torch_tensor_np = np.transpose(torch_tensor_np, (0, 2, 3, 1))

    return torch_tensor_np


def find_text_files(scene_model_name_pth, scene_name):
    """
    !! 임시 !!
    4.13에 만든 각 모델 마다의 결과 txt 병합을 위해 임시로 만듦.

    """
    # scene_name = "bathroom2"  # bathroom2, car, kitchen, living-room, living-room-3, veach-ajar
    input_spp_list = [32, 100, 256, 512, 1024]

    with open(scene_model_name_pth + '/relMSE_all_spp.txt', 'w') as outfile:
        for i in range(len(input_spp_list)):
            file_pth = scene_model_name_pth + "/" + scene_name + "_" + str(input_spp_list[i]) + "/test_imgs"
            for file_name in os.listdir(file_pth):
                if file_name.endswith(".txt"):
                    file = open(os.path.join(file_pth, file_name))
                    outfile.write(file.read())


def get_nfor_img_and_get_relmse(in_pth):
    """
    !! 임시 !!
    기존에 tungsten 랜더러에서 나온 buffer에서 nfor을 가져옴.
    그리고 relMSE를 나오게 뽑음.
    """
    input_spp_list = [32, 100, 256, 512, 1024, 2048, 4096] # [32, 100, 256, 512, 1024]
    out_pth = in_pth + "/nfor"

    if not os.path.exists(out_pth):
        os.mkdir(out_pth)

    f = open(out_pth + '/nfor_relMSE.txt', 'w')

    ref_buffer = exr.read_all(os.path.join(in_pth, "out_64kspp.exr"))
    ref_color = ref_buffer['diffuse'] + ref_buffer['specular']

    for i in range(len(input_spp_list)):
        input_name = "out_" + str(input_spp_list[i]) + "spp.exr"

        input_buffer = exr.read_all(os.path.join(in_pth, input_name))

        input_nfor = input_buffer['nfor']

        rmse = calcRelMSE(input_nfor, ref_color)

        rmse_str = str(input_spp_list[i]) + "spp image relMSE : " + str(rmse)
        f.write(rmse_str)
        f.write("\n")
        print(rmse_str)

        exr.write(out_pth + "/" + str(input_spp_list[i]) + "spp_nfor.exr", input_nfor)



# if __name__ == "__main__":
#     "임시"
#     find_text_files("F:/DH/history/4.13/210412_WLS_net_FG_v1(g-buffer)_norm_epoch_1000/bathroom2_210412_WLS_net_FG_v1(g-buffer)_norm_epoch_1000",
#                     "bathroom2")
#     find_text_files("F:/DH/history/4.13/210412_WLS_net_FG_v1(g-buffer)_norm_epoch_1000/car_210412_WLS_net_FG_v1(g-buffer)_norm_epoch_1000",
#                     "car")
#     find_text_files("F:/DH/history/4.13/210412_WLS_net_FG_v1(g-buffer)_norm_epoch_1000/kitchen_210412_WLS_net_FG_v1(g-buffer)_norm_epoch_1000",
#                     "kitchen")
#     find_text_files("F:/DH/history/4.13/210412_WLS_net_FG_v1(g-buffer)_norm_epoch_1000/living-room_210412_WLS_net_FG_v1(g-buffer)_norm_epoch_1000",
#                     "living-room")
#     find_text_files("F:/DH/history/4.13/210412_WLS_net_FG_v1(g-buffer)_norm_epoch_1000/living-room-3_210412_WLS_net_FG_v1(g-buffer)_norm_epoch_1000",
#                     "living-room-3")
#     find_text_files("F:/DH/history/4.13/210412_WLS_net_FG_v1(g-buffer)_norm_epoch_1000/veach-ajar_210412_WLS_net_FG_v1(g-buffer)_norm_epoch_1000",
#                     "veach-ajar")

    # bathroom2, car, kitchen, living-room, living-room-3, veach-ajar
    # curly-hair, staircase2, glass-of-water, teapot-full
    # get_nfor_img_and_get_relmse("E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/tungsten_test_scenes/glass-of-water")






