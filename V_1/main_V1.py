import V_1.run_V1 as run
import json



def multi_imgs_test_from_one_image_test():
        print("multi_imgs_test_from_one_image_test start !!")

        base_pth = "E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/tungsten_test_scenes"

        # 입력
        # bathroom2, car, kitchen, living-room, living-room-3, veach-ajar,
        # curly-hair, staircase2, glass-of-water, teapot-full
        scene_name = "living-room"

        input_file_name = "out_32spp.exr"
        input_spp_list = [32, 100, 256, 512, 1024, 2048, 4096]
        ref_file_name = "out_64kspp.exr"


        "model 설정"
        test_params = {
                'trained_model_folder_pth': "F:/DH/history/4.09/210407_WLS_net_v1_128_epoch_1k/04_08_10_18_35",
                'trained_parameter_name': "latest_parameter",
                "grid_order": 0,

                'loss_type': 'l1',  # l1, l2, custom_loss_v1
                'loss_weights': [0.8, 0.2],  # colorloss + gradientloss
                'SMAP_loss': False,  # for custom_loss_v1
                'reinhard_loss': False,  # for custom_loss_v1

                # 입력
                'network_name': "WLS_net_FG_v1",  # WLS_net_v1, KPCN_v1, WLS_net_FG_v1

                # 10. Saving folder setting
                'saving_folder_name': "KPCN_v1_128_epoch_1k_soft_max",
                'saving_sub_folder_name': None,  # if None, it will replaced to time, day string
                'saving_file_name': "KPCN_v1_128_epoch_1k_soft_max",
        }
        ""
        # 입력
        test_params['trained_model_folder_pth'] = "F:/DH/history/4.09/210417_WLS_net_FG_v1(denoised_g-buf)_norm_epoch_1000/04_17_12_16_32"

        # 입력
        test_params['grid_order'] = 1

        # 입력
        test_params['saving_folder_name'] = scene_name + "_" + "WLS_net_FG_v1(denoised_g-buf)_norm_epoch_1000"

        for i in range(len(input_spp_list)):
                input_file_name = "out_" + str(input_spp_list[i]) + "spp.exr"
                test_params['saving_sub_folder_name'] = scene_name + "_" + str(input_spp_list[i])

                # 입력
                test_params['saving_file_name'] = scene_name + "_" + str(input_spp_list[i]) + "_" + "WLS_net_FG_v1(denoised_g-buf)_norm_epoch_1000"


                input_pth = base_pth + "/" + scene_name + "/" + input_file_name
                ref_pth = base_pth + "/" + scene_name + "/" + ref_file_name
                buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal']
                run.one_exr_load_and_test(test_params, input_pth, ref_pth, buffer_list, gpu_id=0)



if __name__ == "__main__":
        "train or test"
        print("train or test start !!")
        run.data_load_and_run(params=None, gpu_id=0)


        "re train"
        # print("re train mode start !!")
        # setting_path = "./results/210411_WLS_net_FG_v1_epoch_100/04_11_15_49_15" + "/settings/setting_params.json"
        # with open(setting_path) as json_file:
        #         params = json.load(json_file)
        # params['trained_model_RETRAIN'] = True
        # run.data_load_and_run(params=params, gpu_id=0)


        "one_image_test"
        # print("one_image_test start !!")
        # test_params = {
        #         'trained_model_folder_pth': "F:/DH/history/4.09/210407_KPCN_v1_128_epoch_1k_soft_max/04_07_17_26_01",
        #         'trained_parameter_name': "latest_parameter",
        #         "grid_order": 1,
        #
        #         'loss_type': 'l1',  # l1, l2, custom_loss_v1
        #         'loss_weights': [0.8, 0.2],  # colorloss + gradientloss
        #         'SMAP_loss': False,  # for custom_loss_v1
        #         'reinhard_loss': False,  # for custom_loss_v1
        #
        #         'network_name': "KPCN_v1",  # WLS_net_v1, KPCN_v1, WLS_net_FG_v1
        #
        #         # 10. Saving folder setting
        #         'saving_folder_name': "KPCN_v1_128_epoch_1k_soft_max",
        #         'saving_sub_folder_name': "None",  # if None, it will replaced to time, day string
        #         'saving_file_name': "KPCN_v1_128_epoch_1k_soft_max",
        # }
        # # test_params['trained_model_folder_pth'] = "F:/DH/history/4.09/210407_WLS_net_v1_128_epoch_100/04_10_11_49_10"
        #
        # base_pth = "E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/tungsten_test_scenes"
        # scene_name = "bathroom2"
        # input_file_name = "out_32spp.exr"
        # ref_file_name = "out_64kspp.exr"
        #
        # input_pth = base_pth + "/" + scene_name + "/" + input_file_name
        # ref_pth = base_pth + "/" + scene_name + "/" + ref_file_name
        # buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal']
        # test_params['saving_folder_name'] = "ONE_IMG_TEST_" + test_params['saving_folder_name']
        # run.one_exr_load_and_test(test_params, input_pth, ref_pth, buffer_list, gpu_id=0)

        # multi_imgs_test_from_one_image_test()