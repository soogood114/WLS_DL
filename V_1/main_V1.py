import V_1.run_V1 as run
import json

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

