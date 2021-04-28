import os
import numpy as np
import glob
from random import shuffle
import h5py
import torch

import Data.exr as exr
import Data.manipulate as mani
import Data.normalization as norm
import Data.design_matrix as design

"""*******************************************************************************"""
"load.py에 대한 정보"
" network train and test에 앞서 exr 또는 numpy 형태의 dataset을 SSD에서 불러오는 함수들로 구성"
"""*******************************************************************************"""


def get_exr_dataset(dataset_dir, suffix='*.exr'):
    dataset = []
    files = os.listdir(dataset_dir)

    files = [fn for fn in glob.glob(os.path.join(dataset_dir, suffix))]

    for file in files:
        filename = os.path.join(dataset_dir, file)
        # print("Loading: ", filename)
        data = exr.read(filename)
        # data = np.clip(data, 0, 1)
        dataset.append(data[:, :, 0:3])

    dataset = np.array(dataset)

    return dataset


def get_dat_dataset(dataset_dir, exr_dataset):
    dataset = []
    files = os.listdir(dataset_dir)

    for i, file in enumerate(files):
        filename = os.path.join(dataset_dir, file)
        data = np.fromfile(filename, dtype=np.float32)

        h, w, _ = exr_dataset[i].shape
        data = np.reshape(data, (h, w, 3))

        dataset.append(data)

    dataset = np.array(dataset)
    return dataset


def save_all_exr_dataset(dataset_dirs, scene, target):
    all_data = []
    for dataset_dir in dataset_dirs:
        files = os.listdir(dataset_dir)
        files = [fn for fn in glob.glob(os.path.join(dataset_dir, '*.exr'))]

        for f in files:
            filename = os.path.join(dataset_dir, f)
            data = exr.read_all(filename)
            all_data.append(data['default'][:, :, 0:3])
            exr.write(os.path.join('D:/training/', target, scene, f), data['default'][:, :, 0:3])

    return np.array(all_data)


def get_all_exr_dataset(dataset_dirs, suffix=""):
    all_data = []
    for dataset_dir in dataset_dirs:
        files = os.listdir(dataset_dir)
        files = [file for file in files if file.endswith(".exr")]
        # files = [fn for fn in glob.glob(os.path.join(dataset_dir, suffix))]

        for f in files:
            filename = os.path.join(dataset_dir, f)
            data = exr.read_all(filename)
            all_data.append(data['default'][:, :, 0:3])

    return np.array(all_data)


def get_all_npy_dataset(dataset_dirs):
    all_data = []
    for dataset_dir in dataset_dirs:
        files = os.listdir(dataset_dir)
        files = [file for file in files if file.endswith(".npy")]
        for f in files:
            filename = os.path.join(dataset_dir, f)
            data = np.load(filename)
            all_data.append(data)

    return np.array(all_data)


def get_all_exr_dataset_one(dataset_dirs, suffix=""):
    all_data = []
    for dataset_dir in dataset_dirs:
        files = os.listdir(dataset_dir)
        files = [file for file in files if file.endswith(".exr")]
        # files = [fn for fn in glob.glob(os.path.join(dataset_dir, suffix))]

        filename = os.path.join(dataset_dir, files[0])
        data = exr.read_all(filename)
        all_data.append(data['default'][:, :, 0:3])

    return np.array(all_data)


def get_all_npy_dataset_one(dataset_dirs):
    all_data = []
    for dataset_dir in dataset_dirs:
        files = os.listdir(dataset_dir)
        files = [file for file in files if file.endswith(".npy")]

        filename = os.path.join(dataset_dir, files[0])
        data = np.load(filename)
        all_data.append(data)

    return np.array(all_data)



def shuffle_list(*ls):
    l = list(zip(*ls))
    shuffle(l)
    return zip(*l)


def concatenate_4d_dataset(*datasets):
    dataset_group = list(zip(*datasets))
    merge = [np.concatenate(ele, axis=2) for ele in dataset_group]
    return np.array(merge)


def get_bmp_dataset_pair(dataset_dir):
    from scipy import misc

    resized_dataset = []
    dataset = []

    scale = 2
    files = os.listdir(dataset_dir)

    for file in files:
        filename = os.path.join(dataset_dir, file)
        data = misc.imread(filename)

        resized_data = misc.imresize(data, size=1.0 / scale, interp='bicubic')
        resized_data = misc.imresize(resized_data, size=scale * 100, interp='bicubic')

        dataset.append(data[:, :, 0:3])
        resized_dataset.append(resized_data[:, :, 0:3])

        exr.write('debug-bmp/' + file + '.exr', data[:, :, 0:3] / 255.0)
        exr.write('debug-bmp-resized/' + file + '.exr', resized_data[:, :, 0:3] / 255.0)

    resized_dataset = np.array(resized_dataset)
    dataset = np.array(dataset)

    return resized_dataset, dataset



"""new ones"""

def get_all_stack_npy_for_each_buffer(dataset_dirs, common_name="total_s_4_", ref_pt=True):
    """
    input : dataset dir pth
    output : throughput, direct, g_buffer, GT
    feature #1 : 각 buffer에 따라 출력을 냄.
    """
    all_throughput_stack = np.load(dataset_dirs + common_name + "throughput_stack.npy")
    all_direct_stack = np.load(dataset_dirs + common_name + "direct_stack.npy")
    all_g_buffer_stack = np.load(dataset_dirs + common_name + "g_buffer_stack.npy")
    if ref_pt:
        all_GT_stack = np.load(dataset_dirs + common_name + "GT_stack_pt.npy")
    else:
        all_GT_stack = np.load(dataset_dirs + common_name + "GT_stack_weighted.npy")

    return all_throughput_stack, all_direct_stack, all_g_buffer_stack, all_GT_stack


def get_all_stack_npy_for_input_buffer(dataset_dirs, common_name="total_s_4_", ref_pt=True):
    """
    input : dataset dir pth
    output : input_stack(throughput + direct, g_buffer), GT
    feature #1 : input stack을 위와 같이 만드러 출력, throughput은 direct와 합쳐짐.
    """
    all_throughput_stack = np.load(dataset_dirs + common_name + "throughput_stack.npy")
    all_direct_stack = np.load(dataset_dirs + common_name + "direct_stack.npy")
    all_g_buffer_stack = np.load(dataset_dirs + common_name + "g_buffer_stack.npy")
    if ref_pt:
        all_GT_stack = np.load(dataset_dirs + common_name + "GT_stack_pt.npy")
    else:
        all_GT_stack = np.load(dataset_dirs + common_name + "GT_stack_weighted.npy")


    "debug"
    # throughput_stack_test = all_throughput_stack[0, :, :, 0, :]
    # exr.write("./throughput_stack_test.exr", throughput_stack_test)
    #
    # GT_stack_test = all_GT_stack[0, :, :, 0, :]
    # exr.write("./GT_stack_test.exr", GT_stack_test)
    #
    # direct_stack_test = all_direct_stack[0, :, :, 0, :]
    # exr.write("./direct_stack_test.exr", direct_stack_test)
    #
    # albedo_stack_test = all_g_buffer_stack[0, :, :, 0, :3]
    # exr.write("./albedo_stack_test.exr", albedo_stack_test)
    #
    # depth_stack_test = all_g_buffer_stack[0, :, :, 0, 3]
    # exr.write("./depth_stack_test.exr", depth_stack_test)

    return np.concatenate((all_throughput_stack + all_direct_stack, all_g_buffer_stack), axis=4), all_GT_stack


def get_all_img_npy_for_input_buffer(dataset_dirs, common_name="total_s_4_"):
    """
    input : dataset dir pth
    output : input_img(throughput + direct, g_buffer), GT
    feature #1 : 위의 함수와는 다르게 stack이 아니라, img 형태로 출력
    """
    all_throughput_img = mani.make_full_res_img_numpy(np.load(dataset_dirs + common_name + "throughput_stack.npy"))
    all_direct_img = mani.make_full_res_img_numpy(np.load(dataset_dirs + common_name + "direct_stack.npy"))
    all_g_buffer_img = mani.make_full_res_img_numpy(np.load(dataset_dirs + common_name + "g_buffer_stack.npy"))
    all_GT_img = mani.make_full_res_img_numpy(np.load(dataset_dirs + common_name + "GT_stack_weighted.npy"))

    return np.concatenate((all_throughput_img + all_direct_img, all_g_buffer_img), axis=3), all_GT_img


def get_all_img_exr_from_stack_npy(DIR, mini_batch=True, test_mode=False):
    """ mini batch, test mode 를 고려해서 npy를 가져와 image혈태의 data로 출력"""
    """ 그러나 gradient와 pth tracing을 학습을 시킬 수 없음."""

    DIR += "npy/"

    if mini_batch:
        train_pth = DIR + "3. mini_batch/"
    else:
        train_pth = DIR + "1. train/"

    test_pth = DIR + "2. test/"

    if not test_mode:
        train_input_img_buffer, train_ref_img_buffer = get_all_img_npy_for_input_buffer(train_pth)
        test_input_img_buffer, test_ref_img_buffer = get_all_img_npy_for_input_buffer(test_pth)
    else:
        train_input_img_buffer = 0
        train_ref_img_buffer = 0
        test_input_img_buffer, test_ref_img_buffer = get_all_img_npy_for_input_buffer(test_pth)

    return train_input_img_buffer, train_ref_img_buffer, test_input_img_buffer, test_ref_img_buffer


def get_input_design_stack_and_normalize(dirs, common_name, params):
    """
    메모리 효율을 위해 이 함수 내에서 loading과 normalization 둘다를 하도록 함.
    또한, input과 design에서 서로
    """
    # load
    input_stack, GT_stack = get_all_stack_npy_for_input_buffer(dirs, common_name, params['ref_pt'])

    # normalization
    norm.normalize_input_stack_v1(input_stack)
    norm.normalize_GT_v1(GT_stack)

    # design matrix
    design_stack = design.generate_design_mat_from_stack_v1(input_stack[:, :, :, :, 3:],
                                                            params['tile_length'], params['grid_order'])

    # exclude boundary if it is ok
    s = params['tile_length']
    tile_size = s ** 2
    tile_size_stit = s ** 2 + s * 2

    if params["no_boundary_for_input"]:
        ch_input = tile_size
    else:
        ch_input = tile_size_stit


    if params["no_boundary_for_design"]:
        # design은 나중에 나올 output의 형태를 결정. 따라서 loss를 구하기 위해 GT도 그에 맞춰야 함.
        ch_design = tile_size
        ch_gt = tile_size
    else:
        ch_design = tile_size_stit
        ch_gt = tile_size_stit

    return input_stack[:, :, :, :ch_input, :], design_stack[:, :, :, :ch_design, :], GT_stack[:, :, :, :ch_gt, :]



def get_all_img_exr_for_ttv_v1(DIR, mini_batch=True, test_mode=False, ):
    """
    dirs : TRAIN or TEST path of DB_FOR_SIGA21
    buffer_info : pr_color + direct, pt_color, albedo, depth, normal, g_dx, g_dy, GT

    output :

    특징 1 : 하나의 함수에서 train, test, minibatch를 다 얻을 수 있도록 함.
    특징 2 : 다만 hard coding을 해서 output format과 buffer info를 바꿀 수 없음.
    """

    def get_all_names_from_folder(list, pth):
        files = os.listdir(pth)
        files = [file for file in files if file.endswith(".exr")]

        if mini_batch:
            list.append(os.path.join(pth, files[0]))
        else:
            for i in range(len(files)):
                list.append(os.path.join(pth, files[i]))

    def get_all_exr_from_names(tile, direct, noisy, albedo, depth, normal, dx, dy, gt):
        sample = exr.read(tile[0])
        h, w, _ = sample.shape
        n = len(tile)

        input_buffer = np.zeros((n, h, w, 3 + 3 + 7 + 3 + 3), dtype=sample.dtype)
        ref_buffer = np.zeros((n, h, w, 3), dtype=sample.dtype)

        for i in range(n):
            input_buffer[i, :, :, :3] = exr.read(tile[i]) + exr.read(direct[i])  # pr + direct
            input_buffer[i, :, :, 3:6] = exr.read(noisy[i])  # noisy

            input_buffer[i, :, :, 6:9] = exr.read(albedo[i])
            depth_3ch = exr.read(depth[i])
            input_buffer[i, :, :, 9] = depth_3ch[:, :, 0]
            input_buffer[i, :, :, 10:13] = exr.read(normal[i])

            input_buffer[i, :, :, 13:16] = exr.read(dx[i])
            input_buffer[i, :, :, 16:19] = exr.read(dy[i])

            ref_buffer[i] = exr.read(gt[i])

        return input_buffer, ref_buffer

    VERSION = ['V_1', 'V_2']

    """###############  Train  ###############"""
    if not test_mode:

        SCENE = ['bathroom', 'bathroom-gpt', 'classroom', 'dining-room', 'kitchen', 'veach-door']

        TILED_COLOR = []
        DIRECT = []
        NOISY_COLOR = []

        ALBEDO = []
        DEPTH = []
        NORMAL = []

        DIFF_R = []
        DIFF_B = []

        GT = []

        for i in range(len(VERSION)):
            for j in range(len(SCENE)):
                get_all_names_from_folder(TILED_COLOR,
                                          os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'throughput'))
                get_all_names_from_folder(DIRECT, os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'direct'))
                get_all_names_from_folder(NOISY_COLOR,
                                          os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'noisy_color'))

                get_all_names_from_folder(ALBEDO, os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'albedo'))
                get_all_names_from_folder(DEPTH, os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'depth'))
                get_all_names_from_folder(NORMAL, os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'normal'))

                get_all_names_from_folder(DIFF_R, os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'g_dx'))
                get_all_names_from_folder(DIFF_B, os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'g_dy'))
                get_all_names_from_folder(GT,
                                          os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'ref', 'weighted_recon'))

        # merge
        train_input_buffer, train_ref_buffer = get_all_exr_from_names(TILED_COLOR, DIRECT, NOISY_COLOR, ALBEDO,
                                                                      DEPTH, NORMAL, DIFF_R, DIFF_B, GT)
    else:
        train_input_buffer = 0  # NULL value
        train_ref_buffer = 0

    """###############  Test  ###############"""
    SCENE = ['bathroom2', 'bookshelf-gpt', 'kitchen-gpt', 'bathroom', 'bathroom-gpt',
             'classroom', 'dining-room', 'kitchen', 'veach-door']

    TILED_COLOR = []
    DIRECT = []
    NOISY_COLOR = []

    ALBEDO = []
    DEPTH = []
    NORMAL = []

    DIFF_R = []
    DIFF_B = []

    GT = []

    for j in range(len(SCENE)):
        get_all_names_from_folder(TILED_COLOR, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'throughput'))
        get_all_names_from_folder(DIRECT, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'direct'))
        get_all_names_from_folder(NOISY_COLOR, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'noisy_color'))

        get_all_names_from_folder(ALBEDO, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'albedo'))
        get_all_names_from_folder(DEPTH, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'depth'))
        get_all_names_from_folder(NORMAL, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'normal'))

        get_all_names_from_folder(DIFF_R, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'g_dx'))
        get_all_names_from_folder(DIFF_B, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'g_dy'))
        get_all_names_from_folder(GT, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'ref', 'weighted_recon'))

    test_input_buffer, test_ref_buffer = get_all_exr_from_names(TILED_COLOR, DIRECT, NOISY_COLOR, ALBEDO,
                                                                DEPTH, NORMAL, DIFF_R, DIFF_B, GT)

    return train_input_buffer, train_ref_buffer, test_input_buffer, test_ref_buffer


def get_all_img_exr_for_ttv_v2(DIR, mini_batch=True, test_mode=False, use_noisy_color=True, use_gradients=True):
    """
    dirs : TRAIN or TEST path of DB_FOR_SIGA21
    buffer_info : [pr_color + direct OR pt_color], albedo, depth, normal, [(g_dx, g_dy) OR NONE], GT

    output :

    특징 1 : 같은 이름의 v1과 거의 동일
    특징 2 : 다만, 칼라와 gradient를 선택 포함할 수 있음.
    """

    def get_all_names_from_folder(list, pth):
        files = os.listdir(pth)
        files = [file for file in files if file.endswith(".exr")]

        if mini_batch:
            list.append(os.path.join(pth, files[0]))
        else:
            for i in range(len(files)):
                list.append(os.path.join(pth, files[i]))

    def get_all_exr_from_names(color, direct, albedo, depth, normal, dx, dy, gt):
        sample = exr.read(color[0])
        h, w, _ = sample.shape
        n = len(color)

        if use_gradients:
            ch = 16
        else:
            ch = 10

        input_buffer = np.zeros((n, h, w, ch), dtype=sample.dtype)

        ref_buffer = np.zeros((n, h, w, 3), dtype=sample.dtype)

        for i in range(n):
            if use_noisy_color:
                input_buffer[i, :, :, :3] = exr.read(color[i])  # noisy
            else:
                input_buffer[i, :, :, :3] = exr.read(color[i]) + exr.read(direct[i])  # pr + direct

            input_buffer[i, :, :, 3:6] = exr.read(albedo[i])
            depth_3ch = exr.read(depth[i])
            input_buffer[i, :, :, 6] = depth_3ch[:, :, 0]
            input_buffer[i, :, :, 7:10] = exr.read(normal[i])

            if use_gradients:
                input_buffer[i, :, :, 10:13] = exr.read(dx[i])
                input_buffer[i, :, :, 13:16] = exr.read(dy[i])

            ref_buffer[i] = exr.read(gt[i])

        return input_buffer, ref_buffer

    VERSION = ['V_1', 'V_2']

    """###############  Train  ###############"""
    if not test_mode:

        SCENE = ['bathroom', 'bathroom-gpt', 'classroom', 'dining-room', 'kitchen', 'veach-door']

        COLOR = []
        DIRECT = []

        ALBEDO = []
        DEPTH = []
        NORMAL = []

        DIFF_R = []
        DIFF_B = []

        GT = []

        for i in range(len(VERSION)):
            for j in range(len(SCENE)):
                if use_noisy_color:
                    get_all_names_from_folder(COLOR, os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'noisy_color'))
                else:
                    get_all_names_from_folder(COLOR, os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'throughput'))
                    get_all_names_from_folder(DIRECT, os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'direct'))

                get_all_names_from_folder(ALBEDO, os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'albedo'))
                get_all_names_from_folder(DEPTH, os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'depth'))
                get_all_names_from_folder(NORMAL, os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'normal'))

                get_all_names_from_folder(DIFF_R, os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'g_dx'))
                get_all_names_from_folder(DIFF_B, os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'g_dy'))
                get_all_names_from_folder(GT,
                                          os.path.join(DIR, VERSION[i], '1. train', SCENE[j], 'ref', 'weighted_recon'))

        # merge
        train_input_buffer, train_ref_buffer = get_all_exr_from_names(COLOR, DIRECT, ALBEDO, DEPTH, NORMAL,
                                                                      DIFF_R, DIFF_B, GT)
    else:
        train_input_buffer = 0  # NULL value
        train_ref_buffer = 0

    """###############  Test  ###############"""
    SCENE = ['bathroom2', 'bookshelf-gpt', 'kitchen-gpt', 'bathroom', 'bathroom-gpt',
             'classroom', 'dining-room', 'kitchen', 'veach-door']

    COLOR = []
    DIRECT = []

    ALBEDO = []
    DEPTH = []
    NORMAL = []

    DIFF_R = []
    DIFF_B = []

    GT = []

    for j in range(len(SCENE)):
        if use_noisy_color:
            get_all_names_from_folder(COLOR, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'noisy_color'))
        else:
            get_all_names_from_folder(COLOR, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'throughput'))
            get_all_names_from_folder(DIRECT, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'direct'))

        get_all_names_from_folder(ALBEDO, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'albedo'))
        get_all_names_from_folder(DEPTH, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'depth'))
        get_all_names_from_folder(NORMAL, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'normal'))

        get_all_names_from_folder(DIFF_R, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'g_dx'))
        get_all_names_from_folder(DIFF_B, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'g_dy'))
        get_all_names_from_folder(GT, os.path.join(DIR, VERSION[0], '2. test', SCENE[j], 'ref', 'weighted_recon'))

    test_input_buffer, test_ref_buffer = get_all_exr_from_names(COLOR, DIRECT, ALBEDO,
                                                                DEPTH, NORMAL, DIFF_R, DIFF_B, GT)

    return train_input_buffer, train_ref_buffer, test_input_buffer, test_ref_buffer


def load_exrs_from_tungsten(DIR, SCENE, BUFFER, mini_batch=True, flag_saving=False, saving_pth="tmp",
                            input_endswith="00128spp.exr", ref_endswith="08192spp.exr", load_dtype="HALF"):
    """
    input : Dir = scene 폴더들이 있는 상위 폴더 위치, SCENE = 가져올 scene, BUFFER = 특정 feature 버퍼 이름
    output : input buffer, ref buffer
    특징 1 : original tungsten DB에서 원하는 input, ref 버퍼 만들기
    특징 2 : 하나하나 읽어와 저장하는 구조라서 시간이 많이 걸림.
    특징 3 : input = 124 spp, target = 8k spp 고정
    특징 4 : 데이터의 양이 너무 많아서 굉장히 불러오는데 오래걸림.

    """

    # DIR = 'D:/Tunsten_deep_learning_denoising_dataset/deep_learning_denoising/renderings'
    # SCENE = ['bathroom2', 'car2', 'classroom', 'house', 'room2', 'room3', 'spaceship', 'staircase']
    # BUFFER = ['diffuse', 'specular', 'albedo', 'depth', 'normal']

    "path 불러오기"
    ALL_SCENES = []  # 모든 spp, feature에 해당하는 buffer 이름
    input_all_features = []  # [scene, files]
    ref_all_feature = []

    total_num_imgs = 0

    for i in range(len(SCENE)):
        # ALL_SCENES.append(os.path.join(DIR, SCENE[i]))

        files = os.listdir(os.path.join(DIR, SCENE[i]))

        input_all_features_one_scene = [file for file in files if file.endswith(input_endswith)]
        ref_all_features_one_scene = [file for file in files if file.endswith(ref_endswith)]

        if not mini_batch:
            input_all_features.append(input_all_features_one_scene)
            ref_all_feature.append(ref_all_features_one_scene)
            total_num_imgs += len(input_all_features_one_scene)
        else:
            input_all_features.append(input_all_features_one_scene[:2])
            ref_all_feature.append(ref_all_features_one_scene[:2])
            total_num_imgs += 2


    "저장 공간 설정 from sample"
    sample_dict = exr.read_all(os.path.join(DIR, SCENE[0], input_all_features[0][0]), precision=load_dtype)
    total_ch = 0
    ch_BUFFER = []
    for b in BUFFER:
        # h, w, ch = sample_dict[b].shape()
        sample_data = sample_dict[b]
        h, w, ch = sample_data.shape
        total_ch += ch
        ch_BUFFER.append(ch)

    input_buffer = np.ones((total_num_imgs, h, w, total_ch), dtype=sample_data.dtype)
    ref_buffer = np.ones((total_num_imgs, h, w, 6), dtype=sample_data.dtype)

    "exr load"
    img_index = 0
    for s in range(len(input_all_features)):
        for f in range(len(input_all_features[s])):
            print(f)

            "input : network input"
            input_file_name = input_all_features[s][f]
            one_input = exr.read_all(os.path.join(DIR, SCENE[s], input_file_name), precision=load_dtype)
            start_ch = 0
            for b in range(len(BUFFER)):
                input_buffer[img_index, :, :, start_ch:start_ch + ch_BUFFER[b]] = one_input[BUFFER[b]]
                start_ch += ch_BUFFER[b]

            "ref : 오직 color 부분만"
            ref_file_name = ref_all_feature[s][f]
            one_ref = exr.read_all(os.path.join(DIR, SCENE[s], ref_file_name), precision=load_dtype)
            ref_buffer[img_index, :, :, :3] = one_ref['diffuse']
            ref_buffer[img_index, :, :, 3:] = one_ref['specular']

            img_index += 1

    if flag_saving:
        # input_saving_name = "input_buffer"
        # ref_saving_name = "ref_buffer"
        #
        # extension_name = ".npy"
        # np.save(saving_pth + input_saving_name + extension_name, input_buffer)
        # np.save(saving_pth + ref_saving_name + extension_name, ref_buffer)

        start_ch = 0
        extension_name = ".npy"
        for b in range(len(BUFFER)):
            input_saving_name = "input_" + BUFFER[b]
            np.save(saving_pth + input_saving_name + extension_name,
                    input_buffer[:, :, :, start_ch:start_ch + ch_BUFFER[b]])
            start_ch += ch_BUFFER[b]

        np.save(saving_pth + "ref_diffuse" + extension_name, ref_buffer[:, :, :, :3])
        np.save(saving_pth + "ref_specular" + extension_name, ref_buffer[:, :, :, 3:6])

        # extension_name = ".h5"
        # hf = h5py.File(saving_pth + ref_saving_name + extension_name, 'w')
        # hf.create_dataset('input', data=input_buffer)
        # hf.create_dataset('ref', data=ref_buffer)
        # hf.close()

    return input_buffer, ref_buffer


def load_normalize_one_exr_for_test(input_pth, ref_pth, BUFFER, load_dtype="HALF", color_merge=True):


    channels = {"diffuse": 3, "specular": 3, "albedo": 3, "depth": 1, "normal": 3, "diffuseVariance": 1,
                "specularVariance": 1, "albedoVariance": 1, "depthVariance": 1, "normalVariance": 1}

    input_all_buffer = exr.read_all(input_pth, precision=load_dtype)
    ref_all_buffer = exr.read_all(ref_pth, precision=load_dtype)

    # test_depth = input_all_buffer['depth']


    "저장 공간 설정 from sample"
    total_ch = 0
    ch_BUFFER = []
    for b in BUFFER:
        # h, w, ch = sample_dict[b].shape()
        sample_data = input_all_buffer[b]
        h, w, ch = sample_data.shape
        total_ch += ch
        ch_BUFFER.append(ch)

    # color
    if color_merge:
        input_buffer = np.zeros((1, h, w, total_ch - 3), dtype=sample_data.dtype)
        ref_buffer = np.zeros((1, h, w, 3), dtype=sample_data.dtype)

        input_buffer[0, :, :, :3] = norm.normalization_signed_log(input_all_buffer["diffuse"]
                                                                  + input_all_buffer["specular"])
        ref_buffer[0, :, :, :3] = norm.normalization_signed_log(ref_all_buffer["diffuse"]
                                                                  + ref_all_buffer["specular"])
        start_ch = 3
    else:
        input_buffer = np.zeros((1, h, w, total_ch), dtype=sample_data.dtype)
        ref_buffer = np.zeros((1, h, w, 6), dtype=sample_data.dtype)

        input_buffer[0, :, :, :3] = norm.normalization_signed_log(input_all_buffer["diffuse"])
        input_buffer[0, :, :, 3:6] = norm.normalization_signed_log(input_all_buffer["specular"])
        ref_buffer[0, :, :, :3] = norm.normalization_signed_log(ref_all_buffer["diffuse"])
        ref_buffer[0, :, :, 3:6] = norm.normalization_signed_log(ref_all_buffer["specular"])
        start_ch = 6

    # g-buffer
    for b in range(len(BUFFER)):
        if BUFFER[b] == 'depth':
            input_buffer[:, :, :, start_ch:start_ch + channels[BUFFER[b]]] = norm.normalize_depth_1ch_v1(
                input_all_buffer[BUFFER[b]])
            start_ch += channels[BUFFER[b]]
        elif BUFFER[b] == 'normal':
            input_buffer[:, :, :, start_ch:start_ch + channels[BUFFER[b]]] = norm.normalize_normal(
                input_all_buffer[BUFFER[b]])
            start_ch += channels[BUFFER[b]]
        elif BUFFER[b] == 'albedo':
            input_buffer[:, :, :, start_ch:start_ch + channels[BUFFER[b]]] = input_all_buffer[BUFFER[b]]
            start_ch += channels[BUFFER[b]]

    return input_buffer, ref_buffer




def get_npy_tungsten_and_normalize_v1(DIR, BUFFER, color_merge=True):
    """
    input : pth, mini_batch
    output : 저장된 npy 파일
    특징 1 : 가장 기본적인 load 함수
    특징 2 : npy의 구성은 color(diffuse, specular) G-buffer 이렇게 나눠지는 것을 load. 그리고 float 16임
    특징 3 : 기존에 했던 방식대로 여기서도 normalization을 진행이 됨.
    """
    # pth = E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/npy/1. train/

    "채널 크기"
    channels = {"diffuse": 3, "specular": 3, "albedo": 3, "depth": 1, "normal": 3, "diffuseVariance": 1,
                "specularVariance": 1, "albedoVariance": 1, "depthVariance": 1, "normalVariance": 1}


    "저장 공간 셋팅"
    total_CH = 0
    for b in range(len(BUFFER)):
        total_CH += channels[BUFFER[b]]

    sample = np.load(DIR + "input_" + BUFFER[0] + ".npy")

    N, H, W, initial_ch = sample.shape

    if color_merge:
        input_buffer = np.ones((N, H, W, total_CH - 3), dtype=sample.dtype)
        ref_buffer = np.ones((N, H, W, 3), dtype=sample.dtype)
    else:
        input_buffer = np.ones((N, H, W, total_CH), dtype=sample.dtype)
        ref_buffer = np.ones((N, H, W, 6), dtype=sample.dtype)

    del sample
    
    "load"
    # color
    if color_merge:
        input_buffer[:, :, :, :3] = norm.normalization_signed_log(np.load(DIR + "input_diffuse" + ".npy")
                                                                  + np.load(DIR + "input_specular" + ".npy"))
        ref_buffer[:, :, :, :3] = norm.normalization_signed_log(np.load(DIR + "ref_diffuse" + ".npy")
                                                                + np.load(DIR + "ref_specular" + ".npy"))
        start_ch = 3
    else:
        input_buffer[:, :, :, :3] = norm.normalization_signed_log(np.load(DIR + "input_diffuse" + ".npy"))
        input_buffer[:, :, :, 3:6] = norm.normalization_signed_log(np.load(DIR + "input_specular" + ".npy"))
        ref_buffer[:, :, :, :3] = norm.normalization_signed_log(np.load(DIR + "ref_diffuse" + ".npy"))
        ref_buffer[:, :, :, 3:6] = norm.normalization_signed_log(np.load(DIR + "ref_specular" + ".npy"))
        start_ch = 6

    # g-buffer
    # BUFFER.remove('diffuse')
    # BUFFER.remove('specular')
    for b in range(len(BUFFER)):
        if BUFFER[b] == 'depth':
            input_buffer[:, :, :, start_ch:start_ch + channels[BUFFER[b]]] = norm.normalize_depth_1ch_v1(np.load(
                DIR + "input_" + BUFFER[b] + ".npy"))
            start_ch += channels[BUFFER[b]]
        elif BUFFER[b] == 'normal':
            input_buffer[:, :, :, start_ch:start_ch + channels[BUFFER[b]]] = norm.normalize_normal(np.load(
                DIR + "input_" + BUFFER[b] + ".npy"))
            start_ch += channels[BUFFER[b]]
        elif BUFFER[b] == 'albedo':
            input_buffer[:, :, :, start_ch:start_ch + channels[BUFFER[b]]] = np.load(
                DIR + "input_" + BUFFER[b] + ".npy")
            start_ch += channels[BUFFER[b]]

    return input_buffer, ref_buffer



"Adv MC GAN 전용"
def get_npy_tungsten_for_AdvMC(DIR, color_type="diffuse"):
    """
    input : pth, mini_batch
    output : 저장된 npy 파일
    특징 1 : AdvMC에 특화된 데이터 load
    특징 2 : 일단은 diffuse, specular, albedo, depth, normal 각각 나오게 됨.
    특징 3 : nomalization도 안 함.
    """
    # pth = E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/npy/1. train/

    "채널 크기"

    if color_type == "diffuse":
        input_dif_or_spe = np.load(DIR + "input_diffuse" + ".npy")
        ref_dif_or_spe = np.load(DIR + "ref_diffuse" + ".npy")
    else:
        input_dif_or_spe = np.load(DIR + "input_specular" + ".npy")
        ref_dif_or_spe = np.load(DIR + "ref_specular" + ".npy")

    input_albedo = np.load(DIR + "input_albedo" + ".npy")
    input_depth = np.load(DIR + "input_depth" + ".npy")
    input_normal = np.load(DIR + "input_normal" + ".npy")


    return input_dif_or_spe, input_albedo, input_depth, input_normal, ref_dif_or_spe



def make_DB_from_tungsten(DIR, SCENE, BUFFER, OUT_PTH="tmp",
                            input_endswith="00128spp.exr", ref_endswith="08192spp.exr", mini_batch=False):
    """
    input : Dir = scene 폴더들이 있는 상위 폴더 위치, SCENE = 가져올 scene, BUFFER = 특정 feature 버퍼 이름
    output : torch.data 또는 h5py로 각 feature들이 나오로독 함.
    특징 1 : 위에서 언급을 했듯이 기존의 이름을 유지를 하고 데이터 형태를 바꿈.
    특징 2 : torch.data나 h5py로 바꿀 수 있게 함.
    특징 3 : 하나 하나의 이미지 그리고 피처를 갖고 저장이 되도록 함. 전체 이미지를 묶어서 하질 않음.
    특징 4 : FLOAT 32로 저장

    h5py 특징 : h, w, ch의 형태로 저장이 되고 'data'라고 tag가 달려 있음.
    torch.data 특징 : torch에 맞게 ch, h, w의 형태로 저장이 되고 마찬가지로 'data'라는 tag가 있음.

    """

    # DIR = 'D:/Tunsten_deep_learning_denoising_dataset/deep_learning_denoising/renderings'
    # SCENE = ['bathroom2', 'car2', 'classroom', 'house', 'room2', 'room3', 'spaceship', 'staircase']
    # BUFFER = ['diffuse', 'specular', 'albedo', 'depth', 'normal']

    "path 불러오기"
    # OUT_PTH = "E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/img_by_img/torch_data/"
    input_all_features = []  # [scene, files]
    ref_all_feature = []

    total_num_imgs = 0

    for i in range(len(SCENE)):
        # ALL_SCENES.append(os.path.join(DIR, SCENE[i]))

        files = os.listdir(os.path.join(DIR, SCENE[i]))

        input_all_features_one_scene = [file for file in files if file.endswith(input_endswith)]
        ref_all_features_one_scene = [file for file in files if file.endswith(ref_endswith)]

        if not mini_batch:
            input_all_features.append(input_all_features_one_scene)
            ref_all_feature.append(ref_all_features_one_scene)
        else:
            input_all_features.append(input_all_features_one_scene[:5])
            ref_all_feature.append(ref_all_features_one_scene[:5])
        total_num_imgs += len(input_all_features_one_scene)


    "exr load"
    for s in range(len(input_all_features)):

        # scene 마다의 폴더 만들기
        scene_folder_pth = os.path.join(OUT_PTH, SCENE[s])
        if not os.path.exists(scene_folder_pth):
            os.mkdir(scene_folder_pth)

        for f in range(len(input_all_features[s])):
            print(f)

            "input : network input"
            input_file_name = input_all_features[s][f]
            one_input = exr.read_all(os.path.join(DIR, SCENE[s], input_file_name))
            out_pth_for_one_input = os.path.join(scene_folder_pth, input_file_name.split(".")[0])

            ref_file_name = ref_all_feature[s][f]
            one_ref = exr.read_all(os.path.join(DIR, SCENE[s], ref_file_name))
            out_pth_for_one_ref = os.path.join(scene_folder_pth, ref_file_name.split(".")[0])


            for b in range(len(BUFFER)):
                one_input_one_feature = one_input[BUFFER[b]]
                out_pth_for_input_feature = out_pth_for_one_input + "_" + BUFFER[b]

                one_ref_one_feature = one_ref[BUFFER[b]]
                out_pth_for_ref_feature = out_pth_for_one_ref + "_" + BUFFER[b]

                "saving h5py"
                # input
                hf = h5py.File(out_pth_for_input_feature + ".h5", 'w')
                hf.create_dataset('data', data=one_input_one_feature)
                hf.close()

                # ref
                hf = h5py.File(out_pth_for_ref_feature + ".h5", 'w')
                hf.create_dataset('data', data=one_ref_one_feature)
                hf.close()

                "saving torch.data"
                # input
                # one_input_one_feature_torch = one_input_one_feature.transpose((2, 0, 1))
                # one_input_one_feature_torch = torch.from_numpy(one_input_one_feature_torch)
                # torch.save({'data': one_input_one_feature_torch}, out_pth_for_input_feature)

                # ref
                # one_ref_one_feature_torch = one_ref_one_feature.transpose((2, 0, 1))
                # one_ref_one_feature_torch = torch.from_numpy(one_ref_one_feature_torch)
                # torch.save({'data': one_ref_one_feature_torch}, out_pth_for_ref_feature)

                aa=1



def get_all_pth_from_tungsten_torch_data(DIR, SCENE, BUFFER, input_spp="00128spp", ref_spp="08192spp", mini_batch=False,
                                         img_by_img_type=None):
    """
    input : Dir = scene 폴더들이 있는 상위 폴더 위치. 그 안에는 torch.data가 있음.
    output : input_buffer_pth, ref_buffer_pth (torch data에 해당하는 path들).
    특징 1 : dict 형태로 feature 마다의 path 들을 내주게 됨.

    torch.data 특징 : torch에 맞게 ch, h, w의 형태로 저장이 되고 마찬가지로 'data'라는 tag가 있음.
    output dict 특징 : {"diffuse" : diffuse pths..., "specular" : specular pths...., ..}

    ! 문제 발견 !

    """

    input_buffer_dict = {}
    ref_buffer_dict = {}

    for i in range(len(BUFFER)):

        input_one_feature_list = []
        ref_one_feature_list = []

        for j in range(len(SCENE)):
            files = os.listdir(os.path.join(DIR, SCENE[j]))

            if img_by_img_type == "h5":
                input_endswith = input_spp + "_" + BUFFER[i] + ".h5"
                ref_endswith = ref_spp + "_" + BUFFER[i] + ".h5"
            else:
                input_endswith = input_spp + "_" + BUFFER[i]
                ref_endswith = ref_spp + "_" + BUFFER[i]

            input_all_features_one_scene = [os.path.join(DIR, SCENE[j], file) for file in files if file.endswith(input_endswith)]
            ref_all_features_one_scene = [os.path.join(DIR, SCENE[j], file) for file in files if file.endswith(ref_endswith)]

            # input_one_feature_list.append(input_all_features_one_scene)
            # ref_one_feature_list.append(ref_all_features_one_scene)

            input_all_features_one_scene.sort()
            ref_all_features_one_scene.sort()

            if not mini_batch:
                input_one_feature_list += input_all_features_one_scene
                ref_one_feature_list += ref_all_features_one_scene
            else:
                input_one_feature_list += input_all_features_one_scene[:2]
                ref_one_feature_list += ref_all_features_one_scene[:2]

        input_buffer_dict[BUFFER[i]] = input_one_feature_list
        ref_buffer_dict[BUFFER[i]] = ref_one_feature_list

    return input_buffer_dict, ref_buffer_dict




# if __name__ == "__main__":
        # load_exrs_from_tungsten('D:/Tunsten_deep_learning_denoising_dataset/deep_learning_denoising/renderings',
        #                         ['bathroom2', 'car2', 'classroom', 'house', 'room2', 'room3', 'spaceship', 'staircase'],
        #                         ['diffuse', 'specular', 'albedo', 'depth', 'normal'],
        #                         mini_batch=False,
        #                         flag_saving=True,
        #                         saving_pth="E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/npy/",
        #                         input_endswith="00128spp.exr", ref_endswith="08192spp.exr")

        # # from exp computer tungsten
        # load_exrs_from_tungsten('F:/DH/history/WLS_DL/From_exp_computer/test_data',
        #                         ['bathroom2', 'car', 'kitchen', 'living-room', 'living-room-3', 'veach-ajar'],
        #                         ['diffuse', 'specular', 'albedo', 'depth', 'normal'],
        #                         mini_batch=False,
        #                         flag_saving=True,
        #                         saving_pth="E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/npy/",
        #                         input_endswith="100spp.exr", ref_endswith="64kspp.exr",
        #                         load_dtype="HALF")

        # buffer_list = ['input_diffuse', 'input_specular', 'input_albedo', 'input_depth', 'input_normal',
        #                'ref_diffuse', 'ref_specular']
        # test_pth = "E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/npy/2. test"
        # mini_pth = "E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/npy/3. mini_batch"
        # saving_pth = "E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/npy"
        # for b in buffer_list:
        #     buffer_test = np.load(test_pth + "/" + b + ".npy")
        #     buffer_mini = np.load(mini_pth + "/" + b + ".npy")
        #     np.save(saving_pth + "/" + b + ".npy", np.concatenate((buffer_test, buffer_mini), axis=0))

        #
        # get_npy_tungsten_and_normalize_v1("E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/npy/1. train/",
        #                                   ['diffuse', 'specular', 'albedo', 'depth', 'normal'])



        # for train
        # make_DB_from_tungsten('D:/Tunsten_deep_learning_denoising_dataset/deep_learning_denoising/renderings',
        #                         ['bathroom2', 'car2', 'classroom', 'house', 'room2', 'room3', 'spaceship', 'staircase'],
        #                         ['diffuse', 'specular', 'albedo', 'depth', 'normal', 'diffuseVariance', 'specularVariance',
        #                          'albedoVariance', 'depthVariance', 'normalVariance'],
        #                       #  "E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/img_by_img/torch_data/1. train/",
        #                        "E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/img_by_img/h5py_data/1. train/",
        #                         input_endswith="00128spp.exr", ref_endswith="08192spp.exr", mini_batch=False)

        # for test
        # make_DB_from_tungsten('E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/tungsten_test_scenes',
        #                       ['bathroom2', 'car', 'kitchen', 'living-room', 'living-room-3', 'veach-ajar', 'curly-hair', 'staircase2',
        #                        'glass-of-water', 'teapot-full'],
        #                       ['diffuse', 'specular', 'albedo', 'depth', 'normal', 'diffuseVariance',
        #                        'specularVariance',
        #                        'albedoVariance', 'depthVariance', 'normalVariance'],
        #                       # "E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/img_by_img/torch_data/2. test/",
        #                       "E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/img_by_img/h5py_data/2. test/",
        #                       input_endswith="100spp.exr", ref_endswith="64kspp.exr")


        # train
        # get_all_pth_from_tungsten_torch_data("E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/img_by_img/torch_data/1. train/",
        #                                      ['bathroom2', 'car2', 'classroom', 'house', 'room2', 'room3', 'spaceship', 'staircase'],
        #                                      ['diffuse', 'specular', 'albedo', 'depth', 'normal', 'diffuseVariance',
        #                                       'specularVariance', 'albedoVariance', 'depthVariance', 'normalVariance'],
        #                                      input_spp="00128spp", ref_spp="08192spp", mini_batch=True
        #                                      )

        # test
        # get_all_pth_from_tungsten_torch_data(
        #     "E:/Work_space/CG_MRF_reconstruction_code/Adaptive_PR_recons_project/DB/WLS_DL_DB/img_by_img/torch_data/2. test/",
        #     ['bathroom2', 'car', 'kitchen', 'living-room', 'living-room-3', 'veach-ajar', 'curly-hair', 'staircase2', 'glass-of-water', 'teapot-full'],
        #     ['diffuse', 'specular', 'albedo', 'depth', 'normal', 'diffuseVariance',
        #      'specularVariance', 'albedoVariance', 'depthVariance', 'normalVariance'],
        #     input_spp="100spp", ref_spp="64kspp"
        #     )
