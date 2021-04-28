from torch.utils.data import Dataset, DataLoader
import copy
import torch
import h5py
import numpy as np

import time

"""*******************************************************************************"""
"dataset.py에 대한 정보"
" numpy 기반의 stack 버퍼를 받은 다음, feed transform을 통해서 네티워크의 feed 데이터 생성"
" 여기서 transform을 거치게 되어 패치 단위의 데이터가 나오게 됨."
"""*******************************************************************************"""


class Supervised_dataset(Dataset):
    """supervised dataset v1"""
    """ 먼저 이 class는 이미 numpy로 만들어진 객체를 input으로 받는다."""
    """ 만든 가장 큰 이유는 transform을 하기 위해서다."""

    def __init__(self, input, target, train=True, transform=None):
        """
        Args:
            input, target : N, H, W, C
            input : color + features
            GT : ref color
        """
        self.input_for_network = input
        self.GT_for_network = target
        self.transform = transform
        self.is_train = train

    def __len__(self):
        return self.input_for_network.shape[0]

    def __getitem__(self, idx):
        input = self.input_for_network[idx]  # color + features + diff
        GT = self.GT_for_network[idx]

        sample = {'input': input, 'target': GT}

        if self.transform:
            sample = self.transform(sample)

        return sample



class Supervised_dataset_img_by_img_tensor(Dataset):
    """supervised dataset img_by_img"""
    """ 이름 그대로 image by image 형태로 들어오는 path를 받아 getitem을 하는 순간에 데이터를 load하는 형태"""
    """ 바로 위의 것과는 다르게 여기서 normalization과 concat 모두가 수행이 됨."""

    def __init__(self, input_path_dict, target_path_dict, buffer_list=None, train=True, transform=None):
        """
        Args:
            input_path_dict, target_path_dict : {"diffuse" : diffuse pths..., "specular" : specular pths...., ..}
            buffer_list 는 본격적으로 학습을 하는 과정에서 필요로 하는 텐서의 구성을 요소를 결정짓는 list
            기본적으로는 ['diffuse', 'specular', 'albedo', 'depth', 'normal']
        """
        if buffer_list is None:
            buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal']  # albedo, depth, normal

        # 각 이미지의 경로 저장
        self.input_buffer_pth = input_path_dict
        self.ref_buffer_pth = target_path_dict

        # 경로에 맞는 이미지 또는 patch를 담는 바구나와 같은 역할
        # 이 역시 buffer_list에 나온 이름대로 저장이 될 것임.
        self.input_dict = {}
        self.target_dict = {}

        self.buffer_list = buffer_list

        self.transform = transform
        

    def __len__(self):
        return len(self.input_buffer_pth["diffuse"])

    def __getitem__(self, idx):

        # dict에 load를 하여 저장
        #start = time.time()

        for buffer in self.buffer_list:
            self.input_dict[buffer] = torch.load(self.input_buffer_pth[buffer][idx])['data']
            self.target_dict[buffer] = torch.load(self.ref_buffer_pth[buffer][idx])['data']

        # end = time.time()
        # print(end - start)

        # depth 만큼은 먼저 normalization 수행하기 위해서 image 단위의 max를 저장을 함.
        if "depth" in self.buffer_list:
            self.input_dict["max_depth"] = torch.max(self.input_dict["depth"])
            self.target_dict["max_depth"] = torch.max(self.target_dict["depth"])

        sample = {'input_dict':  self.input_dict, 'target_dict': self.target_dict}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Supervised_dataset_img_by_img_np(Dataset):
    """supervised dataset img_by_img"""
    """ 이름 그대로 image by image 형태로 들어오는 path를 받아 getitem을 하는 순간에 데이터를 load하는 형태"""
    """ 바로 위의 것과는 다르게 여기서 normalization과 concat 모두가 수행이 됨."""

    def __init__(self, input_path_dict, target_path_dict, buffer_list=None, train=True, transform=None):
        """
        Args:
            input_path_dict, target_path_dict : {"diffuse" : diffuse pths..., "specular" : specular pths...., ..}
            buffer_list 는 본격적으로 학습을 하는 과정에서 필요로 하는 텐서의 구성을 요소를 결정짓는 list
            기본적으로는 ['diffuse', 'specular', 'albedo', 'depth', 'normal']
        """
        if buffer_list is None:
            buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal']  # albedo, depth, normal

        # 각 이미지의 경로 저장
        self.input_buffer_pth = input_path_dict
        self.ref_buffer_pth = target_path_dict

        # 경로에 맞는 이미지 또는 patch를 담는 바구나와 같은 역할
        # 이 역시 buffer_list에 나온 이름대로 저장이 될 것임.
        self.input_dict = {}
        self.target_dict = {}

        self.buffer_list = buffer_list

        self.transform = transform

    def __len__(self):
        return len(self.input_buffer_pth["diffuse"])

    def __getitem__(self, idx):
        # dict에 load를 하여 저장
        # start = time.time()
        for buffer in self.buffer_list:
            self.input_dict[buffer] = h5py.File(self.input_buffer_pth[buffer][idx], 'r')['data'][:]
            self.target_dict[buffer] = h5py.File(self.ref_buffer_pth[buffer][idx], 'r')['data'][:]

        # end = time.time()
        # print(end-start)

        # depth 만큼은 먼저 normalization 수행하기 위해서 image 단위의 max를 저장을 함.
        if "depth" in self.buffer_list:
            self.input_dict["max_depth"] = np.max(self.input_dict["depth"])
            self.target_dict["max_depth"] = np.max(self.target_dict["depth"])

        sample = {'input_dict': self.input_dict, 'target_dict': self.target_dict}

        if self.transform:
            sample = self.transform(sample)

        return sample



class Supervised_dataset_AdvMC(Dataset):
    """Supervised_dataset_AdvMC"""
    """ AdvMC에 맞게 변형된 것."""
    """ Concat과 Norm은 다 이 안에서 이뤄짐."""
    """ !!! ref_albedo.npy의 부재로 인해 실패 !! """

    def __init__(self, in_diff_or_spec, in_al, in_de, in_nor, ref_diff_or_spec,
                 color_type="diffuse", transform=None):
        """
        Args:
            input, target : N, H, W, C
            input : color + features
            GT : ref color
        """
        self.in_diff_or_spec = in_diff_or_spec
        self.in_albedo = in_al
        self.in_depth = in_de
        self.in_normal = in_nor
        self.ref_diff_or_spec = ref_diff_or_spec

        self.color_type = color_type  # diffuse or specular


    def __len__(self):
        return self.in_diff_or_spec.shape[0]

    def __getitem__(self, idx):
        in_diff_or_spec = self.in_diff_or_spec[idx]
        in_albedo = self.in_albedo[idx]
        in_depth = self.in_depth[idx]
        in_normal = self.in_normal[idx]

        ref_diff_or_spec = self.ref_diff_or_spec[idx]


        sample = {'in_diff_or_spec': in_diff_or_spec, 'in_albedo': in_albedo, 'in_depth': in_depth,
                  'in_normal': in_normal, 'ref_diff_or_spec': ref_diff_or_spec}

        if self.transform:
            sample = self.transform(sample)

        return sample


