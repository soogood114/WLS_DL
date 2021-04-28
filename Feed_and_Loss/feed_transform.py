import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data
import Data.exr as exr
import Data.normalization as norm

"""*******************************************************************************"""
"feed_transform.py에 대한 정보"
" network feed를 하기 앞서 patch를 만들고 transform을 하여 torch tensor를 만드는 함수들로 구성"
" 즉, 목저은 patch generation, data augmentation, torch tensor generation 이다."
"""*******************************************************************************"""


class RandomCrop(object):
    """
    Args:
        가장 기본적인 RandomCrop 함수.
    """

    def     __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        input, target = sample['input'], sample['target']

        h, w = input.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        input = input[top: top + new_h,
                left: left + new_w]

        target = target[top: top + new_h,
                 left: left + new_w]

        return {'input': input, 'target': target}


class RandomCrop_stack_with_design(object):
    """
    input : output_size, tile_size, stack_img_sample(input, design, GT)
    output : stack_patch_sample
    feature : 한 마디로 patch를 random하게 각 이미지 마다 한 장을 만들어 내는 용도임.
    """

    def __init__(self, output_size, tile_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.tile_size = tile_size

    def __call__(self, sample):
        input, design, target = sample['input'], sample['design'], sample['target']

        h_d, w_d = input.shape[:2]
        new_h_d, new_w_d = self.output_size

        new_h_d //= self.tile_size
        new_w_d //= self.tile_size

        top = np.random.randint(0, h_d - new_h_d)
        left = np.random.randint(0, w_d - new_w_d)

        input = input[top: top + new_h_d,
                left: left + new_w_d]

        design = design[top: top + new_h_d,
                left: left + new_w_d]

        target = target[top: top + new_h_d,
                 left: left + new_w_d]

        return {'input': input, 'design': design, 'target': target}


class RandomCrop_img_stack_with_design(object):
    """
    input : output_size, tile_size, stack_img_sample(input, design, GT)
    output : stack_patch_sample
    feature #1 : 한 마디로 patch를 random하게 각 이미지 마다 한 장을 만들어 내는 용도임.
    feature #2 : img형태의 input과 stack 형태의 design을 대상으로 함.
    feature #3 : 한마디로 img와 stack 모두를 다룸
    """

    def __init__(self, output_size, tile_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.tile_size = tile_size

    def __call__(self, sample):
        input, design, target = sample['input'], sample['design'], sample['target']

        h_d, w_d = input.shape[:2]
        new_h, new_w = self.output_size

        new_h_d = new_h // self.tile_size
        new_w_d = new_w // self.tile_size

        top_d = np.random.randint(0, h_d - new_h_d)
        left_d = np.random.randint(0, w_d - new_w_d)

        top = top_d * self.tile_size
        left = left_d * self.tile_size

        input = input[top: top + new_h,
                left: left + new_w]

        target = target[top: top + new_h,
                left: left + new_w]

        design = design[top_d: top_d + new_h_d,
                 left_d: left_d + new_w_d]

        return {'input': input, 'design': design, 'target': target}


class RandomFlip(object):
    """
    Args:
        horizontal과 vertical 방향으로 각각 0.5의 확률로 패치를 돌린다.
    """

    def __init__(self, multi_crop=False):
        self.multi_crop = multi_crop


    def __call__(self, sample):
        input, target = sample['input'], sample['target']
        # H, W, C

        prob_vertical_flip = np.random.randint(0, 100)
        prob_horizontal_flip = np.random.randint(0, 100)

        if not self.multi_crop:
            axis_index = 0
        else:
            axis_index = 1

        if prob_vertical_flip > 50:
            input = np.flip(input, axis=axis_index)
            target = np.flip(target, axis=axis_index)

        if prob_horizontal_flip > 50:
            input = np.flip(input, axis=axis_index + 1)
            target = np.flip(target, axis=axis_index + 1)

        return {'input': input.copy(), 'target': target.copy()}

class RandomFlip_with_design(object):
    """
    input : patch_img_sample(input, design, GT)
    output : rotated patch_img_sample(input, design, GT) according to h and w
    feature #1 : h w 각각에 대해 50% 확률로 뒤집어 놓음.
    !!! 현재 문제가 있음을 확인
    """

    def __init__(self, multi_crop=False):
        self.multi_crop = multi_crop

    def __call__(self, sample):
        input, design, target = sample['input'], sample['design'], sample['target']
        # H, W, C

        prob_vertical_flip = np.random.randint(0, 100)
        prob_horizontal_flip = np.random.randint(0, 100)

        if not self.multi_crop:
            axis_index = 0
        else:
            axis_index = 1

        if prob_vertical_flip > 50:
            input = np.flip(input, axis=axis_index)
            design = np.flip(design, axis=axis_index)
            target = np.flip(target, axis=axis_index)

        if prob_horizontal_flip > 50:
            input = np.flip(input, axis=axis_index + 1)
            design = np.flip(design, axis=axis_index + 1)
            target = np.flip(target, axis=axis_index + 1)

        return {'input': input.copy(), 'design': design.copy(), 'target': target.copy()}


class PermuteColor(object):
    """
    Args:
        color, albedo, diff의 color ch를 permutation을 한다.
    """

    def __init__(self, multi_crop=False):

        self.color_start_index = 0
        self.albedo_start_index = 3

        self.color_iter = 1
        self.albedo_iter = 1

        self.multi_crop = multi_crop

    def __call__(self, sample):
        input, target = sample['input'], sample['target']
        # H, W, C

        if not self.multi_crop:
            _, _, all_ch = input.shape
        else:
            _, _, _, all_ch = input.shape

        permute_3ch = np.random.permutation(3)
        permute_all = np.arange(all_ch)

        color_start_index = self.color_start_index
        albedo_start_index = self.albedo_start_index

        # color permutation
        for i in range(self.color_iter):
            start = color_start_index
            end = start + 3
            permute_all[start:end] = permute_3ch + start
            color_start_index += 3

        # albedo permutation
        for i in range(self.albedo_iter):
            start = albedo_start_index
            end = start + 3
            permute_all[start:end] = permute_3ch + start
            albedo_start_index += 3

        if not self.multi_crop:
            input = input[:, :, permute_all]
            target = target[:, :, permute_3ch]
        else:
            input = input[:, :, :, permute_all]
            target = target[:, :, :, permute_3ch]

        return {'input': input, 'target': target}




class ToTensor(object):
    """numpy array를 tensor(torch)로 변환 시켜줍니다."""
    """ transform.ToTensor와는 다르게 dic의 형태로 입력을 받아낸다."""

    def __init__(self, multi_crop=False):
        self.multi_crop = multi_crop

    def __call__(self, sample):
        input, target = sample['input'], sample['target']

        # swap color axis because
        # numpy image: H x W x C or N x H x W x C
        # torch image: C X H X W or N x C X H X W
        if not self.multi_crop:
            input = input.transpose((2, 0, 1))
            target = target.transpose((2, 0, 1))
        else:
            input = input.transpose((0, 3, 1, 2))
            target = target.transpose((0, 3, 1, 2))

        return {'input': torch.from_numpy(input),
                'target': torch.from_numpy(target)}


class ToTensor_stack_with_design(object):
    """
    input : numpy patch_img_sample(input, design, GT)
    output : torch tensor patch_img_sample(input, design, GT)
    feature #1 : 단순히 numpy patch를 torch tensor로 바꿈.
    feature #2 : torch tensor에 맞춰 N C H W 형태로 바꾸고 아직은 GPU에 올라가질 않음.
    """

    def __init__(self, multi_crop=False, double_to_float=True):
        self.multi_crop = multi_crop
        self.double_to_float = double_to_float

    def __call__(self, sample):
        input, design, target = sample['input'], sample['design'], sample['target']

        # input : (multi_crop) h w  tile_size c
        # target : (multi_crop) h w  tile_size c

        # swap color axis because
        # numpy image: H x W x C or N x H x W x C
        # torch image: C X H X W or N x C X H X W

        if not self.multi_crop:
            input = input.transpose((2, 3, 0, 1))
            design = design.transpose((2, 3, 0, 1))
            target = target.transpose((2, 3, 0, 1))
        else:
            input = input.transpose((0, 3, 4, 1, 2))
            design = design.transpose((0, 3, 4, 1, 2))
            target = target.transpose((0, 3, 4, 1, 2))

        return {'input': torch.from_numpy(input).to(dtype=torch.float),
                'design': torch.from_numpy(design).to(dtype=torch.float),
                'target': torch.from_numpy(target).to(dtype=torch.float)}

class ToTensor_img_stack_with_design(object):
    """
    input : numpy patch_img_sample(input, design, GT)
    output : torch tensor patch_img_sample(input, design, GT)
    feature #1 : 단순히 numpy patch를 torch tensor로 바꿈.
    feature #2 : torch tensor에 맞춰 N C H W 형태로 바꾸고 아직은 GPU에 올라가질 않음.
    feature #3 : img형태의 input과 stack 형태의 design을 대상으로 함.
    feature #4 : 한마디로 img와 stack 모두를 다룸
    """

    def __init__(self, multi_crop=False, double_to_float=True):
        self.multi_crop = multi_crop
        self.double_to_float = double_to_float

    def __call__(self, sample):
        input, design, target = sample['input'], sample['design'], sample['target']

        # input : (multi_crop) h w  tile_size c
        # target : (multi_crop) h w  tile_size c

        # swap color axis because
        # numpy image: H x W x C or N x H x W x C
        # torch image: C X H X W or N x C X H X W

        if not self.multi_crop:
            input = input.transpose((2, 0, 1))
            design = design.transpose((2, 3, 0, 1))
            target = target.transpose((2, 0, 1))
        else:
            input = input.transpose((0, 3, 1, 2))
            design = design.transpose((0, 3, 4, 1, 2))
            target = target.transpose((0, 3, 1, 2))

        return {'input': torch.from_numpy(input).to(dtype=torch.float),
                'design': torch.from_numpy(design).to(dtype=torch.float),
                'target': torch.from_numpy(target).to(dtype=torch.float)}


""" For AdvMC"""

class GAN_RandomCrop(object):
    """
    Args:
        가장 기본적인 RandomCrop 함수.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # input, target = sample['input'], sample['target']

        in_diff_or_spec = sample['in_diff_or_spec']
        in_albedo = sample['in_albedo']
        in_depth = sample['in_depth']
        in_normal = sample['in_normal']

        ref_diff_or_spec = sample['ref_diff_or_spec']


        h, w = in_diff_or_spec.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)


        in_diff_or_spec = in_diff_or_spec[top: top + new_h, left: left + new_w]
        in_albedo = in_albedo[top: top + new_h, left: left + new_w]
        in_depth = in_depth[top: top + new_h, left: left + new_w]
        in_normal = in_normal[top: top + new_h, left: left + new_w]

        ref_diff_or_spec = ref_diff_or_spec[top: top + new_h, left: left + new_w]

        return {'in_diff_or_spec': in_diff_or_spec, 'in_albedo': in_albedo, 'in_depth': in_depth,
                  'in_normal': in_normal, 'ref_diff_or_spec': ref_diff_or_spec}

class GAN_RandomFlip(object):
    """
    Args:
        horizontal과 vertical 방향으로 각각 0.5의 확률로 패치를 돌린다.
    """

    def __init__(self, multi_crop=False):
        self.multi_crop = multi_crop


    def __call__(self, sample):
        # input, target = sample['input'], sample['target']
        # H, W, C

        in_diff_or_spec = sample['in_diff_or_spec']
        in_albedo = sample['in_albedo']
        in_depth = sample['in_depth']
        in_normal = sample['in_normal']

        ref_diff_or_spec = sample['ref_diff_or_spec']


        prob_vertical_flip = np.random.randint(0, 100)
        prob_horizontal_flip = np.random.randint(0, 100)

        if not self.multi_crop:
            axis_index = 0
        else:
            axis_index = 1

        if prob_vertical_flip > 50:
            in_diff_or_spec = np.flip(in_diff_or_spec, axis=axis_index).copy()
            in_albedo = np.flip(in_albedo, axis=axis_index).copy()
            in_depth = np.flip(in_depth, axis=axis_index).copy()
            in_normal = np.flip(in_normal, axis=axis_index).copy()
            ref_diff_or_spec = np.flip(ref_diff_or_spec, axis=axis_index).copy()

        if prob_horizontal_flip > 50:
            in_diff_or_spec = np.flip(in_diff_or_spec, axis=axis_index + 1).copy()
            in_albedo = np.flip(in_albedo, axis=axis_index + 1).copy()
            in_depth = np.flip(in_depth, axis=axis_index + 1).copy()
            in_normal = np.flip(in_normal, axis=axis_index + 1).copy()
            ref_diff_or_spec = np.flip(ref_diff_or_spec, axis=axis_index + 1).copy()

        return {'in_diff_or_spec': in_diff_or_spec, 'in_albedo': in_albedo, 'in_depth': in_depth,
                  'in_normal': in_normal, 'ref_diff_or_spec': ref_diff_or_spec}



class GAN_PermuteColor(object):
    """
    Args:
        color, albedo, diff의 color ch를 permutation을 한다.
    """

    def __init__(self, multi_crop=False):

        self.color_start_index = 0
        self.albedo_start_index = 3

        self.color_iter = 1
        self.albedo_iter = 1

        self.multi_crop = multi_crop

    def __call__(self, sample):
        # input, target = sample['input'], sample['target']
        # H, W, C

        in_diff_or_spec = sample['in_diff_or_spec']
        in_albedo = sample['in_albedo']
        in_depth = sample['in_depth']
        in_normal = sample['in_normal']

        ref_diff_or_spec = sample['ref_diff_or_spec']

        permute_3ch = np.random.permutation(3)


        if not self.multi_crop:
            in_diff_or_spec = in_diff_or_spec[:, :, permute_3ch]
            in_albedo = in_albedo[:, :, permute_3ch]
            ref_diff_or_spec = ref_diff_or_spec[:, :, permute_3ch]
        else:
            in_diff_or_spec = in_diff_or_spec[:, :, :, permute_3ch]
            in_albedo = in_albedo[:, :, :, permute_3ch]
            ref_diff_or_spec = ref_diff_or_spec[:, :, :, permute_3ch]

        return {'in_diff_or_spec': in_diff_or_spec, 'in_albedo': in_albedo, 'in_depth': in_depth,
                  'in_normal': in_normal, 'ref_diff_or_spec': ref_diff_or_spec}



# class GAN_Norm_and_Concat(object):
#     """
#     Args:
#         color, albedo, diff의 color ch를 permutation을 한다.
#     """
#
#     def __init__(self, color_type="diffuse", multi_crop=False):
#         self.color_type = color_type
#         self.multi_crop = multi_crop
#
#     def __call__(self, sample):
#         # input, target = sample['input'], sample['target']
#         # H, W, C
#
#
#
#         in_diff_or_spec = sample['in_diff_or_spec']
#         in_albedo = sample['in_albedo']
#         in_depth = sample['in_depth']
#         in_normal = sample['in_normal']
#
#         ref_diff_or_spec = sample['ref_diff_or_spec']
#
#         h, w = in_diff_or_spec.shape[:2]
#
#         if self.color_type == "diffuse":
#             # DIFFUSE
#             in_diff_or_spec[in_diff_or_spec < 0.0] = 0.0
#             in_diff_or_spec = in_diff_or_spec / (in_albedo + 0.00316)
#             in_diff_or_spec = self.LogTransform(in_diff_or_spec)
#
#             ref_diff_or_spec[ref_diff_or_spec < 0.0] = 0.0
#             in_diff_or_spec = ref_diff_or_spec / (ref_albedo + 0.00316)
#             in_diff_or_spec = self.LogTransform(in_diff_or_spec)
#
#
#             in_normal = np.nan_to_num(in_normal)
#         else:
#             # SPECULAR
#             in_diff_or_spec[in_diff_or_spec < 0.0] = 0.0
#             in_diff_or_spec = self.LogTransform(in_diff_or_spec)
#
#             in_normal = np.nan_to_num(in_normal)
#             in_normal = (in_normal + 1.0) * 0.5
#             in_normal = np.maximum(np.minimum(in_normal, 1.0), 0.0)
#
#
#         in_depth_max = np.max(in_depth)
#         in_depth /= (in_depth_max + 0.00000001)
#
#         if not self.multi_crop:
#             feautres = np.concatenate((in_normal, in_depth, in_albedo), axis=2)
#         else:
#             feautres = np.concatenate((in_normal, in_depth, in_albedo), axis=3)
#
#
#         if self.color_type == "diffuse":
#             return {
#                 "seg": feautres,
#                 "diffuse_in": in_diff_or_spec
#                 "diffuse_ref":
#             }
#         else:
#
#
#
#
#         return {'in_diff_or_spec': in_diff_or_spec, 'in_albedo': in_albedo, 'in_depth': in_depth,
#                   'in_normal': in_normal, 'ref_diff_or_spec': ref_diff_or_spec}
#
#     def LogTransform(self, data):
#         assert (np.sum(data < 0) == 0)
#         return np.log(data + 1.0)



"""*******************************************************************************"""
"""                           IMAGE BY IMAGE TRANSFORM                            """
"""*******************************************************************************"""

class IBI_RandomCrop_tensor(object):
    """
    Args:
        IBI : image by image의 줄임말
        특징 : 모든 sample은 dict 형태로 존재를 함. 그리고 indexing을 위해서 buffer list도 있음.

        !! 주의 모든 이미지의 형태는 ch h w의 형태라는 것을 잊지 말아야 함.  !!
    """

    def __init__(self, output_size, buffer_list=None):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        if buffer_list is None:
            buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal']

        self.buffer_list = buffer_list

    def __call__(self, sample):
        input_dict, target_dict = sample['input_dict'], sample['target_dict']

        h, w = input_dict[self.buffer_list[0]].shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        for buffer in self.buffer_list:
            input_dict[buffer] = input_dict[buffer][:, top: top + new_h, left: left + new_w]
            target_dict[buffer] = target_dict[buffer][:, top: top + new_h, left: left + new_w]

        return {'input_dict':  input_dict, 'target_dict': target_dict}

class IBI_RandomCrop_np(object):
    """
    Args:
        IBI : image by image의 줄임말
        특징 : 모든 sample은 dict 형태로 존재를 함. 그리고 indexing을 위해서 buffer list도 있음.

        !! 주의 모든 이미지의 형태는 ch h w의 형태라는 것을 잊지 말아야 함.  !!
    """

    def __init__(self, output_size, buffer_list=None):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        if buffer_list is None:
            buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal']

        self.buffer_list = buffer_list

    def __call__(self, sample):
        input_dict, target_dict = sample['input_dict'], sample['target_dict']

        h, w = input_dict[self.buffer_list[0]].shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        for buffer in self.buffer_list:
            input_dict[buffer] = input_dict[buffer][top: top + new_h, left: left + new_w]
            target_dict[buffer] = target_dict[buffer][top: top + new_h, left: left + new_w]

        return {'input_dict':  input_dict, 'target_dict': target_dict}




class IBI_RandomFlip_tensor(object):
    """
    Args:
        horizontal과 vertical 방향으로 각각 0.5의 확률로 패치를 돌린다.

        IBI : image by image의 줄임말
        !! 주의 모든 이미지의 형태는 ch h w의 형태라는 것을 잊지 말아야 함.  !!
    """

    def __init__(self, multi_crop=False, buffer_list=None):
        self.multi_crop = multi_crop

        if buffer_list is None:
            buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal']

        self.buffer_list = buffer_list


    def __call__(self, sample):
        input_dict, target_dict = sample['input_dict'], sample['target_dict']
        # H, W, C

        prob_vertical_flip = np.random.randint(0, 100)
        prob_horizontal_flip = np.random.randint(0, 100)

        if not self.multi_crop:
            ver_axis_index = [1]
            hor_axis_index = [2]
        else:
            ver_axis_index = [2]
            hor_axis_index = [3]

        if prob_vertical_flip > 50:
            for buffer in self.buffer_list:
                input_dict[buffer] = torch.flip(input_dict[buffer], dims=ver_axis_index)
                target_dict[buffer] = torch.flip(target_dict[buffer], dims=ver_axis_index)

        if prob_horizontal_flip > 50:
            for buffer in self.buffer_list:
                input_dict[buffer] = torch.flip(input_dict[buffer], dims=hor_axis_index)
                target_dict[buffer] = torch.flip(target_dict[buffer], dims=hor_axis_index)

        return {'input_dict':  input_dict, 'target_dict': target_dict}


class IBI_RandomFlip_np(object):
    """
    Args:
        horizontal과 vertical 방향으로 각각 0.5의 확률로 패치를 돌린다.

        IBI : image by image의 줄임말
        !! 주의 모든 이미지의 형태는 ch h w의 형태라는 것을 잊지 말아야 함.  !!
    """

    def __init__(self, multi_crop=False, buffer_list=None):
        self.multi_crop = multi_crop

        if buffer_list is None:
            buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal']

        self.buffer_list = buffer_list


    def __call__(self, sample):
        input_dict, target_dict = sample['input_dict'], sample['target_dict']
        # H, W, C

        prob_vertical_flip = np.random.randint(0, 100)
        prob_horizontal_flip = np.random.randint(0, 100)

        if not self.multi_crop:
            ver_axis_index = 0
            hor_axis_index = 1
        else:
            ver_axis_index = 1
            hor_axis_index = 2

        if prob_vertical_flip > 50:
            for buffer in self.buffer_list:
                input_dict[buffer] = np.flip(input_dict[buffer], axis=ver_axis_index).copy()
                target_dict[buffer] = np.flip(target_dict[buffer], axis=ver_axis_index).copy()

        if prob_horizontal_flip > 50:
            for buffer in self.buffer_list:
                input_dict[buffer] = np.flip(input_dict[buffer], axis=hor_axis_index).copy()
                target_dict[buffer] = np.flip(target_dict[buffer], axis=hor_axis_index).copy()

        return {'input_dict':  input_dict, 'target_dict': target_dict}



class IBI_PermuteColor_tensor(object):
    """
    Args:
        IBI : image by image의 줄임말
        !! 주의 모든 이미지의 형태는 ch h w의 형태라는 것을 잊지 말아야 함.  !!
    """

    def __init__(self, multi_crop=False, buffer_list=None):

        self.multi_crop = multi_crop

        if buffer_list is None:
            buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal']

        self.buffer_list = buffer_list

    def __call__(self, sample):
        input_dict, target_dict = sample['input_dict'], sample['target_dict']
        # H, W, C

        permute_3ch = np.random.permutation(3)

        if not self.multi_crop:
            if "diffuse" in self.buffer_list:
                input_dict["diffuse"] = input_dict["diffuse"][permute_3ch, :, :]

            if "specular" in self.buffer_list:
                input_dict["specular"] = input_dict["specular"][permute_3ch, :, :]

            if "albedo" in self.buffer_list:
                input_dict["albedo"] = input_dict["albedo"][permute_3ch, :, :]

        else:
            if "diffuse" in self.buffer_list:
                input_dict["diffuse"] = input_dict["diffuse"][:, permute_3ch, :, :]

            if "specular" in self.buffer_list:
                input_dict["specular"] = input_dict["specular"][:, permute_3ch, :, :]

            if "albedo" in self.buffer_list:
                input_dict["albedo"] = input_dict["albedo"][:, permute_3ch, :, :]

        return {'input_dict':  input_dict, 'target_dict': target_dict}

class IBI_PermuteColor_np(object):
    """
    Args:
        IBI : image by image의 줄임말
        !! 주의 모든 이미지의 형태는 ch h w의 형태라는 것을 잊지 말아야 함.  !!
    """

    def __init__(self, multi_crop=False, buffer_list=None):

        self.multi_crop = multi_crop

        if buffer_list is None:
            buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal']

        self.buffer_list = buffer_list

    def __call__(self, sample):
        input_dict, target_dict = sample['input_dict'], sample['target_dict']
        # H, W, C

        permute_3ch = np.random.permutation(3)

        if not self.multi_crop:
            if "diffuse" in self.buffer_list:
                input_dict["diffuse"] = input_dict["diffuse"][:, :, permute_3ch]

            if "specular" in self.buffer_list:
                input_dict["specular"] = input_dict["specular"][:, :, permute_3ch]

            if "albedo" in self.buffer_list:
                input_dict["albedo"] = input_dict["albedo"][:, :, permute_3ch]

        else:
            if "diffuse" in self.buffer_list:
                input_dict["diffuse"] = input_dict["diffuse"][:, :, :, permute_3ch]

            if "specular" in self.buffer_list:
                input_dict["specular"] = input_dict["specular"][:, :, :, permute_3ch]

            if "albedo" in self.buffer_list:
                input_dict["albedo"] = input_dict["albedo"][:, :, :, permute_3ch]

        return {'input_dict':  input_dict, 'target_dict': target_dict}



class IBI_Normalize_Concat_tensor_v1(object):
    """
    Args:
        IBI : image by image의 줄임말
        !! 주의 모든 이미지의 형태는 ch h w의 형태라는 것을 잊지 말아야 함.  !!

        v1의 특징
        여러가지의 noamalization이 있지만 그중 가장 기본적으로 해왔던 방식으로 함.
        그리고 tensor를 concat을 하는 것도 기존의 것과 동일하게 함.
        따라서, color merge는 기본적으로 들어가게 됨.
        그리고 diffse, specular, albedo, depth, normal 순으로 저장이 된 dict가 들어옴.
    """

    def __init__(self, multi_crop=False, buffer_list=None):
        self.multi_crop = multi_crop

        if buffer_list is None:
            buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal']

        self.buffer_list = buffer_list


    def __call__(self, sample):
        input_dict, target_dict = sample['input_dict'], sample['target_dict']
        # H, W, C

        # color merge and normalization
        input_color = norm.normalization_signed_log_tensor(input_dict["diffuse"] + input_dict["specular"])
        target_color = norm.normalization_signed_log_tensor(target_dict["diffuse"] + target_dict["specular"])

        # albedo is already normalized [0, 1]
        input_albedo = input_dict["albedo"]
        # target_albedo = target_dict["albedo"]

        # depth normalization
        if input_dict["max_depth"] == 0:
            input_depth = input_dict["depth"] / (input_dict["max_depth"] + 0.0000001)
            # target_depth = target_dict["depth"] / (target_dict["max_depth"] + 0.0000001)
        else:
            input_depth = input_dict["depth"] / (input_dict["max_depth"])
            # target_depth = target_dict["depth"] / (target_dict["max_depth"])

        # normal normalization
        input_normal = (input_dict["normal"] + 1) / 2
        # target_normal = (target_dict["normal"] + 1) / 2

        if not self.multi_crop:
            input = torch.cat((input_color, input_albedo, input_depth, input_normal), dim=0)
        else:
            input = torch.cat((input_color, input_albedo, input_depth, input_normal), dim=1)

        target = target_color

        return {'input': input, 'target': target}


class IBI_Normalize_Concat_np_v1(object):
    """
    Args:
        IBI : image by image의 줄임말
        !! 주의 모든 이미지의 형태는 ch h w의 형태라는 것을 잊지 말아야 함.  !!

        v1의 특징
        여러가지의 noamalization이 있지만 그중 가장 기본적으로 해왔던 방식으로 함.
        그리고 tensor를 concat을 하는 것도 기존의 것과 동일하게 함.
        따라서, color merge는 기본적으로 들어가게 됨.
        그리고 diffse, specular, albedo, depth, normal 순으로 저장이 된 dict가 들어옴.
    """

    def __init__(self, multi_crop=False, buffer_list=None):
        self.multi_crop = multi_crop

        if buffer_list is None:
            buffer_list = ['diffuse', 'specular', 'albedo', 'depth', 'normal']

        self.buffer_list = buffer_list


    def __call__(self, sample):
        input_dict, target_dict = sample['input_dict'], sample['target_dict']
        # H, W, C

        # color merge and normalization
        input_color = norm.normalization_signed_log(input_dict["diffuse"] + input_dict["specular"])
        target_color = norm.normalization_signed_log(target_dict["diffuse"] + target_dict["specular"])

        # albedo is already normalized [0, 1]
        input_albedo = input_dict["albedo"]
        # target_albedo = target_dict["albedo"]

        # depth normalization
        if input_dict["max_depth"] == 0:
            input_depth = input_dict["depth"] / (input_dict["max_depth"] + 0.0000001)
            # target_depth = target_dict["depth"] / (target_dict["max_depth"] + 0.0000001)
        else:
            input_depth = input_dict["depth"] / (input_dict["max_depth"])
            # target_depth = target_dict["depth"] / (target_dict["max_depth"])

        # normal normalization
        input_normal = (input_dict["normal"] + 1) / 2
        # target_normal = (target_dict["normal"] + 1) / 2

        if not self.multi_crop:
            input = np.concatenate((input_color, input_albedo, input_depth, input_normal), axis=2)
        else:
            input = np.concatenate((input_color, input_albedo, input_depth, input_normal), axis=3)

        target = target_color

        return {'input': input, 'target': target}