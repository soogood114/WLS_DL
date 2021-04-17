import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import Models.resnet as resnet


class KPCN_v1(nn.Module):
    """
    네트워크 개요
    - KPCN 또는 KPAL의 형태를 그대로 본 땀.
    - 실험을 위해서 soft max와 같은 것을 reg를 할 수 있는 방안을 on / off 가능


    주요 기능
    - reg_W_without_soft_max : 문제 없이 동작함.

    정리
    """

    def __init__(self, params, loss_fn, ch_in=10, kernel_size=3, n_layers=25, length_p_kernel=21, no_soft_max=True,
                 pad_mode=0, is_resnet=False):
        super(KPCN_v1, self).__init__()

        self.ch_in = ch_in
        self.k_size = kernel_size

        self.pad_mode = pad_mode  # 0: zero, 1: reflected, 2: circular

        self.length_p_kernel = length_p_kernel
        self.size_p_kernel = int(length_p_kernel ** 2)

        "new features"
        self.params = params
        self.no_soft_max = no_soft_max

        # loss
        self.loss_fn = loss_fn

        # setting of layers
        self.start_ch = ch_in
        self.inter_ch = 100
        self.final_ch = int(self.size_p_kernel)

        self.layers = [nn.Conv2d(self.start_ch, self.inter_ch, kernel_size, padding=(kernel_size - 1) // 2)]
        if is_resnet:
            for l in range(n_layers // 2 - 1):
                self.layers += [
                    resnet.Resblock_2d(self.inter_ch, self.inter_ch, self.inter_ch, kernel_size)
                ]
            self.feature_layers_feed = nn.Sequential(*self.layers)  # to get the feature
            self.layers_for_weights_feed = nn.Conv2d(self.inter_ch, self.final_ch, kernel_size,
                                                     padding=(kernel_size - 1) // 2)
        else:
            for l in range(n_layers - 2):
                self.layers += [
                    nn.Conv2d(self.inter_ch, self.inter_ch, kernel_size, padding=(kernel_size - 1) // 2),
                    nn.LeakyReLU()
                ]
            self.feature_layers_feed = nn.Sequential(*self.layers)  # to get the feature
            self.layers_for_weights_feed = nn.Conv2d(self.inter_ch, self.final_ch, kernel_size,
                                                     padding=(kernel_size - 1) // 2)

        # self.layers_for_weights_feed.weight.data.fill_(0.0)
        # self.layers_for_weights_feed.bias.data.fill_(1.0)


    def forward(self, input, ref, only_img_out=False):
        """
        input : B C_in H W
        design : B C_de H W

        특이하게 gt를 넣어서 loss까지 한방에 계산을 하도록 함.
        output : resulting image and loss

        """
        "INITIAL SETTING"
        b = input.size(0)
        h, w = input.size(2), input.size(3)

        "GET THE FEATURE FROM FEATURE NETWORK"
        feature = self.feature_layers_feed(input)

        "W FROM THE FEATURE"  # 여기서 메모리가 뻥튀기 됨.
        W = self.layers_for_weights_feed(feature)  # b, size_Pkernel, h, w,

        if self.no_soft_max:
            W = self.reg_W_without_soft_max(W)
        else:
            W = F.softmax(W, dim=1)


        "KERNEL REGRESSION"
        out = self.kernel_regression(input, W, test_mode=False)
        # out = out.view(b, h, w, 3).permute(0, 3, 1, 2)

        if only_img_out:
            return out.view(b, h, w, 3).permute(0, 3, 1, 2)
        else:
            return out.view(b, h, w, 3).permute(0, 3, 1, 2), self.get_loss(out, ref)



    def test_chunkwise(self, input_full, chunk_size=200):
        """
        input : full res img form buffer
        chunk : 이미지 block size
        output : resulting image and loss
        chunk 단위로 image를 쪼갤 수 있어 메모리 절감 효과
        """
        "INITIAL SETTING"
        self.pad_mode = -1  ## !! test mode에서는 꼭 해줘야함.

        b = input_full.size(0)
        h_full, w_full = input_full.size(2), input_full.size(3)

        length_p_kernel = int(self.length_p_kernel)  # length of inter tile
        size_p_kernel = length_p_kernel ** 2

        # overlapping or non overlapping
        out_full = torch.zeros((b, 3, h_full, w_full), dtype=input_full.dtype,
                                   layout=input_full.layout, device=input_full.device)

        "GET THE FEATURE FROM FEATURE NETWORK"
        feature = self.feature_layers_feed(input_full)

        "W FROM THE FEATURE"
        W_full = self.layers_for_weights_feed(feature)  # b, size_Pkernel, h, w

        if self.no_soft_max:
            W_full = self.reg_W_without_soft_max(W_full)
        else:
            W_full = F.softmax(W_full, dim=1)

        # W = W.view((b * hw), size_Pkernel)

        "PADDING INPUT FOR PREVENTING COLOR INFO. LOSSES"
        size_pad = length_p_kernel // 2
        pad = (size_pad, size_pad, size_pad, size_pad)
        input_full = nn.functional.pad(input_full, pad, mode='constant')

        "CHUNK WISE REGRESSION"
        for w_start in np.arange(0, w_full, chunk_size):
            for h_start in np.arange(0, h_full, chunk_size):

                w_end = min(w_start + chunk_size, w_full)
                h_end = min(h_start + chunk_size, h_full)

                w_start_p = w_start  # PAD
                h_start_p = h_start
                w_end_p = w_end + size_pad * 2
                h_end_p = h_end + size_pad * 2

                "SETTING"
                h_p = h_end_p - h_start_p
                w_p = w_end_p - w_start_p

                h = h_p - size_pad * 2
                w = w_p - size_pad * 2

                "CROP INPUT, REF, AND W"
                # 이미 padding이 앞서 진행됨.
                input = input_full[:, :, h_start_p:h_end_p, w_start_p:w_end_p]  # 지금 padding 된 상태

                # padding 진행 안됨.
                W = W_full[:, :, h_start:h_end, w_start:w_end]

                "KERNEL REGRESSION"
                out = self.kernel_regression(input, W, test_mode=True)

                out_full[:, :, h_start:h_end, w_start:w_end] = out.view(b, h, w, 3).permute(0, 3, 1, 2)

        return out_full

    def kernel_regression(self, input, W, test_mode=True):
        """
            input : img form buffer (b, ch_in, h, w)
            W : img form weight (b, size_Pkernel, h, w)
            test mode : on/off test mode
            특징 : KPCN 과 같이 W가 나오면 KERNEL regrssion으로 후처리를 하는 매우 중요한 함수.
        """
        "INITIAL SETTING"
        b = input.size(0)
        h, w = input.size(2), input.size(3)
        hw = h * w

        length_p_kernel = int(self.length_p_kernel)  # length of inter tile
        size_Pkernel = length_p_kernel ** 2

        "MODE SELECTION"
        if test_mode:
            # in the test mode, input is padded already.
            h -= (length_p_kernel // 2) * 2
            w -= (length_p_kernel // 2) * 2
            hw = h * w

            W = W.permute((0, 2, 3, 1)).contiguous()
            W = W.reshape((b * hw), size_Pkernel, 1)
        else:
            W = W.permute((0, 2, 3, 1)).contiguous()
            W = W.view((b * hw), size_Pkernel, 1)

        "OUTPUT"
        out = torch.zeros((b * hw, 3), dtype=input.dtype, layout=input.layout,
                              device=input.device)

        "SOLVING NORMAL EQUATION"
        for ch in range(3):
            "Y FROM INPUT"
            Y = input[:, ch, :, :].unsqueeze(1)  # b, 1, h_d, w_d
            Y = self.unfold_and_padding(Y)  # b, size_Pkernel, hw_d

            Y = Y.permute(0, 2, 1).contiguous().view(b * hw, 1, size_Pkernel)

            out_1ch = torch.bmm(Y, W)  # b * hw, 1, 1
            out[:, ch] = out_1ch[:, 0, 0]

        return out

    def get_loss(self, y_pred, y):
        "unfolded loss 때문에 따로 만듦"
        " 그래서 W 값을 이용해 loss를 구함. 이는 APR과 비슷"
        return self.loss_fn(y_pred, y.permute(0, 2, 3, 1).reshape(-1, 3))


    def unfold_and_padding(self, x):
        """
        input : x (4D)
        output : Unfolded x
        feature #1 : unfolding을 하는 함수. padding mode를 조절할 수 있음.
        """
        kernel_length = self.length_p_kernel
        if self.pad_mode > 0:
            pad = (kernel_length // 2, kernel_length // 2, kernel_length // 2, kernel_length // 2)
            if self.pad_mode == 1:
                x = nn.functional.pad(x, pad, mode='reflect')
            elif self.pad_mode == 2:
                x = nn.functional.pad(x, pad, mode='circular')
            else:
                x = nn.functional.pad(x, pad, mode='reflect')

            x_unfolded = F.unfold(x, kernel_length, padding=0)
        elif self.pad_mode == 0:  # zero padding
            # automatically zero padding
            x_unfolded = F.unfold(x, kernel_length, padding=kernel_length // 2)
        else:
            # image resolution gonna be reduced
            x_unfolded = F.unfold(x, kernel_length, padding=0)

        return x_unfolded

    def reg_W_without_soft_max(self, W, epsilon=0.000001):
        "W에서 하나의 픽셀에 해당하는 가중치의 합을 1로 고정을 하는 함수"
        " 특징 1: W안 요소들이 negative여도 상관 없음."
        " W : b, size_Pkernel, h, w"
        return W / (torch.sum(W, dim=1, keepdim=True) + epsilon)


class KPCN_for_FG_v1(nn.Module):
    """
    네트워크 개요
    - KPCN 또는 KPAL의 형태를 그대로 본 땀.
    - FG라고 표시가 된 것에 맞게 feature generation 또는 refine을 하는 것이 목표
    - input은 design에 들어가는 g-buffer로 지정됨.


    주요 기능
    - 꼭 albedo, depth, normal 순으로 저장된 7ch buffer가 들어가야 함

    정리
    """

    def __init__(self, params, ch_in=7, kernel_size=3, n_layers=8, length_p_kernel=5, no_soft_max=True,
                 pad_mode=0, is_resnet=True):
        super(KPCN_for_FG_v1, self).__init__()

        self.ch_in = ch_in
        self.k_size = kernel_size

        self.pad_mode = pad_mode  # 0: zero, 1: reflected, 2: circular

        self.length_p_kernel = length_p_kernel
        self.size_p_kernel = int(length_p_kernel ** 2)

        "new features"
        self.params = params
        self.no_soft_max = no_soft_max

        # setting of layers
        self.start_ch = ch_in
        self.inter_ch = 100
        self.final_ch = int(self.size_p_kernel)

        # feature generator
        self.layers = [nn.Conv2d(self.start_ch, self.inter_ch, kernel_size, padding=(kernel_size - 1) // 2)]
        if is_resnet:
            for l in range(n_layers // 2 - 1):
                self.layers += [
                    resnet.Resblock_2d(self.inter_ch, self.inter_ch, self.inter_ch, kernel_size)
                ]
            self.feature_layers_feed = nn.Sequential(*self.layers)  # to get the feature
        else:
            for l in range(n_layers - 2):
                self.layers += [
                    nn.Conv2d(self.inter_ch, self.inter_ch, kernel_size, padding=(kernel_size - 1) // 2),
                    nn.LeakyReLU()
                ]
            self.feature_layers_feed = nn.Sequential(*self.layers)  # to get the feature

        # albedo
        self.layers_W_albedo = [
            nn.Conv2d(self.inter_ch, self.inter_ch // 2, kernel_size, padding=(kernel_size - 1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(self.inter_ch // 2, self.final_ch, kernel_size, padding=(kernel_size - 1) // 2)
        ]
        self.W_albedo_feed = nn.Sequential(*self.layers_W_albedo)

        # depth
        self.layers_W_depth = [
            nn.Conv2d(self.inter_ch, self.inter_ch // 2, kernel_size, padding=(kernel_size - 1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(self.inter_ch // 2, self.final_ch, kernel_size, padding=(kernel_size - 1) // 2)
        ]
        self.W_depth_feed = nn.Sequential(*self.layers_W_depth)

        # normal
        self.layers_W_normal = [
            nn.Conv2d(self.inter_ch, self.inter_ch // 2, kernel_size, padding=(kernel_size - 1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(self.inter_ch // 2, self.final_ch, kernel_size, padding=(kernel_size - 1) // 2)
        ]
        self.W_normal_feed = nn.Sequential(*self.layers_W_normal)

        # self.layers_for_weights_feed.weight.data.fill_(0.0)
        # self.layers_for_weights_feed.bias.data.fill_(1.0)


    def forward(self, input):
        """
        input : B C_in H W
        design : B C_de H W
        output : resulting image and loss

        """
        "INITIAL SETTING"
        b = input.size(0)
        h, w = input.size(2), input.size(3)
        ch = self.ch_in

        "GET THE FEATURE FROM FEATURE NETWORK"
        feature = self.feature_layers_feed(input)

        "W FROM THE FEATURE"  # 여기서 메모리가 뻥튀기 됨.
        W_albedo = self.W_albedo_feed(feature)  # b, size_Pkernel, h, w,
        W_depth = self.W_depth_feed(feature)
        W_normal = self.W_normal_feed(feature)

        if self.no_soft_max:
            W_albedo = self.reg_W_without_soft_max(W_albedo)
            W_depth = self.reg_W_without_soft_max(W_depth)
            W_normal = self.reg_W_without_soft_max(W_normal)
        else:
            W_albedo = F.softmax(W_albedo, dim=1)
            W_depth = F.softmax(W_depth, dim=1)
            W_normal = F.softmax(W_normal, dim=1)


        "KERNEL REGRESSION"
        out = self.kernel_regression(input, W_albedo, W_depth, W_normal, test_mode=False)
        # out = out.view(b, h, w, 3).permute(0, 3, 1, 2)

        return out.view(b, h, w, ch).permute(0, 3, 1, 2)



    def test_chunkwise(self, input_full, chunk_size=200):
        """
        input : full res img form buffer
        chunk : 이미지 block size
        output : resulting image and loss
        chunk 단위로 image를 쪼갤 수 있어 메모리 절감 효과
        """
        "INITIAL SETTING"
        self.pad_mode = -1  ## !! test mode에서는 꼭 해줘야함.

        b = input_full.size(0)
        h_full, w_full = input_full.size(2), input_full.size(3)
        ch = self.ch_in

        length_p_kernel = int(self.length_p_kernel)  # length of inter tile
        size_p_kernel = length_p_kernel ** 2

        # overlapping or non overlapping
        out_full = torch.zeros((b, ch, h_full, w_full), dtype=input_full.dtype,
                                   layout=input_full.layout, device=input_full.device)

        "GET THE FEATURE FROM FEATURE NETWORK"
        feature = self.feature_layers_feed(input_full)

        "W FROM THE FEATURE"
        W_full = self.layers_for_weights_feed(feature)  # b, size_Pkernel, h, w

        if self.no_soft_max:
            W_full = self.reg_W_without_soft_max(W_full)
        else:
            W_full = F.softmax(W_full, dim=1)

        # W = W.view((b * hw), size_Pkernel)

        "PADDING INPUT FOR PREVENTING COLOR INFO. LOSSES"
        size_pad = length_p_kernel // 2
        pad = (size_pad, size_pad, size_pad, size_pad)
        input_full = nn.functional.pad(input_full, pad, mode='constant')

        "CHUNK WISE REGRESSION"
        for w_start in np.arange(0, w_full, chunk_size):
            for h_start in np.arange(0, h_full, chunk_size):

                w_end = min(w_start + chunk_size, w_full)
                h_end = min(h_start + chunk_size, h_full)

                w_start_p = w_start  # PAD
                h_start_p = h_start
                w_end_p = w_end + size_pad * 2
                h_end_p = h_end + size_pad * 2

                "SETTING"
                h_p = h_end_p - h_start_p
                w_p = w_end_p - w_start_p

                h = h_p - size_pad * 2
                w = w_p - size_pad * 2

                "CROP INPUT, REF, AND W"
                # 이미 padding이 앞서 진행됨.
                input = input_full[:, :, h_start_p:h_end_p, w_start_p:w_end_p]  # 지금 padding 된 상태

                # padding 진행 안됨.
                W = W_full[:, :, h_start:h_end, w_start:w_end]

                "KERNEL REGRESSION"
                out = self.kernel_regression(input, W, test_mode=True)

                out_full[:, :, h_start:h_end, w_start:w_end] = out.view(b, h, w, 3).permute(0, 3, 1, 2)

        return out_full


    def kernel_regression(self, input, W_albedo, W_depth, W_normal, test_mode=True):
        """
            input : img form buffer (b, ch_in, h, w)
            W : img form weight (b, size_Pkernel, h, w)
            test mode : on/off test mode
            특징 : KPCN 과 같이 W가 나오면 KERNEL regrssion으로 후처리를 하는 매우 중요한 함수.
        """
        "INITIAL SETTING"
        b = input.size(0)
        h, w = input.size(2), input.size(3)
        hw = h * w

        # ch_albedo = 3
        # ch_depth = 1
        # ch_normal = 3
        ch_total = 7

        length_p_kernel = int(self.length_p_kernel)  # length of inter tile
        size_Pkernel = length_p_kernel ** 2

        "MODE SELECTION"
        if test_mode:
            # in the test mode, input is padded already.
            h -= (length_p_kernel // 2) * 2
            w -= (length_p_kernel // 2) * 2
            hw = h * w

        "W processing"
        W_albedo = W_albedo.permute((0, 2, 3, 1)).contiguous()
        W_albedo = W_albedo.view((b * hw), size_Pkernel, 1)

        W_depth = W_depth.permute((0, 2, 3, 1)).contiguous()
        W_depth = W_depth.view((b * hw), size_Pkernel, 1)

        W_normal = W_normal.permute((0, 2, 3, 1)).contiguous()
        W_normal = W_normal.view((b * hw), size_Pkernel, 1)


        "OUTPUT"
        out = torch.zeros((b * hw, ch_total), dtype=input.dtype, layout=input.layout,
                              device=input.device)


        "Y FROM INPUT"
        Y = self.unfold_and_padding(input)  # b, 7 * size_Pkernel, hw_d
        Y = Y.permute(0, 2, 1).contiguous().view(b * hw, 7, size_Pkernel)

        "FOR ALBEDO, SOLVING NORMAL EQUATION"
        for ch in range(3):
            out_1ch = torch.bmm(Y[:, ch, :].unsqueeze(1), W_albedo)  # b * hw, 1, 1
            out[:, ch] = out_1ch[:, 0, 0]

        "FOR DEPTH, SOLVING NORMAL EQUATION"
        out_1ch = torch.bmm(Y[:, 3, :].unsqueeze(1), W_depth)  # b * hw, 1, 1
        out[:, 3] = out_1ch[:, 0, 0]

        "FOR ALBEDO, SOLVING NORMAL EQUATION"
        for ch in range(4, 7):
            out_1ch = torch.bmm(Y[:, ch, :].unsqueeze(1), W_normal)  # b * hw, 1, 1
            out[:, ch] = out_1ch[:, 0, 0]

        return out

    def unfold_and_padding(self, x):
        """
        input : x (4D)
        output : Unfolded x
        feature #1 : unfolding을 하는 함수. padding mode를 조절할 수 있음.
        """
        kernel_length = self.length_p_kernel
        if self.pad_mode > 0:
            pad = (kernel_length // 2, kernel_length // 2, kernel_length // 2, kernel_length // 2)
            if self.pad_mode == 1:
                x = nn.functional.pad(x, pad, mode='reflect')
            elif self.pad_mode == 2:
                x = nn.functional.pad(x, pad, mode='circular')
            else:
                x = nn.functional.pad(x, pad, mode='reflect')

            x_unfolded = F.unfold(x, kernel_length, padding=0)
        elif self.pad_mode == 0:  # zero padding
            # automatically zero padding
            x_unfolded = F.unfold(x, kernel_length, padding=kernel_length // 2)
        else:
            # image resolution gonna be reduced
            x_unfolded = F.unfold(x, kernel_length, padding=0)

        return x_unfolded

    def reg_W_without_soft_max(self, W, epsilon=0.000001):
        "W에서 하나의 픽셀에 해당하는 가중치의 합을 1로 고정을 하는 함수"
        " 특징 1: W안 요소들이 negative여도 상관 없음."
        " W : b, size_Pkernel, h, w"
        return W / (torch.sum(W, dim=1, keepdim=True) + epsilon)
