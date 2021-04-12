import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import Feed_and_Loss.loss as my_loss
import Models.resnet as resnet

import Models.net_op as net_op
import Data.exr as exr
import Data.other_tools as other_tools


class WLS_net_v1(nn.Module):
    """
    네트워크 개요
    - PR 버전의 NPR_net_stack_v2 구조를 활용함.
    - loss of unfolded data, norm in prediction window
    - 새로운 데이터의 특성 상 loss함수를 따로 안에 정의를 함..
    - DESIGN matrix를 만드는 함수도 포함. ( 당연히 grid를 만든 것도 포함. 단, intial 이 부분에서만 동작.)

    주요 기능
    - loss of unfolded : unfolded 된 상태에서 loss를 구함.
    - overlapping : 각 prediction window의 크기 만큼 prediction and overlapping between them
    - norm in prediction window : WLR이나 다른 논문들 처럼 prediction window of regression에서 [0, 1]로 맞춤.
    - refine the W generator : resnet version 가능.

    정리
    - loss of unfolded :
    - overlapping : make full res test를 만들어서 억지로 함. window가 클 경우 CPU모드를 켜야함.
    - norm in prediction window :
    - half W :

    """

    def __init__(self, params, loss_fn, ch_in=10, kernel_size=3, n_layers=25, length_p_kernel=21, epsilon=0.0001,
                 pad_mode=0, loss_type=0, kernel_accum=False, norm_in_window=False, is_resnet=False):
        super(WLS_net_v1, self).__init__()

        self.ch_in = ch_in
        self.k_size = kernel_size

        self.epsilon = epsilon

        self.pad_mode = pad_mode  # 0: zero, 1: reflected, 2: circular

        self.length_p_kernel = length_p_kernel
        self.size_p_kernel = int(length_p_kernel ** 2)

        "new features"
        self.params = params
        self.reg_order = int(self.params["grid_order"])

        self.kernel_accum = kernel_accum
        
        # grid
        if self.reg_order > 0:
            self.grid_seed_p_kernel = self.make_grid_in_p_kernel()
            self.grid_seed_p_kernel = self.grid_seed_p_kernel.unsqueeze(0)  # fit the dimension

        # overlapping
        self.norm_in_window = norm_in_window

        # loss
        self.loss_fn = loss_fn

        "중요 : 0 or other: normal pix2pix loss, 1: unfolded loss, 2: unfolded kernel loss(APR)"
        self.loss_type = loss_type
        
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
        # self.layers_for_weights_feed.bias.data.fill_(0.0)
        # self.b, self.ch_in, self.ch_de, self.h, self.w = [0, 0, 0, 0, 0]


    def forward(self, input, ref, only_img_out=False, full_res_out=True):
        """
        input : B C_in H W
        design : B C_de H W

        특이하게 gt를 넣어서 loss까지 한방에 계산을 하도록 함.
        output : resulting image and loss

        """
        "INITIAL SETTING"
        b = input.size(0)
        h, w = input.size(2), input.size(3)
        length_p_kernel = self.length_p_kernel  # length of inter tile
        size_p_kernel = length_p_kernel ** 2

        "GET THE FEATURE FROM FEATURE NETWORK"
        feature = self.feature_layers_feed(input)

        "W FROM THE FEATURE"  # 여기서 메모리가 뻥튀기 됨.
        W = self.layers_for_weights_feed(feature)  # b, size_Pkernel, h, w,
        W = F.softmax(W, dim=1)

        "KERNEL REGRESSION"
        out = self.kernel_regression(input, W, test_mode=False)

        if only_img_out == True:
            return self.make_full_res_out(out, b, h, w, length_p_kernel, W)
        else:
            if full_res_out:
                return self.make_full_res_out(out, b, h, w, length_p_kernel, W), self.get_loss(out, ref, W)
            else:
                return out, self.get_loss(out, ref, W)


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
        if self.loss_type == 1 or self.loss_type == 2:
            out_full = torch.zeros((b, 3, size_p_kernel, h_full, w_full), dtype=input_full.dtype,
                                   layout=input_full.layout, device=input_full.device)
        else:
            out_full = torch.zeros((b, 3, h_full, w_full), dtype=input_full.dtype,
                                   layout=input_full.layout, device=input_full.device)


        "GET THE FEATURE FROM FEATURE NETWORK"
        feature = self.feature_layers_feed(input_full)

        "W FROM THE FEATURE"  # 여기서 메모리가 뻥튀기 됨.
        W_full = self.layers_for_weights_feed(feature)  # b, size_Pkernel, h, w
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

                if self.loss_type == 1 or self.loss_type == 2:
                    out_full[:, :, :, h_start:h_end, w_start:w_end] =\
                        out.view(b, h, w, size_p_kernel, 3).permute(0, 4, 3, 1, 2)
                else:
                    out_full[:, :, h_start:h_end, w_start:w_end] = out.view(b, h, w, 3).permute(0, 3, 1, 2)

        if self.loss_type == 1 or self.loss_type == 2:
            return self.make_full_res_out_test_mode(out_full, b, h_full, w_full, length_p_kernel, W_full)
        else:
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
        ch_in = input.size(1)  # 10 (color + g_buffer)
        ch_de = 1 + 7 + 2 * self.reg_order
        h, w = input.size(2), input.size(3)
        hw = h * w

        length_p_kernel = int(self.length_p_kernel)  # length of inter tile
        size_Pkernel = length_p_kernel ** 2

        "UNFOLDING FOR DESIGN MATRIX"
        ones = torch.ones((b, 1, h, w), dtype=input.dtype, layout=input.layout, device=input.device)
        # n 1+7 h w
        design = torch.cat((ones, input[:, 3:, :, :]), dim=1)  # input의 3ch 이후에는 모두 g-buffer라 가정.
        design = self.unfold_and_padding(design)  # b, 8 * size_Pkernel, hw_d


        "MODE SELECTION"
        if test_mode:
            # in the test mode, input is padded already.
            h -= (length_p_kernel // 2) * 2
            w -= (length_p_kernel // 2) * 2
            hw = h * w

            W = W.permute((0, 2, 3, 1)).contiguous()
            W = W.reshape((b * hw), size_Pkernel)
        else:
            W = W.permute((0, 2, 3, 1)).contiguous()
            W = W.view((b * hw), size_Pkernel)


        "OUTPUT"
        if self.loss_type == 1 or self.loss_type == 2:
            out = torch.zeros((b * hw * size_Pkernel, 3), dtype=input.dtype, layout=input.layout,
                          device=input.device)
        else:
            out = torch.zeros((b * hw, 3), dtype=input.dtype, layout=input.layout,
                              device=input.device)


        "MAKE XT AND DOMAIN"
        design = design.permute(0, 2, 1).contiguous().view(b * hw, 1 + ch_in - 3, size_Pkernel)

        if self.norm_in_window:
            design = self.norm_in_prediction_window(design)

        # b * hw, ch_de, size_Pkernel
        if self.reg_order > 0:
            XT = torch.cat((design, self.grid_seed_p_kernel.repeat(b * hw, 1, 1)), dim=1)  # add grids
        else:
            XT = design

        # domain and X
        if self.loss_type == 1 or self.loss_type == 2:
            domain = XT.permute(0, 2, 1).contiguous()  # b * hw, ch_de, size_Pkernel
            # b * hw * size_Pkernel, 1, ch_de
            domain = domain.view(b * hw * size_Pkernel, 1, ch_de)
        else:
            domain = XT[:, :, size_Pkernel // 2].contiguous().unsqueeze(1)  # b * hw, 1, ch_de


        "XTW & XTWX"
        XTW = self.make_XTW_from_linear_W(XT, W)
        XTWX = torch.bmm(XTW, XT.permute(0, 2, 1))  # b * hw, ch_de, ch_de

        # epsilon
        regular_term = (torch.eye(XTWX.size(1)).cuda()) * self.epsilon
        regular_term[0, 0] = 0
        XTWX = XTWX + regular_term  # A from Ax=B


        "SOLVING NORMAL EQUATION"
        for ch in range(3):
            "Y FROM INPUT"
            Y = input[:, ch, :, :].unsqueeze(1)  # b, 1, h_d, w_d
            Y = self.unfold_and_padding(Y)  # b, size_Pkernel, hw_d

            Y = Y.permute(0, 2, 1).contiguous().view(b * hw, size_Pkernel, 1)

            XTWY = torch.bmm(XTW, Y)  # b * hw, ch_de, 1

            "SOLVING LEAST SQUARE OF AX = B"
            para, _ = torch.solve(XTWY, XTWX)  # (b * hw_d), ch_de, 1

            if self.loss_type == 1 or self.loss_type == 2:
                # modifying para for applying to unfolded data  # (b * hw_d) * size_Pkernel, ch_de, 1
                para = para.unsqueeze(1).repeat(1, size_Pkernel, 1, 1).view((b * hw) * size_Pkernel, ch_de, 1)

            "PREDICTION FOR DE NOISED COLOR BY PARA"
            out_1ch = torch.bmm(domain, para)  # (b*hw), 1 or (b*hw*size_Pkernel), 1
            out[:, ch] = out_1ch[:, 0, 0]

        return out



    def get_loss(self, y_pred, y, W):
        "unfolded loss 때문에 따로 만듦"
        " 그래서 W 값을 이용해 loss를 구함. 이는 APR과 비슷"
        b, ch, h, w = y.size()


        if self.loss_type == 1:
            y = self.unfold_and_padding(y)  # b, 3 * size_p_kernel, hw
            y = y.view(b, 3, self.size_p_kernel, h * w).permute(0, 3, 2, 1).contiguous()
            y = y.view(-1, 3)  # b*hw*size_p_kernel 3
            return self.loss_fn(y_pred, y)

        elif self.loss_type == 2:
            y = self.unfold_and_padding(y)  # b, 3 * size_p_kernel, hw
            y = y.view(b, 3, self.size_p_kernel, h * w).permute(0, 3, 2, 1).contiguous()
            y = y.view(-1, 3)  # b*hw*size_p_kernel 3

            W = W.permute((0, 2, 3, 1)).contiguous()  # b, h, w, size_Pkernel
            W = W.view((-1, 1))  # b * h * w * size_Pkernel
            return self.loss_fn(y_pred * W, y * W)

        else:
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
            x_unfolded = F.unfold(x, kernel_length, padding=kernel_length//2)
        else:
            # image resolution gonna be reduced
            x_unfolded = F.unfold(x, kernel_length, padding=0)

        return x_unfolded

    def norm_in_prediction_window(self, design):
        """
                input : design (b*hw_d, ch_de, P_kernel)
                output : normalized design in terms of a prediction window
                feature #1 : 꼭 input 형태에 유의를 할 필요가 있음.
        """
        def min_max_norm(input):
            # input : b*hw_d, ch_de, P_kernel
            a = input.dim()
            # min max
            if a == 3:
                min_input = torch.min(torch.min(input, 1, True)[0], 2, True)[0]
                max_input = torch.max(torch.max(input, 1, True)[0], 2, True)[0]
            else:
                min_input = torch.min(input, 1, True)[0]
                max_input = torch.max(input, 1, True)[0]

            return (input - min_input) / (max_input - min_input + 0.001)

        # bhw_d, inter_tile, t, ch_de = design.size()

        # albedo
        design[:, 1:4, :] = min_max_norm(design[:, 1:4, :])

        # depth
        design[:, 4, :] = min_max_norm(design[:, 4, :])

        # normal
        design[:, 5:8, :] = min_max_norm(design[:, 5:8, :])

        # # grid
        # ch_grid = ch_de - 8
        # for i in range(ch_grid):
        #     design[:, :, :, 8 + i] = min_max_norm(design[:, :, :, 8 + i])

        return design

    def make_XTW_from_linear_W(self, XT, W_linear):
        "W_linear가 diagonal을 만들어서 하면 memory문제가 생김."
        " XT = b*hw, ch_de, size_Pkernel"
        " W_linear = b*hw, size_Pkernel"

        XTW = torch.ones_like(XT)
        ch_de = XT.size(1)
        for i in range(ch_de):
            XTW[:, i, :] = torch.multiply(XT[:, i, :], W_linear)
        return XTW

    def make_grid_in_p_kernel(self):
        " prediction window를 범위로 [min, max]범위로 두는 grid 생성 "
        " self에 저장이 된 param의 정보를 받고 grid 생성 "
        " unfolded 형태에 맞춰 output은 ch, s ** 2 임."

        s = self.length_p_kernel

        grid_x, grid_y = torch.meshgrid([torch.linspace(0, 1, steps=(s)),
                                         torch.linspace(0, 1, steps=(s))])  # 0, 1
        grid = torch.stack([grid_x, grid_y], 2)  # s s 2

        for i in range(self.params["grid_order"]):
            if i == 0:
                out = grid
            else:
                order = i + 1
                out = torch.cat([out, torch.pow(grid, order)], 2)

        out = out.permute(2, 0, 1).view(2 * self.params["grid_order"], s ** 2)

        return out.cuda()


    def make_full_res_out(self, out, b, h, w, length_p_kernel, W):
        " b*hw, 3 또는 b*hw*Pkernel, 3과 같은 out을 full res로 만듦 "

        size_p_kernel = length_p_kernel ** 2

        "folding을 해야 full res가 나옴."
        if self.loss_type == 1 or self.loss_type == 2:
            "out = b*hw*Pkernel, 3"
            out = out.view(b, h*w, size_p_kernel, 3).permute(0, 3, 2, 1).contiguous()  # b, 3, Pkernel, hw
            out = out.view(b, 3 * size_p_kernel, h*w)

            if self.kernel_accum:
                # 1ch -> 3ch
                W = W.unsqueeze(1).repeat(1, 3, 1, 1, 1)  # b, 3, size_Pkernel, h, w
                W = W.view(b, 3 * size_p_kernel, h * w)

                W_over = F.fold(W, output_size=(h, w), kernel_size=length_p_kernel,
                                   padding=length_p_kernel // 2)

                out_over = F.fold(out * W, output_size=(h, w), kernel_size=length_p_kernel,
                                  padding=length_p_kernel // 2) / W_over

            else:
                ones = torch.ones_like(out)
                ones_over = F.fold(ones, output_size=(h, w), kernel_size=length_p_kernel,
                                   padding=length_p_kernel // 2)

                out_over = F.fold(out, output_size=(h, w), kernel_size=length_p_kernel,
                                      padding=length_p_kernel // 2) / ones_over

            return out_over
        else:
            "out = b*hw, 3"
            out = out.view(b, h*w, 3).permute(0, 2, 1).contiguous()
            return out.view(b, 3, h, w)


    def make_full_res_out_test_mode(self, out, b, h, w, length_p_kernel, W, CPU_mode=False):
        " contiguous()에 의한 메모리를 줄이기 위해 만듦."
        " b, 3, h, w 또는 b, 3 * size_p_kernel, h*w 과 같은 out을 full res로 만듦 "

        size_p_kernel = length_p_kernel ** 2


        "folding을 해야 full res가 나옴."
        if self.loss_type == 1 or self.loss_type == 2:
            if CPU_mode:
                out = out.cpu()
                W = W.cpu()


            out = out.view(b, 3 * size_p_kernel, h * w)

            if self.kernel_accum:
                # 1ch -> 3ch
                W = W.unsqueeze(1).repeat(1, 3, 1, 1, 1)  # b, 3, size_Pkernel, h, w
                W = W.view(b, 3 * size_p_kernel, h * w)

                W_over = F.fold(W, output_size=(h, w), kernel_size=length_p_kernel,
                                   padding=length_p_kernel // 2)

                out_over = F.fold(out * W, output_size=(h, w), kernel_size=length_p_kernel,
                                  padding=length_p_kernel // 2) / W_over

            else:
                ones = torch.ones_like(out)
                ones_over = F.fold(ones, output_size=(h, w), kernel_size=length_p_kernel,
                                   padding=length_p_kernel // 2)

                out_over = F.fold(out, output_size=(h, w), kernel_size=length_p_kernel,
                                      padding=length_p_kernel // 2) / ones_over

            if CPU_mode:
                out_over = out_over.cuda()

            return out_over
        else:
            return out

