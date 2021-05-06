import torch.nn as nn
import Models.resnet as resnet


class FG_pixtransform_v1(nn.Module):
    "1x1 conv로 feature를 pixel wise로 converting을 함."
    " resnet을 활용할 수 있도록 함."

    def __init__(self, ch_in=5, ch_out=3, n_layers=12, is_resnet=True, forbid_var=True):
        super(FG_pixtransform_v1, self).__init__()

        self.start_ch = ch_in
        self.inter_ch = 200
        self.final_ch = ch_out
        self.forbid_var = forbid_var

        if forbid_var:
            print("Even input may have var values, but FG_pixtransform_v1 only takes color values.")
            self.start_ch = 7

        self.layers = [nn.Conv2d(self.start_ch, self.inter_ch, 1, padding=(1 - 1) // 2)]
        if is_resnet:
            for l in range(n_layers // 2 - 1):
                self.layers += [
                    resnet.Resblock_2d(self.inter_ch, self.inter_ch, self.inter_ch, 1)
                ]
            self.feature_layers_feed = nn.Sequential(*self.layers)  # to get the feature
            self.layers_for_weights_feed = nn.Conv2d(self.inter_ch, self.final_ch, 1,
                                                     padding=(1 - 1) // 2)
        else:
            for l in range(n_layers - 2):
                self.layers += [
                    nn.Conv2d(self.inter_ch, self.inter_ch, 1, padding=(1 - 1) // 2),
                    nn.LeakyReLU()
                ]
            self.feature_layers_feed = nn.Sequential(*self.layers)  # to get the feature
            self.layers_for_weights_feed = nn.Conv2d(self.inter_ch, self.final_ch, 1,
                                                     padding=(1 - 1) // 2)


    def forward(self, input):
        if self.forbid_var:
            return self.layers_for_weights_feed(self.feature_layers_feed(input[:, :7, :, :]))
        else:
            return self.layers_for_weights_feed(self.feature_layers_feed(input))
