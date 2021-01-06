import torch
import torch.nn as nn
import torchvision
from torch.hub import load_state_dict_from_url

from bnn import nn as bnn_layers


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def recover_weights(model_dict):
    pretrained_dict = {k: v for k, v in load_state_dict_from_url('https://download.pytorch.org/models/mobilenet_v2'
                                                                 '-b0353104.pth').items() if
                       k in model_dict and v.shape == model_dict[k].shape}

    model_dict.update(pretrained_dict)
    #print('Keys recovered from the pretrained model state that are still compatible:', len(pretrained_dict))
    return model_dict


def final_activator(num_classes, mode):
    f = lambda x: x.reshape(-1, num_classes, 2)
    if mode == 'sigmoid':
        g = nn.Sigmoid()
        f = lambda x: g(x).reshape(-1, num_classes, 2)
    elif mode == 'softmax':
        g = nn.Softmax(dim=-1)
        f = lambda x: g(x.reshape(-1, num_classes, 2))
    elif mode == 'log_softmax':
        g = nn.LogSoftmax(dim=-1)
        f = lambda x: g(x.reshape(-1, num_classes, 2))
    else:
        print(f'Mode \'{mode}\' not supported, will perform only reshape')
    return f


class ConvNet(torchvision.models.MobileNetV2):
    def __init__(self, inverted_residual_setting,
                 num_classes, mode='softmax',
                 width_mult=1, round_nearest=8):
        # Call constructor to build most of the newtork
        super(ConvNet, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                      width_mult=width_mult, round_nearest=round_nearest)

        # Salvage the pretained weights that can be salvaged
        self.load_state_dict(recover_weights(self.state_dict()), strict=True)

        # Redefine last convolutional block of the original architechture
        # In this way, we can control the output channels, they are the input channels of our extra convolution
        last_conv_in = _make_divisible(inverted_residual_setting[-1][1] * width_mult, round_nearest)
        #print(last_conv_in)
        self.features[-1] = torchvision.models.mobilenet.ConvBNReLU(last_conv_in, last_conv_in * 2, kernel_size=1)

        # Whether we want a bayesian network or not, append the proper layer type to 'features'
        study_layer = nn.Conv2d(last_conv_in * 2, last_conv_in * 4, 3, 1, 1, bias=False)
        self.features.add_module('study', study_layer)

        # Redefine 'classifier' block at end of the network
        # In this way, we can control the input channels, they are the output channels of our extra convolution
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_conv_in * 4, num_classes * 2)
        )

        # Final touch, if the user requested it
        self.final_touch = final_activator(num_classes, mode)

    def forward(self, x):
        x = self._forward_impl(x)
        x = self.final_touch(x)
        return x


class BayesConvNet(bnn_layers.BayesianNetworkModule, torchvision.models.MobileNetV2):
    def __init__(self, inverted_residual_setting,
                 samples=12,
                 num_classes=400, mode='softmax', bayes_classifier=False,
                 width_mult=1., round_nearest=8):
        # Call constructor to build most of the newtork
        super(BayesConvNet, self).__init__(in_channels=3, out_channels=num_classes,  # Popped by BNN
                                           inverted_residual_setting=inverted_residual_setting,
                                           width_mult=width_mult, round_nearest=round_nearest)

        print(f'This bayesian network will sample {samples} times')
        self.samples = samples

        self.load_state_dict(recover_weights(self.state_dict()), strict=True)

        # TODO: check for NaN weights
        # for inv_res_block in self.features.parameters():
        #    inv_res_block.requires_grad = False
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)

        # Redefine last convolutional block of the original mobilenetv2 architechture
        # In this way, we can control the output channels, they are the input channels of our extra convolution
        last_block_in = _make_divisible(inverted_residual_setting[-1][1] * width_mult, round_nearest)
        # last_block_in=320 # last_block_in * samples?
        self.features[-1] = torchvision.models.mobilenet.ConvBNReLU(last_block_in, last_block_in * 2, kernel_size=1)

        # Add out study bayesian convolutional block
        self.features.add_module('study',
                                 bnn_layers.NormalConv2d(last_block_in * 2, last_block_in * 4, 3, 1, 1, bias=False))

        # Redefine 'classifier' block at end of the network
        # In this way, we can control the input channels, they are the output channels of our extra convolution
        linear_layer = bnn_layers.NormalLinear if bayes_classifier else nn.Linear
        self.classifier = linear_layer(last_block_in * 4, num_classes * 2)

        # Final touch, if the user requested it
        self.final_touch = final_activator(num_classes, mode)

    def _forward(self, x, **kwargs):
        x = self._forward_impl(x)  # MRO will resolve to MobileNetV2
        x = self.final_touch(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary
    import hiddenlayer as hl

    common_parameters = {'num_classes': 400, 'mode': 'softmax'}

    # Different irs to achieve more or less 1mil params on both the normal net and the one with a bayesian layer

    #       t, c, n, s
    irs = [[1, 16, 1, 1],
           [6, 24, 2, 2],
           [6, 32, 3, 2],
           [6, 64, 2, 2],
           [6, 96, 1, 1]]
    model = ConvNet(irs, **common_parameters)
    model.eval()
    print('Total model parameters:', sum(p.numel() for p in model.parameters()))
    summary(model, (3, 240, 320))


    #g = hl.build_graph(model, torch.zeros([1, 3, 240, 320])).build_dot()
    #g.view()
    # print(model(torch.randn(1, 3, 240, 320)).shape)

    # t, c, n, s
    irs = [[1, 16, 1, 1],
           [6, 24, 2, 2],
           [6, 32, 3, 2],
           [6, 64, 4, 2]]
    model = BayesConvNet(irs, samples=1, **common_parameters)
    print('Total model parameters:', sum(p.numel() for p in model.parameters()))
    summary(model, (3, 120, 160))
    print(model)
