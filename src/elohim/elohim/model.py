import torch.nn as nn
import torchvision

import bnn
from bnn import nn as bnn_layers


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


from torch.hub import load_state_dict_from_url


def recover_weights(model_dict):
    # TODO: freeze these weights with required_grad = False
    pretrained_dict = {k: v for k, v in load_state_dict_from_url('https://download.pytorch.org/models/mobilenet_v2'
                                                                 '-b0353104.pth').items() if
                       k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    print('Keys recovered from the pretrained model state that are still compatible:', len(pretrained_dict))
    return model_dict


def final_activator(mode):
    f = lambda x: x#.reshape(-1, 400, 2)
    if mode == 'sigmoid':
        print('Logistic function has been appended')

        def f(x):
            x = nn.Sigmoid()(x)
            x = x.reshape(-1, 400, 2)
            return x
    elif mode == 'softmax':
        print('LogSoftmax function has been appended')

        def f(x):
            x = x.reshape(-1, 400, 2)
            x = nn.LogSoftmax(dim=2)(x)
            return x
    else:
        print(f'Mode \'{mode}\' not supported, will perform only reshape')
    return f


class ConvNet(torchvision.models.MobileNetV2):
    def __init__(self, inverted_residual_setting,
                 in_planes, out_planes,
                 num_classes, mode='softmax',
                 width_mult=1, round_nearest=8):
        # Call constructor to build most of the newtork
        super(ConvNet, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                      width_mult=width_mult, round_nearest=round_nearest)

        # Salvage the pretained weights that can be salvaged
        #self.load_state_dict(recover_weights(self.state_dict()), strict=True)

        # Redefine last convolutional block of the original architechture
        # In this way, we can control the output channels, they are the input channels of our extra convolution
        self.features[-1] = torchvision.models.mobilenet.ConvBNReLU(
            _make_divisible(inverted_residual_setting[-1][1] * width_mult, round_nearest),
            in_planes, kernel_size=1)

        # Whether we want a bayesian network or not, append the proper layer type to 'features'
        study_layer = nn.Conv2d(in_planes, out_planes, 3, 1, 1, bias=False)
        self.features.add_module('extra', study_layer)

        # Redefine 'classifier' block at end of the network
        # In this way, we can control the input channels, they are the output channels of our extra convolution
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(out_planes, num_classes * 2)
        )

        # Final touch, if the user requested it
        self.final_touch = final_activator(mode)

    def forward(self, x):
        x = self._forward_impl(x)
        x = self.final_touch(x)
        return x


class BayesConvNet(bnn_layers.BayesianNetworkModule, torchvision.models.MobileNetV2):
    def __init__(self, inverted_residual_setting,
                 in_planes, out_planes, samples=12,
                 num_classes=400, mode='softmax', bayes_classifier=False,
                 width_mult=1, round_nearest=8):
        # Call constructor to build most of the newtork
        super(BayesConvNet, self).__init__(in_channels=3, out_channels=num_classes,  # Popped by BNN
                                           inverted_residual_setting=inverted_residual_setting,
                                           width_mult=width_mult, round_nearest=round_nearest)

        # TODO: print samples
        self.samples = samples

        self.load_state_dict(recover_weights(self.state_dict()), strict=True)

        # Redefine last convolutional block of the original mobilenetv2 architechture
        # In this way, we can control the output channels, they are the input channels of our extra convolution
        last_block_in = _make_divisible(inverted_residual_setting[-1][1] * width_mult, round_nearest)
        print(last_block_in)
        # last_block_in=320 # last_block_in * samples?
        self.features[-1] = torchvision.models.mobilenet.ConvBNReLU(last_block_in, in_planes, kernel_size=1)

        # Add out study bayesian convolutional block
        self.features.add_module('study', bnn_layers.NormalConv2d(in_planes, out_planes, 3, 1, 1, bias=False))

        # Redefine 'classifier' block at end of the network
        # In this way, we can control the input channels, they are the output channels of our extra convolution
        linear_layer = bnn_layers.NormalLinear if bayes_classifier else nn.Linear
        self.classifier = nn.Sequential(
            # Flatten(),
            nn.Dropout(0.2),
            linear_layer(out_planes, num_classes * 2)
        )

        # Final touch, if the user requested it
        self.final_touch = final_activator(mode)

    def _forward(self, x, **kwargs):
        x = self._forward_impl(x)  # MRO will resolve to MobileNetV2
        x = self.final_touch(x)
        return x


if __name__ == '__main__':
    import numpy as np
    import torch

    irs = [[1, 8, 1, 1],
           [6, 16, 2, 2],
           [6, 24, 3, 2],
           [6, 32, 1, 1]]

    # model = ConvNet(irs, in_planes=200, out_planes=400, num_classes=400)
    # print('Total model parameters:', sum(p.numel() for p in model.parameters()))
    # print(model(torch.randn(1, 3, 240, 320)).shape)

    model = BayesConvNet(irs, in_planes=200, out_planes=250,
                         samples=2, num_classes=400, mode='softmax')
    print('Total model parameters:', sum(p.numel() for p in model.parameters()))

    output = model(torch.randn(16, 3, 240, 320))
    print(output)
    import torch.optim as optim
    loss_function = torch.nn.CrossEntropyLoss()
    KLdiv = bnn_layers.KLDivergence(number_of_batches=1)
    get_entropy = bnn_layers.Entropy(dim=-1)

    # display(self.features[-3:], self.classifier)