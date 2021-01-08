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
    # print('Keys recovered from the pretrained model state that are still compatible:', len(pretrained_dict))
    return model_dict


def flexible_weights(weights_path, device):
    return {k.replace('module.', ''): v
            for k, v in torch.load(weights_path, map_location=device).items()}


class ConvNet(torchvision.models.MobileNetV2):
    def __init__(self, inverted_residual_setting, num_classes, width_mult=1, round_nearest=8):
        self.num_classes = num_classes
        # Call constructor to build most of the newtork
        super(ConvNet, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                      width_mult=width_mult, round_nearest=round_nearest)

        # Salvage the pretained weights that can be salvaged
        self.load_state_dict(recover_weights(self.state_dict()), strict=True)

        # Redefine last convolutional block of the original architechture
        # In this way, we can control the output channels, they are the input channels of our extra convolution
        last_conv_in = _make_divisible(inverted_residual_setting[-1][1] * width_mult, round_nearest)
        # print(last_conv_in)
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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self._forward_impl(x)
        return self.softmax(x.reshape(-1, self.num_classes, 2))


class BayesConvNet(bnn_layers.BayesianNetworkModule, torchvision.models.MobileNetV2):
    def __init__(self, inverted_residual_setting, samples=12, num_classes=400, bayes_classifier=False,
                 width_mult=1., round_nearest=8):
        self.num_classes = num_classes

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

        self.softmax = nn.Softmax(dim=-1)

    def _forward(self, x, **kwargs):
        x = self._forward_impl(x)
        return self.softmax(x.reshape(-1, self.num_classes, 2))


def initialize_model(batch_size, samples=None, weights_path=None):
    common_parameters = {'num_classes': 400}
    if samples is not None:
        #       t, c, n, s
        irs = [[1, 16, 1, 1],
               [6, 24, 2, 2],
               [6, 32, 3, 2],
               [6, 64, 4, 2]]
        m = BayesConvNet(inverted_residual_setting=irs, samples=samples, **common_parameters)
    else:
        irs = [[1, 16, 1, 1],
               [6, 24, 2, 2],
               [6, 32, 3, 2],
               [6, 64, 2, 2],
               [6, 96, 1, 1]]
        m = ConvNet(inverted_residual_setting=irs, **common_parameters)

    # Deal with multiple GPUs if present
    parallel = torch.cuda.device_count() > 1

    if parallel:
        assert batch_size is not None
        print(f"Using {torch.cuda.device_count()} GPUs!")
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        batch_size = int(batch_size * torch.cuda.device_count())
    else:
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        print('Using device:', device)

    if weights_path is not None:
        try:
            m.load_state_dict(flexible_weights(weights_path, device))
            print('Weights loaded successfully')
        except RuntimeError:
            print('Model is incompatible')

    if parallel:
        m = nn.DataParallel(m)

    m.to(device)
    return m, batch_size, device, parallel


if __name__ == '__main__':
    from torchsummary import summary
    import hiddenlayer as hl

    # Different irs to achieve more or less 1mil params on both the normal net and the one with a bayesian layer
    model, _, _, _ = initialize_model(batch_size=1)
    model.eval()
    print('Total model parameters:', sum(p.numel() for p in model.parameters()))
    # summary(model, (3, 240, 320))

    # g = hl.build_graph(model, torch.zeros([1, 3, 240, 320])).build_dot()
    # g.view()

    output = model(torch.randn(1, 3, 240, 320))
    print(output.min(), output.max(), output.shape)

    # t, c, n, s
    model, _, _, _ = initialize_model(batch_size=1, samples=1)
    model.eval()
    print('Total model parameters:', sum(p.numel() for p in model.parameters()))
    # summary(model, (3, 120, 160))
    # print(model)
