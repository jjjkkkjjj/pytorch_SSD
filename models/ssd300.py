from .core.layers import *
from torch import nn

_dbox_nums = [4, 4, 6, 6, 4, 4]

class SSD300(nn.Module):
    def __init__(self):
        super().__init__()

        vgg_layers = [
            Conv2dRelu_block(2, 3, 64, batch_norm=False),

            Conv2dRelu_block(2, 64, 128, batch_norm=False),

            Conv2dRelu_block(3, 128, 256, batch_norm=False),

            Conv2dRelu_block(3, 256, 512, batch_norm=False),

            Conv2dRelu_block(3, 512, 512, batch_norm=False, pool_k_size=(3, 3), pook_stride=(1, 1)), # replace last maxpool layer's kernel and stride
        ]
        self.vgg_layers = nn.Sequential(*vgg_layers)

        extra_layers = [
            Conv2dRelu(512, 1024, kernel_size=(3, 3), padding=6, dilation=6, relu_inplace=True), # Atrous convolution

            Conv2dRelu(1024, 1024, kernel_size=(1, 1), padding=1),

            Conv2dRelu(1024, 256, kernel_size=(1, 1), padding=1),
            Conv2dRelu(256, 512, kernel_size=(3, 3), stride=2, padding=1),

            Conv2dRelu(512, 128, kernel_size=(1, 1), padding=1),
            Conv2dRelu(128, 256, kernel_size=(3, 3), stride=2, padding=1),

            Conv2dRelu(256, 128, kernel_size=(1, 1), padding=1),
            Conv2dRelu(128, 256, kernel_size=(3, 3), padding=1),

            Conv2dRelu(256, 128, kernel_size=(1, 1), padding=1),
            Conv2dRelu(128, 256, kernel_size=(3, 3), padding=1),
        ]

        # classifier's source layers
        classifier_sources = [vgg_layers[-1], ]

        classifier_layers = [
            Conv2dRelu(vgg_layers[-1].out_channels, )
        ]

def extra_conv_block(block_num, in_channels, out_channels, batch_norm=True, **kwargs):
    """
    :return
        list of layers
    """
    kernel_size = kwargs.pop('conv_k_size', (3, 3))
    stride = kwargs.pop('conv_stride', (1, 1))
    padding = kwargs.pop('conv_padding', 1)

    in_c = in_channels
    ret = []
    # append conv block
    for _ in range(block_num):
        if not batch_norm:
            ret += [nn.Conv2d(in_c, out_channels, kernel_size, stride=stride, padding=padding),
                    nn.ReLU(True)]
        else:
            ret += [nn.Conv2d(in_c, out_channels, kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)]
        in_c = out_channels

    kernel_size = kwargs.pop('pool_k_size', (2, 2))
    stride = kwargs.pop('pook_stride', (2, 2))

    # append maxpooling
    ret += [nn.MaxPool2d(kernel_size, stride)]

    return ret