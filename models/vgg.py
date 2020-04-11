from torch import nn
from .vgg_base import VGG
from .core.layers import conv2dRelu_block
from collections import OrderedDict
"""
https://stackoverflow.com/questions/55140554/convolutional-encoder-error-runtimeerror-input-and-target-shapes-do-not-matc/55143487#55143487

Here is the formula;

N --> Input Size, F --> Filter Size, stride-> Stride Size, pdg-> Padding size

ConvTranspose2d;

OutputSize = N*stride + F - stride - pdg*2

Conv2d;

OutputSize = (N - F)/stride + 1 + pdg*2/stride [e.g. 32/3=10 it ignores after the comma]
"""


class VGG11_bn(VGG):
    def __init__(self, input_channels, **kwargs):
        conv_layers = [
            *conv2dRelu_block('1', 1, input_channels, 64),
        
            *conv2dRelu_block('2', 1, 64, 128),

            *conv2dRelu_block('3', 2, 128, 256),

            *conv2dRelu_block('4', 2, 256, 512),

            *conv2dRelu_block('5', 2, 512, 512),
        ]

        super().__init__(model_name='vgg11_bn', conv_layers=nn.ModuleDict(OrderedDict(conv_layers)), **kwargs)


class VGG11(VGG):
    def __init__(self, input_channels, **kwargs):
        conv_layers = [
            *conv2dRelu_block('1', 1, input_channels, 64, batch_norm=False),

            *conv2dRelu_block('2', 1, 64, 128, batch_norm=False),

            *conv2dRelu_block('3', 2, 128, 256, batch_norm=False),

            *conv2dRelu_block('4', 2, 256, 512, batch_norm=False),

            *conv2dRelu_block('5', 2, 512, 512, batch_norm=False)
        ]

        super().__init__(model_name='vgg11', conv_layers=nn.ModuleDict(OrderedDict(conv_layers)), **kwargs)

class VGG16_bn(VGG):
    def __init__(self, input_channels, **kwargs):
        conv_layers = [
            *conv2dRelu_block('1', 2, input_channels, 64),

            *conv2dRelu_block('2', 2, 64, 128),

            *conv2dRelu_block('3', 3, 128, 256),

            *conv2dRelu_block('4', 3, 256, 512),

            *conv2dRelu_block('5', 3, 512, 512),
        ]

        super().__init__(model_name='vgg16_bn', conv_layers=nn.ModuleDict(OrderedDict(conv_layers)), **kwargs)

class VGG16(VGG):
    def __init__(self, input_channels, **kwargs):
        conv_layers = [
            *conv2dRelu_block('1', 2, input_channels, 64, batch_norm=False),

            *conv2dRelu_block('2', 2, 64, 128, batch_norm=False),

            *conv2dRelu_block('3', 3, 128, 256, batch_norm=False),

            *conv2dRelu_block('4', 3, 256, 512, batch_norm=False),

            *conv2dRelu_block('5', 3, 512, 512, batch_norm=False),
        ]

        super().__init__(model_name='vgg16', conv_layers=nn.ModuleDict(OrderedDict(conv_layers)), **kwargs)

