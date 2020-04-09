from torch import nn
from .vgg_base import VGG, conv_block

class VGG16_bn(VGG):
    def __init__(self, input_channels, **kwargs):
        conv_layers = [
            *conv_block(2, input_channels, 64),

            *conv_block(2, 64, 128),

            *conv_block(3, 128, 256),

            *conv_block(3, 256, 512),

            *conv_block(3, 512, 512),
        ]

        super().__init__(model_name='vgg16_bn', conv_layers=nn.Sequential(*conv_layers), **kwargs)

class VGG16(VGG):
    def __init__(self, input_channels, **kwargs):
        conv_layers = [
            *conv_block(2, input_channels, 64, batch_norm=False),

            *conv_block(2, 64, 128, batch_norm=False),

            *conv_block(3, 128, 256, batch_norm=False),

            *conv_block(3, 256, 512, batch_norm=False),

            *conv_block(3, 512, 512, batch_norm=False),
        ]

        super().__init__(model_name='vgg16', conv_layers=nn.Sequential(*conv_layers), **kwargs)

