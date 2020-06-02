from ..core.layers import *
from .base import SSDvggBase, SSDConfig, load_vgg_weights
from ..core.boxes import *

from torch import nn


class SSD300(SSDvggBase):
    def __init__(self, class_nums, input_shape=(300, 300, 3), batch_norm=False):
        """
        :param class_nums: int, class number
        :param input_shape: tuple, 3d and (height, width, channel)
        :param batch_norm: bool, whether to add batch normalization layers
        """
        ### config ###
        config = SSDConfig(class_nums=class_nums, input_shape=input_shape, batch_norm=batch_norm,

                           aspect_ratios=((1, 2), (1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2), (1, 2)),
                           classifier_source_names=('convRL4_3', 'convRL7', 'convRL8_2', 'convRL9_2', 'convRL10_2', 'convRL11_2'),
                           addon_source_names=('convRL4_3',),

                           norm_means=(0.0, 0.0, 0.0, 0.0), norm_stds=(0.1, 0.1, 0.2, 0.2))

        super().__init__(config, defaultBox=DBoxSSD300Original(scale_conv4_3=0.1, scale_range=(0.2, 0.9),
                                                               aspect_ratios=config.aspect_ratios))

        ### layers ###
        Conv2d.batch_norm = batch_norm
        vgg_layers = [
            *Conv2d.relu_block('1', 2, config.input_channel, 64),

            *Conv2d.relu_block('2', 2, 64, 128),

            *Conv2d.relu_block('3', 3, 128, 256, pool_ceil_mode=True),

            *Conv2d.relu_block('4', 3, 256, 512),

            *Conv2d.relu_block('5', 3, 512, 512, pool_k_size=(3, 3), pool_stride=(1, 1), pool_padding=1),
            # replace last maxpool layer's kernel and stride

            # Atrous convolution
            *Conv2d.relu_one('6', 512, 1024, kernel_size=(3, 3), padding=6, dilation=6),

            *Conv2d.relu_one('7', 1024, 1024, kernel_size=(1, 1)),
        ]

        extra_layers = [
            *Conv2d.relu_one('8_1', 1024, 256, kernel_size=(1, 1)),
            *Conv2d.relu_one('8_2', 256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1),

            *Conv2d.relu_one('9_1', 512, 128, kernel_size=(1, 1)),
            *Conv2d.relu_one('9_2', 128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),

            *Conv2d.relu_one('10_1', 256, 128, kernel_size=(1, 1)),
            *Conv2d.relu_one('10_2', 128, 256, kernel_size=(3, 3)),

            *Conv2d.relu_one('11_1', 256, 128, kernel_size=(1, 1)),
            *Conv2d.relu_one('11_2', 128, 256, kernel_size=(3, 3), batch_norm=False),
            # if batch_norm = True, error is thrown. last layer's channel == 1 may be caused
        ]
        vgg_layers = nn.ModuleDict(vgg_layers)
        extra_layers = nn.ModuleDict(extra_layers)

        self.build(vgg_layers, extra_layers)

    def load_vgg_weights(self):
        if self.batch_norm:
            load_vgg_weights(self, 'vgg16_bn')
        else:
            load_vgg_weights(self, 'vgg16')