from ..core.layers import *
from .base import SSDvggBase, SSDTrainConfig, SSDValConfig, load_vgg_weights
from ..core.boxes import *

from torch import nn


class SSD512(SSDvggBase):
    def __init__(self, class_labels, input_shape=(512, 512, 3), batch_norm=False,
                 val_config=SSDValConfig(val_conf_threshold=0.01, vis_conf_threshold=0.6, iou_threshold=0.45, topk=200)):
        """
        :param class_labels: list or tuple of str
        :param input_shape: tuple, 3d and (height, width, channel)
        :param batch_norm: bool, whether to add batch normalization layers
        """
        ### train_config ###
        if not batch_norm:
            train_config = SSDTrainConfig(class_labels=class_labels, input_shape=input_shape, batch_norm=batch_norm,

                                          aspect_ratios=((1, 2), (1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2), (1, 2)),
                                          classifier_source_names=('convRL4_3', 'convRL7', 'convRL8_2', 'convRL9_2', 'convRL10_2', 'convRL11_2', 'convRL12_2'),
                                          addon_source_names=('convRL4_3',),

                                          codec_means=(0.0, 0.0, 0.0, 0.0), codec_stds=(0.1, 0.1, 0.2, 0.2),
                                          rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))
        else:
            train_config = SSDTrainConfig(class_labels=class_labels, input_shape=input_shape, batch_norm=batch_norm,

                                          aspect_ratios=((1, 2), (1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2), (1, 2)),
                                          classifier_source_names=('convBnRL4_3', 'convBnRL7', 'convBnRL8_2', 'convBnRL9_2', 'convBnRL10_2', 'convRLBn11_2', 'convRL12_2'),
                                          addon_source_names=('convBnRL4_3',),

                                          codec_means=(0.0, 0.0, 0.0, 0.0), codec_stds=(0.1, 0.1, 0.2, 0.2),
                                          rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))

        ### layers ###
        Conv2d.batch_norm = batch_norm
        vgg_layers = [
            *Conv2d.relu_block('1', 2, train_config.input_channel, 64),

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
            *Conv2d.relu_one('10_2', 128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),

            *Conv2d.relu_one('11_1', 256, 128, kernel_size=(1, 1)),
            *Conv2d.relu_one('11_2', 128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),

            *Conv2d.relu_one('12_1', 256, 128, kernel_size=(1, 1)),
            *Conv2d.relu_one('12_2', 128, 256, kernel_size=(4, 4), stride=(1, 1), padding=1),
            # if batch_norm = True, error is thrown. last layer's channel == 1 may be caused
        ]
        vgg_layers = nn.ModuleDict(vgg_layers)
        extra_layers = nn.ModuleDict(extra_layers)

        super().__init__(train_config, val_config, defaultBox=DBoxSSDOriginal(img_shape=input_shape,
                                                                              scale_conv4_3=0.07, scale_range=(0.15, 0.9),
                                                                              aspect_ratios=train_config.aspect_ratios),
                         vgg_layers=vgg_layers, extra_layers=extra_layers)

    def load_vgg_weights(self):
        if self.batch_norm:
            load_vgg_weights(self, 'vgg16_bn')
        else:
            load_vgg_weights(self, 'vgg16')