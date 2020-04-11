from .core.layers import *
from .core.utils import *

from torch import nn
import torch
from collections import OrderedDict

# defalut boxes number for each feature map
_dbox_nums = [4, 6, 6, 6, 4, 4]

# classifier's source layers
# consists of conv4_3, conv7, conv8_2, conv9_2, conv10_2, conv11_2
_classifier_source_names = ['conv4_3', 'conv7', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2']
_l2norm_names = ['conv4_3']
class SSD300(nn.Module):
    def __init__(self, class_nums):
        super().__init__()

        self.class_nums = class_nums

        vgg_layers = [
            *conv2dRelu_block('1', 2, 3, 64, batch_norm=False),

            *conv2dRelu_block('2', 2, 64, 128, batch_norm=False),

            *conv2dRelu_block('3', 3, 128, 256, batch_norm=False, pool_ceil_mode=True),

            *conv2dRelu_block('4', 3, 256, 512, batch_norm=False),

            *conv2dRelu_block('5', 3, 512, 512, batch_norm=False, pool_k_size=(3, 3), pool_stride=(1, 1), pool_padding=1), # replace last maxpool layer's kernel and stride

            # Atrous convolution
            *conv2dRelu('6', 512, 1024, kernel_size=(3, 3), padding=6, dilation=6, relu_inplace=True),

            *conv2dRelu('7', 1024, 1024, kernel_size=(1, 1), relu_inplace=True),
        ]

        extra_layers = [
            *conv2dRelu('8_1', 1024, 256, kernel_size=(1, 1), relu_inplace=True),
            *conv2dRelu('8_2', 256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1, relu_inplace=True),

            *conv2dRelu('9_1', 512, 128, kernel_size=(1, 1), relu_inplace=True),
            *conv2dRelu('9_2', 128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1, relu_inplace=True),

            *conv2dRelu('10_1', 256, 128, kernel_size=(1, 1), relu_inplace=True),
            *conv2dRelu('10_2', 128, 256, kernel_size=(3, 3), relu_inplace=True),

            *conv2dRelu('11_1', 256, 128, kernel_size=(1, 1), relu_inplace=True),
            *conv2dRelu('11_2', 128, 256, kernel_size=(3, 3), relu_inplace=True),
        ]

        self.feature_layers = nn.ModuleDict(OrderedDict(vgg_layers + extra_layers))

        classifier_layers = []
        for i, (source_name, dbox_num) in enumerate(zip(_classifier_source_names, _dbox_nums)):
            source = self.feature_layers[source_name]
            postfix = '_feature{}'.format(i + 1)
            if not source_name in _l2norm_names:
                layers = [
                    *conv2dRelu(postfix, source.out_channels, dbox_num * (class_nums + 4), kernel_size=(3, 3), padding=1, relu_inplace=True),
                    #('flatten{}'.format(postfix), Flatten())
                ]

            else:
                layers = [
                    ('l2norm{}'.format(postfix), L2Normalization(source.out_channels, gamma=20)),
                    *conv2dRelu(postfix, source.out_channels, dbox_num * (class_nums + 4), kernel_size=(3, 3), padding=1, relu_inplace=True),
                    #('flatten{}'.format(postfix), Flatten())
                ]

            postfix = postfix[1:]
            classifier_layers += [(postfix, nn.Sequential(OrderedDict(layers)))]

        self.classifier_layers = nn.ModuleDict(OrderedDict(classifier_layers))
        self.defaultBox = DefaultBox()
        self.classProb

    def forward(self, x):
        features = []
        i = 1
        for name, layer in self.feature_layers.items():
            x = layer(x)
            # get features by feature map convolution
            if name in _classifier_source_names:
                feature = self.classifier_layers['feature{0}'.format(i)](x)
                features.append(feature)
                #print(features[-1].shape)
                i += 1

        dboxes, features = self.defaultBox(features, _dbox_nums)

        return features

