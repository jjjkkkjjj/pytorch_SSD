from .core.layers import *

from torch import nn
import torch
from collections import OrderedDict

# defalut boxes number for each feature map
_dbox_nums = [4, 6, 6, 6, 4, 4]

# classifier's source layers
# consists of conv4_3, conv7, conv8_2, conv9_2, conv10_2, conv11_2
_classifier_source_names = ['conv4_l2norm', 'conv7', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2']

class SSD300(nn.Module):
    def __init__(self, class_nums):
        super().__init__()

        self.class_nums = class_nums

        vgg_layers = [
            ('conv1', Conv2dRelu_block(2, 3, 64, batch_norm=False)),

            ('conv2', Conv2dRelu_block(2, 64, 128, batch_norm=False)),

            ('conv3', Conv2dRelu_block(3, 128, 256, batch_norm=False)),

            ('conv4', Conv2dRelu_block(3, 256, 512, batch_norm=False)),

            ('conv4_l2norm', L2Normalization(512, gamma=20)),

            ('conv5', Conv2dRelu_block(3, 512, 512, batch_norm=False, pool_k_size=(3, 3), pook_stride=(1, 1))), # replace last maxpool layer's kernel and stride
        ]

        extra_layers = [
            ('conv6', Conv2dRelu(512, 1024, kernel_size=(3, 3), padding=6, dilation=6, relu_inplace=True)), # Atrous convolution

            ('conv7', Conv2dRelu(1024, 1024, kernel_size=(1, 1), padding=1)),

            ('conv8_1', Conv2dRelu(1024, 256, kernel_size=(1, 1), padding=1)),
            ('conv8_2', Conv2dRelu(256, 512, kernel_size=(3, 3), stride=2, padding=1)),

            ('conv9_1', Conv2dRelu(512, 128, kernel_size=(1, 1), padding=1)),
            ('conv9_2', Conv2dRelu(128, 256, kernel_size=(3, 3), stride=2, padding=1)),

            ('conv10_1', Conv2dRelu(256, 128, kernel_size=(1, 1), padding=1)),
            ('conv10_2', Conv2dRelu(128, 256, kernel_size=(3, 3), padding=1)),

            ('conv11_1', Conv2dRelu(256, 128, kernel_size=(1, 1), padding=1)),
            ('conv11_2', Conv2dRelu(128, 256, kernel_size=(3, 3), padding=1)),
        ]

        self.feature_layers = nn.ModuleDict(OrderedDict(vgg_layers + extra_layers))

        classifier_layers = []
        for i, (source_name, dbox_num) in enumerate(zip(_classifier_source_names, _dbox_nums)):
            source = self.feature_layers[source_name]
            classifier_layers += [
                ('feature{0}_{1}'.format(i, source_name),
                Conv2dRelu(source.out_channels, dbox_num * (class_nums + 4), kernel_size=(3, 3), padding=1)),
                ('flatten{0}_feature'.format(i), Flatten())
            ]

        self.classifier_layers = nn.ModuleDict(OrderedDict(classifier_layers))


    def forward(self, x):
        features = []
        i = 0
        for name, layer in self.feature_layers.items():
            x = layer(x)
            # get features by feature map convolution
            if name in _classifier_source_names:
                features.append(self.classifier_layers['feature{0}_{1}'.format(i, name)](x))
                i += 1

        features = torch.cat(features, dim=0)
        return features