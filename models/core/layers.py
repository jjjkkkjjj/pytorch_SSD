from torch import nn
import torch
from torch.nn import init

import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class L2Normalization(nn.Module):
    def __init__(self, channels, gamma=20):
        super().__init__()
        self.gamma = gamma
        self.in_channels = channels
        self.out_channels = channels
        self.scales = nn.Parameter(torch.Tensor(self.in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.scales, self.gamma)

    # Note that pytorch's dimension order is batch_size, channels, height, width
    def forward(self, x):
        # |x|_2
        # square element-wise, sum along channel and square element-wise
        norm_x = torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt()
        # normalize (x^)
        x = torch.div(x, norm_x)
        return self.scales.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x


class Predictor(nn.Module):
    def __init__(self, total_dbox_nums, class_nums):
        super().__init__()

        self._total_dbox_nums = total_dbox_nums
        self._class_nums = class_nums

    def forward(self, locs, confs):
        """
        :param locs: list of Tensor, Tensor's shape is (batch, c, h, w)
        :param confs: list of Tensor, Tensor's shape is (batch, c, h, w)
        :return: predicts: localization and confidence Tensor, shape is (batch, total_dbox_num * (4+class_nums))
        """
        locs_reshaped, confs_reshaped = [], []
        for loc, conf in zip(locs, confs):
            batch_num = loc.shape[0]

            # original feature => (batch, (class_num or 4)*dboxnum, fmap_h, fmap_w)
            # converted into (batch, fmap_h, fmap_w, (class_num or 4)*dboxnum)
            # contiguous means aligning stored 1-d memory for given array
            loc = loc.permute((0, 2, 3, 1)).contiguous()
            locs_reshaped += [loc.reshape((batch_num, -1))]

            conf = conf.permute((0, 2, 3, 1)).contiguous()
            confs_reshaped += [conf.reshape((batch_num, -1))]



        locs_reshaped = torch.cat(locs_reshaped, dim=1).reshape((-1, self._total_dbox_nums, 4))
        confs_reshaped = torch.cat(confs_reshaped, dim=1).reshape((-1, self._total_dbox_nums, self._class_nums))

        return torch.cat((locs_reshaped, confs_reshaped), dim=2)

class Conv2dRelu:
    batch_norm = True

    @staticmethod
    def block(order, block_num, in_channels, out_channels, **kwargs):
        """
        :param order: int or str
        :param block_num: int, how many conv layers are sequenced
        :param in_channels: int
        :param out_channels: int
        :param batch_norm: bool
        :param kwargs:
        :return: list of tuple is for OrderedDict
        """
        kernel_size = kwargs.pop('conv_k_size', (3, 3))
        stride = kwargs.pop('conv_stride', (1, 1))
        padding = kwargs.pop('conv_padding', 1)
        relu_inplace = kwargs.pop('relu_inplace', False)# TODO relu inplace problem >>conv4
        batch_norm = kwargs.pop('batch_norm', Conv2dRelu.batch_norm)

        in_c = in_channels
        layers = []
        # append conv block
        for bnum in range(block_num):
            postfix = '{0}_{1}'.format(order, bnum + 1)
            if not batch_norm:
                layers += [
                    ('conv{}'.format(postfix),
                     nn.Conv2d(in_c, out_channels, kernel_size, stride=stride, padding=padding)),
                    ('relu{}'.format(postfix), nn.ReLU(relu_inplace))
                ]
            else:
                layers += [
                    ('conv{}'.format(postfix),
                     nn.Conv2d(in_c, out_channels, kernel_size, stride=stride, padding=padding)),
                    ('bn{}'.format(postfix), nn.BatchNorm2d(out_channels)),
                    ('relu{}'.format(postfix), nn.ReLU(relu_inplace))
                ]
            in_c = out_channels

        kernel_size = kwargs.pop('pool_k_size', (2, 2))
        stride = kwargs.pop('pool_stride', (2, 2))
        ceil_mode = kwargs.pop('pool_ceil_mode', False)
        padding = kwargs.pop('pool_padding', 0)
        # append maxpooling
        layers += [
            ('pool{}'.format(order), nn.MaxPool2d(kernel_size, stride=stride, ceil_mode=ceil_mode, padding=padding))
        ]

        return layers

    @staticmethod
    def one(postfix, *args, relu_inplace=False, **kwargs):
        batch_norm = kwargs.pop('batch_norm', Conv2dRelu.batch_norm)
        if not batch_norm:
            return [
                ('conv{}'.format(postfix), nn.Conv2d(*args, **kwargs)),
                ('relu{}'.format(postfix), nn.ReLU(inplace=relu_inplace))
            ]
        else:
            out_channels = kwargs.pop('out_channels', args[1])
            return [
                ('conv{}'.format(postfix), nn.Conv2d(*args, **kwargs)),
                ('bn{}'.format(postfix), nn.BatchNorm2d(out_channels)),
                ('relu{}'.format(postfix), nn.ReLU(inplace=relu_inplace))
            ]

