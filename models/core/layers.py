from torch import nn
import torch

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

    # Note that pytorch's dimension order is batch_size, channels, height, width
    def forward(self, x):
        # |x|_2
        # square element-wise, sum along channel and square element-wise
        norm_x = torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt()
        # normalize (x^)
        x = torch.div(x, norm_x)
        return self.gamma * x


class DefaultBox(object):
    def __init__(self, img_shape=(300, 300, 3), scale_range=(0.2, 0.9), aspect_ratios=(1, 2, 3), clip=True):
        """
        :param img_shape: tuple, must be 3d
        :param scale_range: tuple of scale range, first element means minimum scale and last is maximum one
        :param aspect_ratios: tuple of aspect ratio, note that all of elements must be greater than 0
        :param clip: bool, whether to force to be 0 to 1
        """
        super().__init__()
        # self.flatten = Flatten()

        assert len(img_shape) == 3, "input image dimension must be 3"
        assert img_shape[0] == img_shape[1], "input image's height and width must be same"
        self._img_shape = img_shape
        self._scale_range = scale_range
        assert np.where(np.array(aspect_ratios) <= 0)[0].size <= 0, "aspect must be greater than 0"
        self._aspect_ratios = aspect_ratios
        self._clip = clip

        self.dboxes = None

    @property
    def scale_min(self):
        return self._scale_range[0]

    @property
    def scale_max(self):
        return self._scale_range[1]

    @property
    def img_height(self):
        return self._img_shape[0]

    @property
    def img_width(self):
        return self._img_shape[1]

    @property
    def img_channels(self):
        return self._img_shape[2]

    @property
    def total_dboxes_nums(self):
        if self.dboxes is not None:
            return self.dboxes.shape[0]
        else:
            raise NotImplementedError('must call build')

    def get_scale(self, k, m):
        return self.scale_min + (self.scale_max - self.scale_min) * (k - 1) / (m - 1)

    def build(self, feature_layers, classifier_source_names, classifier_layers, dbox_nums):
        # this x is pseudo Tensor to get feature's map size
        x = torch.tensor((), dtype=torch.float).new_zeros((1, self.img_channels, self.img_height, self.img_width))

        features = []
        i = 1
        for name, layer in feature_layers.items():
            x = layer(x)
            # get features by feature map convolution
            if name in classifier_source_names:
                feature = classifier_layers['feature{0}'.format(i)](x)
                features.append(feature)
                # print(features[-1].shape)
                i += 1

        self.dboxes = self.forward(features, dbox_nums)
        return self

    def forward(self, features, dbox_nums):
        """
        :param features: list of Tensor, Tensor's shape is (batch, c, h, w)
        :param dbox_nums: list of dbox numbers
        :return: dboxes(Tensor)
                dboxes' shape is (position, cx, cy, w, h)

                bellow is deprecated to LocConf
                features' shape is (position, class)
        """
        dboxes = []
        # ret_features = []
        m = len(features)
        for k, feature, dbox_num in zip(range(1, m + 1), features, dbox_nums):
            _, _, fmap_h, fmap_w = feature.shape
            assert fmap_w == fmap_h, "feature map's height and width must be same"
            # f_k = np.sqrt(fmap_w * fmap_h)

            # get cx and cy
            # (cx, cy) = ((i+0.5)/f_k, (j+0.5)/f_k)

            # / f_k
            step_i, step_j = (np.arange(fmap_w) + 0.5) / fmap_w, (np.arange(fmap_h) + 0.5) / fmap_h
            # ((i+0.5)/f_k, (j+0.5)/f_k) for all i,j
            cx, cy = np.meshgrid(step_i, step_j)
            # cx, cy's shape (fmap_w, fmap_h) to (fmap_w*fmap_h, 1)
            cx, cy = cx.reshape(-1, 1), cy.reshape(-1, 1)
            total_dbox_num = cx.size
            for i in range(int(dbox_num / 2)):
                # normal aspect
                aspect = self._aspect_ratios[i]
                scale = self.get_scale(k, m)
                box_w, box_h = scale * np.sqrt(aspect), scale / np.sqrt(aspect)
                box_w, box_h = np.broadcast_to([box_w], (total_dbox_num, 1)), np.broadcast_to([box_h],
                                                                                              (total_dbox_num, 1))
                dboxes += [np.concatenate((cx, cy, box_w, box_h), axis=1)]

                # reciprocal aspect
                aspect = 1 / aspect
                if aspect == 1:  # if aspect is 1, scale = sqrt(s_k * s_k+1)
                    scale = np.sqrt(scale * self.get_scale(k + 1, m))
                box_w, box_h = scale * np.sqrt(aspect), scale / np.sqrt(aspect)
                box_w, box_h = np.broadcast_to([box_w], (total_dbox_num, 1)), np.broadcast_to([box_h],
                                                                                              (total_dbox_num, 1))
                dboxes += [np.concatenate((cx, cy, box_w, box_h), axis=1)]

            # ret_features += [self.flatten(feature)]

        dboxes = np.concatenate(dboxes, axis=0)
        dboxes = torch.from_numpy(dboxes).float()

        # ret_features = torch.cat(ret_features, dim=1)
        if self._clip:
            dboxes = dboxes.clamp(min=0, max=1)

        return dboxes  # , ret_features


class Predictor(nn.Module):
    def __init__(self, total_dbox_nums, class_nums):
        super().__init__()
        self.flatten = Flatten()

        self._total_dox_nums = total_dbox_nums
        self._class_nums = class_nums

    def forward(self, features):
        """
        :param features: list of Tensor, Tensor's shape is (batch, c, h, w)
        :return: predicts: localization and confidence Tensor, shape is (batch, total_dbox_num, 4+class_nums)

                 below is deprecated
                 localizations: Tensor, shape is (batch, total_dboxnum, 4)
                 confidences: Tensor, shape is (batch, total_dboxnum, class_nums)
        """
        predicts = []
        for feature in features:
            predicts += [self.flatten(feature)]
        predicts = torch.cat(predicts, dim=1).reshape((-1, self._total_dox_nums, 4 + self._class_nums))

        return predicts
        # return predicts[:, :, :4], predicts[:, :, 4:]

def conv2dRelu_block(order, block_num, in_channels, out_channels, batch_norm=True, **kwargs):
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

    in_c = in_channels
    layers = []
    # append conv block
    for bnum in range(block_num):
        postfix = '{0}_{1}'.format(order, bnum + 1)
        if not batch_norm:
            layers += [
                ('conv{}'.format(postfix), nn.Conv2d(in_c, out_channels, kernel_size, stride=stride, padding=padding)),
                ('relu{}'.format(postfix), nn.ReLU(True))
            ]
        else:
            layers += [
                ('conv{}'.format(postfix), nn.Conv2d(in_c, out_channels, kernel_size, stride=stride, padding=padding)),
                ('bn{}'.format(postfix), nn.BatchNorm2d(out_channels)),
                ('relu{}'.format(postfix), nn.ReLU(True))
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


def conv2dRelu(postfix, *args, relu_inplace=False, **kwargs):
    return [
        ('conv{}'.format(postfix), nn.Conv2d(*args, **kwargs)),
        ('relu{}'.format(postfix), nn.ReLU(inplace=relu_inplace))
    ]

