from torch import nn
import torch

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

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