from torch import nn

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class Conv2dRelu_block(nn.Module):
    def __init__(self, block_num, in_channels, out_channels, batch_norm=True, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = kwargs.pop('conv_k_size', (3, 3))
        stride = kwargs.pop('conv_stride', (1, 1))
        padding = kwargs.pop('conv_padding', 1)

        in_c = in_channels
        layers = []
        # append conv block
        for _ in range(block_num):
            if not batch_norm:
                layers += [nn.Conv2d(in_c, out_channels, kernel_size, stride=stride, padding=padding),
                        nn.ReLU(True)]
            else:
                layers += [nn.Conv2d(in_c, out_channels, kernel_size, stride=stride, padding=padding),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(True)]
            in_c = out_channels

        kernel_size = kwargs.pop('pool_k_size', (2, 2))
        stride = kwargs.pop('pook_stride', (2, 2))

        # append maxpooling
        layers += [nn.MaxPool2d(kernel_size, stride)]

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

class Conv2dRelu(nn.Conv2d):
    def __init__(self, *args, relu_inplace=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU(relu_inplace)


    def forward(self, x):
        x = super().forward(x)
        return self.relu(x)

class L2Normalization(nn.Module):
    def __init__(self):
        super().__init__()