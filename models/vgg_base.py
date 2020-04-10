import torch
from torch import nn
from torchvision.models.utils import load_state_dict_from_url
from .core.layers import Flatten
from collections import OrderedDict

import os

class VGG(nn.Module):
    """
    :param
        load_model  : path, Bool or None

    """
    def __init__(self, model_name, conv_layers, class_nums=1000, load_model=None):
        super().__init__()
        self.model_name = model_name

        self.conv_layers = conv_layers
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, class_nums)
        )

        if isinstance(load_model, bool) and load_model:
            model_url = get_model_url(self.model_name)

            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
            pretrained_state_dict = load_state_dict_from_url(model_url, model_dir=model_dir)
            model_state_dict = self.state_dict()

            if len(model_state_dict) != len(pretrained_state_dict):
                raise ValueError('cannot load model due to unsame model architecture')

            # rename
            renamed = []
            # note that model_state_dict and pretrained_state_dict are ordereddict
            for (pre_key, mod_key) in zip(pretrained_state_dict.keys(), model_state_dict.keys()):
                renamed.append((mod_key, pretrained_state_dict[pre_key]))

            pretrained_state_dict = OrderedDict(renamed)
            self.load_state_dict(pretrained_state_dict)
            torch.save(self.state_dict(), os.path.join(model_dir, '{}'.format(model_url.split('/')[-1])))

        elif isinstance(load_model, str):
            self.load_state_dict(torch.load(load_model))

        elif load_model is None:
            self._init_weights()



    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 1e-2)
                nn.init.constant_(module.bias, 0)


    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.fc(x)


def get_model_url(name):
    model_urls = {
        'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
        'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
        'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    }

    return model_urls[name]