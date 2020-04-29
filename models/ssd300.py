from .core.layers import *
from .core.boxes import DefaultBox
from .core.utils import _weights_path
from .core.inference import InferenceBox, toVisualizeImg
from .vgg_base import get_model_url

from torch import nn
from torchvision.models.utils import load_state_dict_from_url
import torch
from collections import OrderedDict
import logging

# defalut boxes number for each feature map
_dbox_nums = [4, 6, 6, 6, 4, 4]

# classifier's source layers
# consists of conv4_3, conv7, conv8_2, conv9_2, conv10_2, conv11_2
_classifier_source_names = ['conv4_3', 'conv7', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2']
_l2norm_source_names = ['conv4_3']
class SSD300(nn.Module):
    def __init__(self, class_nums, input_shape=(300, 300, 3), batch_norm=False):
        """
        :param class_nums: int, class number
        :param input_shape: tuple, 3d and (height, width, channel)
        """
        super().__init__()

        self.class_nums = class_nums
        assert len(input_shape) == 3, "input dimension must be 3"
        assert input_shape[0] == input_shape[1], "input must be square size"
        self.input_shape = input_shape
        self._batch_norm = batch_norm

        Conv2d.batch_norm = self._batch_norm
        vgg_layers = [
            *Conv2d.relu_block('1', 2, self.input_channel, 64),

            *Conv2d.relu_block('2', 2, 64, 128),

            *Conv2d.relu_block('3', 3, 128, 256, pool_ceil_mode=True),

            *Conv2d.relu_block('4', 3, 256, 512),

            *Conv2d.relu_block('5', 3, 512, 512, pool_k_size=(3, 3), pool_stride=(1, 1), pool_padding=1), # replace last maxpool layer's kernel and stride

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
            *Conv2d.relu_one('11_2', 128, 256, kernel_size=(3, 3), batch_norm=False), # if batch_norm = True, error is thrown. last layer's channel == 1 may be caused
        ]

        self.feature_layers = nn.ModuleDict(OrderedDict(vgg_layers + extra_layers))

        # l2norm
        l2norm_layers = []
        for i, sourcename in enumerate(_l2norm_source_names):
            l2norm_layers += [('l2norm_{}'.format(i + 1), L2Normalization(self.feature_layers[sourcename].out_channels, gamma=20))]
        self.l2norm_layers = nn.ModuleDict(OrderedDict(l2norm_layers))

        # loc and conf
        in_channels = tuple(self.feature_layers[sourcename].out_channels for sourcename in _classifier_source_names)

        # loc
        out_channels = tuple(dbox_num * 4 for dbox_num in _dbox_nums)
        localization_layers = [
            *Conv2d.block('_loc', len(_dbox_nums), in_channels, out_channels, kernel_size=(3, 3), padding=1,
                                 batch_norm=False)
        ]

        # conf
        out_channels = tuple(dbox_num * class_nums for dbox_num in _dbox_nums)
        confidence_layers = [
            *Conv2d.block('_conf', len(_dbox_nums), in_channels, out_channels, kernel_size=(3, 3),
                                 padding=1, batch_norm=False)
        ]

        self.localization_layers = nn.ModuleDict(OrderedDict(localization_layers))
        self.confidence_layers = nn.ModuleDict(OrderedDict(confidence_layers))

        self.defaultBox = DefaultBox(img_shape=self.input_shape).build(self.feature_layers, _classifier_source_names, self.localization_layers, _dbox_nums)
        self.predictor = Predictor(self.defaultBox.total_dboxes_nums, self.class_nums)
        self.inferenceBox = None


    def init_weights(self):
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
        """
        :param x: Tensor, input Tensor whose shape is (batch, c, h, w)
        :return:
            predicts: localization and confidence Tensor, shape is (batch, total_dbox_num, 4+class_nums)
            dboxes: Tensor, default boxes Tensor whose shape is (total_dbox_nums, 4)`
        """
        locs, confs = [], []

        feature_i, l2norm_i = 1, 1
        for name, layer in self.feature_layers.items():
            x = layer(x)
            # l2norm
            if name in _l2norm_source_names:
                x = self.l2norm_layers['l2norm_{}'.format(l2norm_i)](x)

            # get features by feature map convolution
            if name in _classifier_source_names:
                loc = self.localization_layers['conv_loc_{0}'.format(feature_i)](x)
                locs.append(loc)

                conf = self.confidence_layers['conv_conf_{0}'.format(feature_i)](x)
                confs.append(conf)
                #print(features[-1].shape)
                feature_i += 1

        predicts = self.predictor(locs, confs)

        return predicts, self.defaultBox.dboxes.clone()

    @property
    def input_height(self):
        return self.input_shape[0]
    @property
    def input_width(self):
        return self.input_shape[1]
    @property
    def input_channel(self):
        return self.input_shape[2]

    def build_test(self, path):
        self.load_weights(path)
        self.inferenceBox = InferenceBox(conf_threshold=0.01, iou_threshold=0.45, topk=200)

        # set test mode
        self.eval()
        # print(self.training)
        # >> False
        return self
    def build_train(self, path=None):
        # set train mode
        self.train()
        if path:
            self.load_weights(path)

        return self

    def inference(self, image, visualize=False, convert_torch=False):
        if self.training:
            raise NotImplementedError("model hasn\'t built as test. Call \'build_test\'")
        if isinstance(image, list):
            img = torch.stack(image)
        elif isinstance(image, np.ndarray):
            img = torch.tensor(image, requires_grad=False)
        elif not isinstance(image, torch.Tensor):
            raise ValueError('Invalid image type')

        if img.ndim == 3:
            img = img.unsqueeze(0) # shape = (1, ?, ?, ?)
        if convert_torch:
            img = img.permute((0, 3, 1, 2))

        input_shape = np.array(self.input_shape)[np.array([2, 0, 1])]
        if list(img.shape[1:]) != input_shape.tolist():
            raise ValueError('image shape was not same as input shape: {}, but got {}'.format(input_shape.tolist(), list(img.shape[1:])))

        # predict
        predicts, dboxes = self(img)
        infers = self.inferenceBox(predicts, dboxes)

        if visualize:
            return

    def load_vgg_weights(self):
        """
        load pre-trained weights for vgg, which means load weights partially.
        After calling this method, vgg_ssd.pth or vgg_bn_ssd.pth will be saved
        :return:
        """
        """
        if not self._batch_norm and os.path.exists('./weights/vgg_ssd.pth'):
            self.load_weights('./weights/vgg_ssd.pth')
            return

        if self._batch_norm and os.path.exists('./weights/vgg_bn_ssd.pth'):
            self.load_weights('./weights/vgg_bn_ssd.pth')
            return
        """
        model_dir = _weights_path(__file__, _root_num=1, dirname='weights')

        model_url = get_model_url('vgg16' if not self._batch_norm else 'vgg16_bn')
        pretrained_state_dict = load_state_dict_from_url(model_url, model_dir=model_dir)

        model_state_dict = self.state_dict()

        renamed = []
        pre_keys, mod_keys = list(pretrained_state_dict.keys()), list(model_state_dict.keys())
        if not self._batch_norm:
            # ssd300 and vgg16 have common 26 (13 weights and biases) layers from first
            for (pre_key, mod_key) in zip(pre_keys[:26], mod_keys[:26]):
                renamed += [(mod_key, pretrained_state_dict[pre_key])]
        else:
            # ssd300(bn) and vgg16_bn have common 78 (weights, biases, running_means and running_vars) layers from first
            for (pre_key, mod_key) in zip(pre_keys[:78], mod_keys[:78]):
                renamed += [(mod_key, pretrained_state_dict[pre_key])]
        # set vgg layer's parameters
        model_state_dict.update(OrderedDict(renamed))
        self.load_state_dict(model_state_dict)

        logging.info("model loaded")

    def load_weights(self, path):
        """
        :param path: str
        :return:
        """
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))