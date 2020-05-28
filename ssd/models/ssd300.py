from ..core.layers import *
from ssd._utils import weights_path
from ..core.inference import InferenceBox, toVisualizeImg, toVisualizeRectangleimg
from ..models.vgg_base import get_model_url
from .ssd_base import SSDBase
from ..core.boxes import *


from torch import nn
from torchvision.models.utils import load_state_dict_from_url
import torch
from collections import OrderedDict
import logging

# defalut boxes number for each feature map
#_dbox_num_per_fpixel = [4, 6, 6, 6, 4, 4]
_aspect_ratios=((1, 2), (1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2), (1, 2))

# classifier's source layers
# consists of conv4_3, conv7, conv8_2, conv9_2, conv10_2, conv11_2
_classifier_source_names = ['conv4_3', 'conv7', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2']
_l2norm_source_names = ['conv4_3']
class SSD300(SSDBase):
    def __init__(self, class_nums, input_shape=(300, 300, 3), batch_norm=False):
        """
        :param class_nums: int, class number
        :param input_shape: tuple, 3d and (height, width, channel)
        :param batch_norm: bool, whether to add batch normalization layers
        """
        codec = Codec(norm_means=(0, 0, 0, 0), norm_stds=(0.1, 0.1, 0.2, 0.2))

        super().__init__(class_nums, input_shape, batch_norm, codec)

        Conv2d.batch_norm = self.batch_norm
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

        feature_layers = nn.ModuleDict(OrderedDict(vgg_layers + extra_layers))

        # l2norm
        l2norm_layers = []
        for i, sourcename in enumerate(_l2norm_source_names):
            assert sourcename in _classifier_source_names, "must be element of _classifier_source_names"
            l2norm_layers += [('l2norm_{}'.format(i + 1), L2Normalization(feature_layers[sourcename].out_channels, gamma=20))]
        l2norm_layers = nn.ModuleDict(OrderedDict(l2norm_layers))

        # loc and conf
        in_channels = tuple(feature_layers[sourcename].out_channels for sourcename in _classifier_source_names)

        _dbox_num_per_fpixel = [len(_aspect_ratio)*2 for _aspect_ratio in _aspect_ratios]
        # loc
        out_channels = tuple(dbox_num * 4 for dbox_num in _dbox_num_per_fpixel)
        localization_layers = [
            *Conv2d.block('_loc', len(_dbox_num_per_fpixel), in_channels, out_channels, kernel_size=(3, 3), padding=1,
                          batch_norm=False)
        ]

        # conf
        out_channels = tuple(dbox_num * class_nums for dbox_num in _dbox_num_per_fpixel)
        confidence_layers = [
            *Conv2d.block('_conf', len(_dbox_num_per_fpixel), in_channels, out_channels, kernel_size=(3, 3),
                          padding=1, batch_norm=False)
        ]

        # build layer
        localization_layers = nn.ModuleDict(OrderedDict(localization_layers))
        confidence_layers = nn.ModuleDict(OrderedDict(confidence_layers))
        self._build_layers(feature_layers, localization_layers, confidence_layers, l2norm_layers)

        # build default box
        defaultBox = DBoxSSD300Original(scale_conv4_3=0.1, aspect_ratios=_aspect_ratios, scale_range=(0.2, 0.9))
        self._build_defaultBox(defaultBox, _classifier_source_names)

        # build inference box
        inferenceBox = InferenceBox(conf_threshold=0.01, iou_threshold=0.45, topk=200)
        self._build_inferenceBox(inferenceBox)

    def forward(self, x):
        """
        TODO: check if ssd300 and ssd512's forward function are common
        :param x: Tensor, input Tensor whose shape is (batch, c, h, w)
        :return:
            predicts: localization and confidence Tensor, shape is (batch, total_dbox_num, 4+class_nums)
        """
        super().forward(x)

        locs, confs = [], []

        feature_i, l2norm_i = 1, 1
        for name, layer in self.feature_layers.items():
            x = layer(x)

            # get features by feature map convolution
            if name in _classifier_source_names:
                source = x
                # l2norm
                if name in _l2norm_source_names:
                    source = self.l2norm_layers['l2norm_{}'.format(l2norm_i)](source)
                    l2norm_i += 1

                loc = self.localization_layers['conv_loc_{0}'.format(feature_i)](source)
                locs.append(loc)

                conf = self.confidence_layers['conv_conf_{0}'.format(feature_i)](source)
                confs.append(conf)
                #print(features[-1].shape)
                feature_i += 1

        predicts = self.predictor(locs, confs)
        return predicts

    def learn(self, x, gts):
        """
        :param x: Tensor, input Tensor whose shape is (batch, c, h, w)
        :param gts: Tensor, shape is (batch*bbox_nums(batch), 1+4+class_nums) = [[img's_ind, cx, cy, w, h, p_class,...],..
        :return:
            pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
            predicts: localization and confidence Tensor, shape is (batch, total_dbox_num, 4+class_nums)
            dboxes: Tensor, default boxes Tensor whose shape is (total_dbox_nums, 4)`
        """
        super().learn(x, gts)

        batch_num = x.shape[0]

        pos_indicator, gts = self.encoder(gts, self.defaultBox.dboxes.clone(), batch_num)
        predicts = self(x)

        return pos_indicator, predicts, gts

    def infer(self, image, conf_threshold=None, toNorm=False,
              rgb_means=(103.939, 116.779, 123.68), rgb_stds=(1.0, 1.0, 1.0),
              visualize=False, visualize_classes=None):

        normed_img, orig_img = super().infer(image, conf_threshold, toNorm, rgb_means, rgb_stds, visualize)

        if conf_threshold is None:
            conf_threshold = 0.6 if visualize else 0.01

        # predict
        predicts = self(normed_img)
        infers = self.inferenceBox(predicts, self.defaultBox.dboxes.clone(), conf_threshold)

        img_num = normed_img.shape[0]
        if visualize:
            return infers, [toVisualizeRectangleimg(orig_img[i], infers[i][:, 1:], verbose=False) for i in range(img_num)]
        else:
            return infers

    def load_vgg_weights(self):
        model_dir = weights_path(__file__, _root_num=2, dirname='weights')

        model_url = get_model_url('vgg16' if not self.batch_norm else 'vgg16_bn')
        pretrained_state_dict = load_state_dict_from_url(model_url, model_dir=model_dir)

        model_state_dict = self.state_dict()

        renamed = []
        pre_keys, mod_keys = list(pretrained_state_dict.keys()), list(model_state_dict.keys())
        if not self.batch_norm:
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
