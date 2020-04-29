from torch import nn
import torch
import numpy as np

from .._utils import check_instance
from ..core.boxes import DefaultBox
from ..core.layers import Predictor
from ..core.inference import InferenceBox

class SSDBase(nn.Module):
    feature_layers: nn.ModuleDict
    l2norm_layers: nn.ModuleDict
    localization_layers: nn.ModuleDict
    confidence_layers: nn.ModuleDict

    defaultBox: DefaultBox
    predictor: Predictor
    inferenceBox: InferenceBox

    classifier_source_names: tuple
    dbox_nums: tuple

    def __init__(self, class_nums, input_shape, batch_norm):
        """
        :param class_nums: int, class number
        :param input_shape: tuple, 3d and (height, width, channel)
        :param batch_norm: bool, whether to add batch normalization layers
        """
        super().__init__()

        self.class_nums = class_nums
        assert len(input_shape) == 3, "input dimension must be 3"
        assert input_shape[0] == input_shape[1], "input must be square size"
        self.input_shape = input_shape
        self.batch_norm = batch_norm

        self._isbuilt_layer = False
        self._isbuilt_box = False
        self._isbuilt_infBox = False

    @property
    def input_height(self):
        return self.input_shape[0]
    @property
    def input_width(self):
        return self.input_shape[1]
    @property
    def input_channel(self):
        return self.input_shape[2]
    @property
    def isBuilt(self):
        return self._isbuilt_layer and self._isbuilt_box and self._isbuilt_infBox


    def _build_layers(self, features, locs, confs, l2norms):
        self.feature_layers = check_instance('feature_layers', features, nn.ModuleDict)
        self.l2norm_layers = check_instance('l2norm_layers', l2norms, nn.ModuleDict)
        self.localization_layers = check_instance('localization_layers', locs, nn.ModuleDict)
        self.confidence_layers = check_instance('confidence_layers', confs, nn.ModuleDict)

        self._isbuilt_layer = True

    def _build_defaultBox(self, classifier_source_names, dbox_nums):
        if not self._isbuilt_layer:
            raise NotImplementedError('Call _build_layers first!')

        self.defaultBox = DefaultBox(img_shape=self.input_shape)
        self.defaultBox = self.defaultBox.build(self.feature_layers, classifier_source_names, self.localization_layers, dbox_nums)
        self.predictor = Predictor(self.defaultBox.total_dboxes_nums, self.class_nums)

        self.classifier_source_names = tuple(classifier_source_names)

        self._isbuilt_box = True

    def _build_inferenceBox(self, inferenceBox):
        self.inferenceBox = check_instance('inferenceBox', inferenceBox, InferenceBox)

        self._isbuilt_infBox = True

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
        if not self.isBuilt:
            raise NotImplementedError('call _build_layers and _build_defaultBox first')

    def inference(self, image, visualize=False, convert_torch=False):
        if self.training:
            raise NotImplementedError("model hasn\'t built as test. Call \'eval()\'")


    def load_vgg_weights(self):
        """
        load pre-trained weights for vgg, which means load weights partially.
        After calling this method, vgg_ssd.pth or vgg_bn_ssd.pth will be saved
        :return:
        """
        pass

    def load_weights(self, path):
        """
        :param path: str
        :return:
        """
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))