from torch import nn
import torch

from .._utils import check_instance
from ..core.boxes.dbox import *
from ..core.layers import Predictor, ConvRelu
from ..core.inference import InferenceBox
from ssd.core.boxes.codec import Codec

class SSDBase(nn.Module):
    _codec: Codec

    feature_layers: nn.ModuleDict
    l2norm_layers: nn.ModuleDict
    localization_layers: nn.ModuleDict
    confidence_layers: nn.ModuleDict

    defaultBox: DefaultBoxBase
    predictor: Predictor
    inferenceBox: InferenceBox

    classifier_source_names: tuple
    dbox_nums: tuple

    def __init__(self, class_nums, input_shape, batch_norm, codec):
        """
        :param class_nums: int, class number
        :param input_shape: tuple, 3d and (height, width, channel)
        :param batch_norm: bool, whether to add batch normalization layers
        :param codec: Codec,
        """
        super().__init__()

        self.class_nums = class_nums
        assert len(input_shape) == 3, "input dimension must be 3"
        assert input_shape[0] == input_shape[1], "input must be square size"
        self.input_shape = input_shape
        self.batch_norm = batch_norm
        self._codec = codec

        self._isbuilt_layer = False
        self._isbuilt_box = False
        self._isbuilt_infBox = False
        self._called_learn_inferr = False

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

    @property
    def encoder(self):
        return self._codec.encoder
    @property
    def decoder(self):
        return self._codec.decoder

    def _build_layers(self, features, locs, confs, l2norms):
        self.feature_layers = check_instance('feature_layers', features, nn.ModuleDict)
        self.l2norm_layers = check_instance('l2norm_layers', l2norms, nn.ModuleDict)
        self.localization_layers = check_instance('localization_layers', locs, nn.ModuleDict)
        self.confidence_layers = check_instance('confidence_layers', confs, nn.ModuleDict)

        self.init_weights()

        self._isbuilt_layer = True

    def _build_defaultBox(self, defaultBox, classifier_source_names):
        if not self._isbuilt_layer:
            raise NotImplementedError('Call _build_layers first!')

        self.defaultBox = defaultBox.build(self.feature_layers, classifier_source_names, self.localization_layers)
        self.predictor = Predictor(self.defaultBox.total_dboxes_nums, self.class_nums)

        self.classifier_source_names = tuple(classifier_source_names)

        self._isbuilt_box = True

    def _build_inferenceBox(self, inferenceBox):
        self.inferenceBox = check_instance('inferenceBox', inferenceBox, InferenceBox)

        self._isbuilt_infBox = True


    def forward(self, x):
        if not self.isBuilt:
            raise NotImplementedError('call _build_layers, _build_defaultBox and _build_infBox first')

        if not self._called_learn_inferr:
            raise NotImplementedError('call learn or infer first')

    def learn(self, x, gts):
        if not self.training:
            raise NotImplementedError("model hasn\'t built as train. Call \'train()\'")
        self._called_learn_inferr = True

    def infer(self, image, conf_threshold=0.01, toNorm=False,
              rgb_means=(103.939, 116.779, 123.68), rgb_stds=(1.0, 1.0, 1.0),
              visualize=False):
        """
        :param image: list of ndarray or Tensor, ndarray or Tensor
        :param conf_threshold: float or None, if it's None, default value (0.01) will be passed
        :param toNorm: bool, whether to normalize passed image
        :param rgb_means: number, tuple,
        :param rgb_stds: number, tuple,
        :param visualize: bool,
        :return:
        """
        if self.training:
            raise NotImplementedError("model hasn\'t built as test. Call \'eval()\'")

        if isinstance(image, list):
            img = []
            for im in image:
                if isinstance(im, np.ndarray):
                    im = torch.tensor(im, requires_grad=False)
                    img += im.permute((2, 0, 1))
                elif isinstance(im, torch.Tensor):
                    img += im
                else:
                    raise ValueError('Invalid image type')
            img = torch.stack(img)
        elif isinstance(image, np.ndarray):
            img = torch.tensor(image, requires_grad=False)
            if img.ndim == 3:
                img = img.permute((2, 0, 1))
            elif img.ndim == 4:
                img = img.permute((0, 3, 1, 2))

        elif isinstance(image, torch.Tensor):
            img = image
        else:
            raise ValueError('Invalid image type')

        if img.ndim == 3:
            img = img.unsqueeze(0) # shape = (1, ?, ?, ?)



        # shape = (1, 3, 1, 1)
        rgb_means = torch.tensor(rgb_means).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        rgb_stds = torch.tensor(rgb_stds).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        if toNorm:
            normed_img = (img - rgb_means) / rgb_stds
            orig_img = img
        else:
            normed_img = img
            orig_img = img*rgb_stds + rgb_means


        input_shape = np.array(self.input_shape)[np.array([2, 0, 1])]
        if list(img.shape[1:]) != input_shape.tolist():
            raise ValueError('image shape was not same as input shape: {}, but got {}'.format(input_shape.tolist(), list(img.shape[1:])))

        self._called_learn_inferr = True

        return normed_img, orig_img

    # device management
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if not self.isBuilt:
            raise NotImplementedError('call _build_layers, _build_defaultBox and _build_infBox first')
        self.defaultBox.dboxes.to(*args, **kwargs)

        return self
    def cuda(self, device=None):
        self.defaultBox.dboxes = self.defaultBox.dboxes.cuda(device)

        return super().cuda(device)

    # weights management
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

    def init_weights(self):
        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                #nn.init.kaiming_norma l_(module.weight, mode='fan_out', nonlinearity='relu')
                #if module.bias is not None:
                #    nn.init.constant_(module.bias, 0)

                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, ConvRelu):
                nn.init.xavier_uniform_(module.conv.weight)
                if module.conv.bias is not None:
                    nn.init.constant_(module.conv.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal(module.weight, 0, 1e-2)
                nn.init.constant_(module.bias, 0)