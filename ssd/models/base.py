from torchvision.models.utils import load_state_dict_from_url
import abc
import logging
import torch

from .._utils import check_instance, _check_ins
from ..core.boxes.dbox import *
from ..core.layers import *
from ..core.inference import InferenceBox
from ..core.boxes.codec import Codec
from .._utils import weights_path, _check_norm
from ..core.inference import InferenceBox, toVisualizeImg, toVisualizeRectangleimg
from ..models.vgg_base import get_model_url

class ObjectDetectionModelBase(nn.Module):

    def __init__(self, class_nums, input_shape, batch_norm):
        """
        :param class_nums: int, class number
        :param input_shape: tuple, 3d and (height, width, channel)
        :param batch_norm: bool, whether to add batch normalization layers
        """
        super().__init__()

        self._class_nums = class_nums
        assert len(input_shape) == 3, "input dimension must be 3"
        assert input_shape[0] == input_shape[1], "input must be square size"
        self._input_shape = input_shape
        self._batch_norm = batch_norm

    @property
    def input_height(self):
        return self._input_shape[0]
    @property
    def input_width(self):
        return self._input_shape[1]
    @property
    def input_channel(self):
        return self._input_shape[2]

    @property
    def class_nums(self):
        return self._class_nums
    @property
    def batch_norm(self):
        return self._batch_norm


    @abc.abstractmethod
    def learn(self, x, targets):
        pass
    @abc.abstractmethod
    def infer(self, *args, **kwargs):
        pass


    def load_weights(self, path):
        """
        :param path: str
        :return:
        """
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))


    def init_weights(self):
        raise NotImplementedError()

class SSDTrainConfig(object):
    def __init__(self, **kwargs):
        self.class_nums = kwargs.get('class_nums')

        input_shape = kwargs.get('input_shape')
        assert len(input_shape) == 3, "input dimension must be 3"
        assert input_shape[0] == input_shape[1], "input must be square size"
        self.input_shape = input_shape

        self.batch_norm = _check_ins('batch_norm', kwargs.get('batch_norm'), bool)

        self.aspect_ratios = _check_ins('aspect_ratios', kwargs.get('aspect_ratios'), (tuple, list))
        self.classifier_source_names = _check_ins('classifier_source_names', kwargs.get('classifier_source_names'), (tuple, list))
        self.addon_source_names = _check_ins('addon_source_names', kwargs.get('addon_source_names'), (tuple, list))

        self.codec_means = _check_ins('codec_means', kwargs.get('codec_means'), (tuple, list, float, int))
        self.codec_stds = _check_ins('codec_stds', kwargs.get('codec_stds'), (tuple, list, float, int))

        self.rgb_means = _check_ins('rgb_means', kwargs.get('rgb_means', (0.485, 0.456, 0.406)), (tuple, list, float, int))
        self.rgb_stds = _check_ins('rgb_stds', kwargs.get('rgb_stds', (0.229, 0.224, 0.225)), (tuple, list, float, int))

    @property
    def input_height(self):
        return self.input_shape[0]
    @property
    def input_width(self):
        return self.input_shape[1]
    @property
    def input_channel(self):
        return self.input_shape[2]

class SSDValConfig(object):
    def __init__(self, **kwargs):
        self.val_conf_threshold = _check_ins('val_conf_threshold', kwargs.get('val_conf_threshold', 0.01), float)
        self.vis_conf_threshold = _check_ins('vis_conf_threshold', kwargs.get('vis_conf_threshold', 0.6), float)
        self.iou_threshold = _check_ins('iou_threshold', kwargs.get('iou_threshold', 0.45), float)
        self.topk = _check_ins('topk', kwargs.get('topk', 200), int)

class SSDBase(ObjectDetectionModelBase):
    defaultBox: DefaultBoxBase
    inferenceBox: InferenceBox
    _train_config: SSDTrainConfig
    _val_config: SSDValConfig


    feature_layers: nn.ModuleDict
    localization_layers: nn.ModuleDict
    confidence_layers: nn.ModuleDict
    addon_layers: nn.ModuleDict

    def __init__(self, train_config, val_config, defaultBox, **build_kwargs):
        """
        :param train_config: SSDTrainConfig
        :param val_config: SSDValConfig
        :param defaultBox: instance inheriting DefaultBoxBase
        """
        self._train_config = _check_ins('train_config', train_config, SSDTrainConfig)
        self._val_config = _check_ins('val_config', val_config, SSDValConfig)
        super().__init__(train_config.class_nums, train_config.input_shape, train_config.batch_norm)

        self.codec = Codec(norm_means=self.codec_means, norm_stds=self.codec_stds)
        self.defaultBox = _check_ins('defaultBox', defaultBox, DefaultBoxBase)

        self.predictor = Predictor(self.class_nums)
        self.inferenceBox = InferenceBox(conf_threshold=self.val_conf_threshold, iou_threshold=self.iou_threshold, topk=self.topk, decoder=self.decoder)

        self.build(**build_kwargs)

    @property
    def isBuilt(self):
        return hasattr(self, 'feature_layers') and\
               hasattr(self, 'localization_layers') and\
               hasattr(self, 'confidence_layers')

    ### build ###
    @abc.abstractmethod
    def build_feature(self, **kwargs):
        pass
    @abc.abstractmethod
    def build_addon(self, **kwargs):
        pass
    @abc.abstractmethod
    def build_classifier(self, **kwargs):
        pass

    ### codec ###
    @property
    def encoder(self):
        return self.codec.encoder
    @property
    def decoder(self):
        return self.codec.decoder

    ### default box ###
    @property
    def dboxes(self):
        return self.defaultBox.dboxes
    @property
    def total_dboxes_num(self):
        return self.defaultBox.total_dboxes_nums

    ### train_config ###
    @property
    def aspect_ratios(self):
        return self._train_config.aspect_ratios
    @property
    def classifier_source_names(self):
        return self._train_config.classifier_source_names
    @property
    def addon_source_names(self):
        return self._train_config.addon_source_names
    @property
    def codec_means(self):
        return self._train_config.codec_means
    @property
    def codec_stds(self):
        return self._train_config.codec_stds
    @property
    def rgb_means(self):
        return self._train_config.rgb_means
    @property
    def rgb_stds(self):
        return self._train_config.rgb_stds
    @property
    def val_conf_threshold(self):
        return self._val_config.val_conf_threshold
    @property
    def vis_conf_threshold(self):
        return self._val_config.vis_conf_threshold
    @property
    def iou_threshold(self):
        return self._val_config.iou_threshold
    @property
    def topk(self):
        return self._val_config.topk

    # device management
    def to(self, *args, **kwargs):
        self.defaultBox.dboxes = self.defaultBox.dboxes.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        self.defaultBox.dboxes = self.defaultBox.dboxes.cuda(device)

        return super().cuda(device)


    def build(self, **kwargs):

        ### feature layers ###
        self.build_feature(**kwargs)

        ### addon layers ###
        self.build_addon(**kwargs)

        ### classifier layers ###
        self.build_classifier(**kwargs)

        ### default box ###
        self.defaultBox = self.defaultBox.build(self.feature_layers, self._train_config.classifier_source_names,
                                                self.localization_layers)

        self.init_weights()

        return self

    def forward(self, x):
        """
        :param x: Tensor, input Tensor whose shape is (batch, c, h, w)
        :return:
            predicts: localization and confidence Tensor, shape is (batch, total_dbox_num, 4+class_nums)
        """
        if not self.isBuilt:
            raise NotImplementedError(
                "Not initialized, implement \'build_feature\', \'build_classifier\', \'build_addon\'")
        if not self.training:
            raise NotImplementedError("call \'train()\' first")

        # feature
        sources = []
        addon_i = 1
        for name, layer in self.feature_layers.items():
            x = layer(x)

            source = x
            if name in self.addon_source_names:
                if name not in self._train_config.classifier_source_names:
                    logging.warning("No meaning addon: {}".format(name))
                source = self.addon_layers['addon_{}'.format(addon_i)](source)
                addon_i += 1

            # get features by feature map convolution
            if name in self._train_config.classifier_source_names:
                sources += [source]

        # classifier
        locs, confs = [], []
        for source, loc_name, conf_name in zip(sources, self.localization_layers, self.confidence_layers):
            locs += [self.localization_layers[loc_name](source)]
            confs += [self.confidence_layers[conf_name](source)]

        predicts = self.predictor(locs, confs)
        return predicts

    def learn(self, x, targets):
        """
        :param x: Tensor, input Tensor whose shape is (batch, c, h, w)
        :param targets: Tensor, list of Tensor, whose shape = (object num, 4 + class num) including background
        :return:
            pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
            predicts: localization and confidence Tensor, shape is (batch, total_dbox_num, 4+class_nums)
            targets: Tensor, matched targets. shape = (batch num, dbox num, 4 + class num)
        """
        if not self.isBuilt:
            raise NotImplementedError(
                "Not initialized, implement \'build_feature\', \'build_classifier\', \'build_addon\'")

        batch_num = x.shape[0]

        pos_indicator, targets = self.encoder(targets, self.dboxes, batch_num)
        predicts = self(x)

        return pos_indicator, predicts, targets

    def infer(self, image, conf_threshold=None, toNorm=False, visualize=False):
        """
        :param image: ndarray or Tensor of list or tuple, or ndarray, or Tensor. Note that each type will be handled as;
            ndarray of list or tuple, ndarray: (?, h, w, c). channel order will be handled as RGB
            Tensor of list or tuple, Tensor: (?, c, h, w). channel order will be handled as RGB
        :param conf_threshold: float or None, if it's None, default value will be passed
        :param toNorm: bool, whether to normalize passed image
        :param visualize: bool,
        :return:
        """
        if not self.isBuilt:
            raise NotImplementedError("Not initialized, implement \'build_feature\', \'build_classifier\', \'build_addon\'")
        if self.training:
            raise NotImplementedError("call \'eval()\' first")

        # img: Tensor, shape = (b, c, h, w)
        img = check_image(image)

        # normed_img, orig_img: Tensor, shape = (b, c, h, w)
        normed_img, orig_img = get_normed_and_origin_img(img, self.rgb_means, self.rgb_stds, toNorm)

        if list(img.shape[1:]) != [self.input_channel, self.input_height, self.input_width]:
            raise ValueError('image shape was not same as input shape: {}, but got {}'.format([self.input_channel, self.input_height, self.input_width], list(img.shape[1:])))


        if conf_threshold is None:
            conf_threshold = self.vis_conf_threshold if visualize else self.val_conf_threshold

        # predict
        predicts = self(normed_img)
        infers = self.inferenceBox(predicts, self.dboxes, conf_threshold)

        img_num = normed_img.shape[0]
        if visualize:
            return infers, [toVisualizeRectangleimg(orig_img[i], infers[i][:, 1:], verbose=False) for i in range(img_num)]
        else:
            return infers


class SSDvggBase(SSDBase):

    def __init__(self, train_config, val_config, defaultBox, **build_kwargs):
        """
        :param train_config: SSDTrainConfig
        :param val_config: SSDValonfig
        :param defaultBox: instance inheriting DefaultBoxBase
        """
        self._vgg_index = -1

        super().__init__(train_config, val_config, defaultBox, **build_kwargs)

    def build_feature(self, **kwargs):
        """
        :param vgg_layers: nn.ModuleDict
        :param extra_layers: nn.ModuleDict
        :return:
        """
        vgg_layers = kwargs.get('vgg_layers')
        extra_layers = kwargs.get('extra_layers')

        feature_layers = []
        vgg_layers = _check_ins('vgg_layers', vgg_layers, nn.ModuleDict)
        for name, module in vgg_layers.items():
            feature_layers += [(name, module)]
        self._vgg_index = len(feature_layers)

        extra_layers = _check_ins('extra_layers', extra_layers, nn.ModuleDict)
        for name, module in extra_layers.items():
            feature_layers += [(name, module)]

        self.feature_layers = nn.ModuleDict(OrderedDict(feature_layers))

    def build_addon(self, **kwargs):
        addon_layers = []
        for i, name in enumerate(self.addon_source_names):
            addon_layers += [
                ('addon_{}'.format(i + 1), L2Normalization(self.feature_layers[name].out_channels, gamma=20))
            ]
        self.addon_layers = nn.ModuleDict(addon_layers)

    def build_classifier(self, **kwargs):
        # loc and conf layers
        in_channels = tuple(self.feature_layers[name].out_channels for name in self.classifier_source_names)

        _dbox_num_per_fpixel = [len(aspect_ratio) * 2 for aspect_ratio in self.aspect_ratios]
        # loc
        out_channels = tuple(dbox_num * 4 for dbox_num in _dbox_num_per_fpixel)
        localization_layers = [
            *Conv2d.block('_loc', len(_dbox_num_per_fpixel), in_channels, out_channels, kernel_size=(3, 3), padding=1,
                          batch_norm=False)
        ]
        self.localization_layers = nn.ModuleDict(OrderedDict(localization_layers))

        # conf
        out_channels = tuple(dbox_num * self.class_nums for dbox_num in _dbox_num_per_fpixel)
        confidence_layers = [
            *Conv2d.block('_conf', len(_dbox_num_per_fpixel), in_channels, out_channels, kernel_size=(3, 3),
                          padding=1, batch_norm=False)
        ]
        self.confidence_layers = nn.ModuleDict(OrderedDict(confidence_layers))


    def init_weights(self):
        _initialize_xavier_uniform(self.feature_layers)
        _initialize_xavier_uniform(self.localization_layers)
        _initialize_xavier_uniform(self.confidence_layers)

# weights management
def load_vgg_weights(model, name):
    assert isinstance(model, SSDvggBase), "must be inherited SSDvggBase"

    model_dir = weights_path(__file__, _root_num=2, dirname='weights')

    model_url = get_model_url(name)
    pretrained_state_dict = load_state_dict_from_url(model_url, model_dir=model_dir)
    #pretrained_state_dict = torch.load('/home/kado/Desktop/program/machile-learning/ssd.pytorch/weights/vgg16_reducedfc.pth')
    model_state_dict = model.state_dict()

    # rename
    renamed = []
    pre_keys, mod_keys = list(pretrained_state_dict.keys()), list(model_state_dict.keys())
    for (pre_key, mod_key) in zip(pre_keys[:model._vgg_index], mod_keys[:model._vgg_index]):
        renamed += [(mod_key, pretrained_state_dict[pre_key])]

    # set vgg layer's parameters
    model_state_dict.update(OrderedDict(renamed))
    model.load_state_dict(model_state_dict)

    logging.info("model loaded")


def _initialize_xavier_uniform(layers):
    for module in layers.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, ConvRelu):
            nn.init.xavier_uniform_(module.conv.weight)
            if module.conv.bias is not None:
                nn.init.constant_(module.conv.bias, 0)

def check_image(image):
    """
    :param image: ndarray or Tensor of list or tuple, or ndarray, or Tensor. Note that each type will be handled as;
            ndarray of list or tuple, ndarray: (?, h, w, c). channel order will be handled as RGB
            Tensor of list or tuple, Tensor: (?, c, h, w). channel order will be handled as RGB
    :return:
        img: Tensor, shape = (b, c, h, w)
    """
    if isinstance(image, (list, tuple)):
        img = []
        for im in image:
            if isinstance(im, np.ndarray):
                im = torch.tensor(im, requires_grad=False)
                img += im.permute((2, 0, 1))
            elif isinstance(im, torch.Tensor):
                img += im
            else:
                raise ValueError('Invalid image type. list or tuple\'s element must be ndarray or Tensor, but got \'{}\''.format(im.__name__))

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
        raise ValueError('Invalid image type. list or tuple\'s element must be'
                         '\'list\', \'tuple\', \'ndarray\' or \'Tensor\', but got \'{}\''.format(image.__name__))

    if img.ndim == 3:
        img = img.unsqueeze(0)  # shape = (1, ?, ?, ?)

    return img

def get_normed_and_origin_img(img, rgb_means, rgb_stds, toNorm):
    """
    :param img: Tensor, shape = (b, c, h, w)
    :param rgb_means: tuple or float
    :param rgb_stds: tuple or float
    :param toNorm: Bool
    :return:
        normed_img: Tensor, shape = (b, c, h, w)
        orig_img: Tensor, shape = (b, c, h, w)
    """
    rgb_means = _check_norm('rgb_means', rgb_means)
    rgb_stds = _check_norm('rgb_stds', rgb_stds)

    # shape = (1, 3, 1, 1)
    rgb_means = rgb_means.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    rgb_stds = rgb_stds.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    if toNorm:
        normed_img = (img - rgb_means) / rgb_stds
        orig_img = img
    else:
        normed_img = img
        orig_img = img * rgb_stds + rgb_means

    return normed_img, orig_img