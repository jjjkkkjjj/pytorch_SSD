from torchvision.models.utils import load_state_dict_from_url
import abc
import logging
import torch

from .._utils import _check_ins
from ..core.boxes.dbox import *
from ..core.layers import *
from ..core.predict import *
from ..core.inference import *
from ..core.boxes.codec import *
from .._utils import weights_path, _check_norm
from ..core.inference import InferenceBox, toVisualizeRGBImg
from ..models.vgg_base import get_model_url

class ObjectDetectionModelBase(nn.Module):

    def __init__(self, class_labels, input_shape, batch_norm):
        """
        :param class_labels: int, class number
        :param input_shape: tuple, 3d and (height, width, channel)
        :param batch_norm: bool, whether to add batch normalization layers
        """
        super().__init__()

        self._class_labels = class_labels
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
    def class_labels(self):
        return self._class_labels
    @property
    def class_nums(self):
        return len(self._class_labels)
    @property
    def class_nums_with_background(self):
        return self.class_nums + 1

    @property
    def batch_norm(self):
        return self._batch_norm


    @abc.abstractmethod
    def learn(self, x, targets):
        pass
    @abc.abstractmethod
    def infer(self, image, visualize=False, **kwargs):
        """
        :param image:
        :param visualize:
        :param kwargs:
        :return:
            infers: Tensor, shape = (box_num, 5=(conf, cx, cy, w, h))
            Note: if there is no boxes, all of infers' elements are -1, which means (-1, -1, -1, -1, -1)
            visualized_images: list of ndarray, if visualize=True
        """
        pass

    @property
    def device(self):
        devices = ({param.device for param in self.parameters()} |
                   {buf.device for buf in self.buffers()})
        if len(devices) != 1:
            raise RuntimeError('Cannot determine device: {} different devices found'
                               .format(len(devices)))
        return next(iter(devices))

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
        self._class_labels = _check_ins('class_labels', kwargs.get('class_labels'), (tuple, list))

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
    def class_labels(self):
        return self._class_labels
    @property
    def class_nums(self):
        return len(self._class_labels)

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

    def __init__(self, train_config, val_config, defaultBox,
                 codec=None, predictor=None, inferenceBox=None, **build_kwargs):
        """
        :param train_config: SSDTrainConfig
        :param val_config: SSDValConfig
        :param defaultBox: instance inheriting DefaultBoxBase
        :param codec: Codec, if it's None, use default Codec
        :param predictor: Predictor, if it's None, use default Predictor
        :param inferenceBox: InferenceBox, if it's None, use default InferenceBox
        """
        self._train_config = _check_ins('train_config', train_config, SSDTrainConfig)
        self._val_config = _check_ins('val_config', val_config, SSDValConfig)
        super().__init__(train_config.class_labels, train_config.input_shape, train_config.batch_norm)

        self.codec = _check_ins('codec', codec, CodecBase, allow_none=True,
                                default=Codec(norm_means=self.codec_means, norm_stds=self.codec_stds))
        self.defaultBox = _check_ins('defaultBox', defaultBox, DefaultBoxBase)

        self.predictor = _check_ins('predictor', predictor, PredictorBase, allow_none=True,
                                    default=Predictor(self.class_nums_with_background))

        self.inferenceBox = _check_ins('inferenceBox', inferenceBox, InferenceBoxBase, allow_none=True,
                                       default=InferenceBox(class_nums_with_background=self.class_nums_with_background,
                                                            filter_func=non_maximum_suppression, val_config=val_config))

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
    def class_labels(self):
        return self._train_config.class_labels
    @property
    def class_nums(self):
        return self._train_config.class_nums

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
        self.defaultBox.dboxes = self.dboxes.to(*args, **kwargs)

        self.codec = self.codec.to(*args, **kwargs)

        self.inferenceBox.device = self.dboxes.device

        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        self.defaultBox.dboxes = self.dboxes.cuda(device)

        self.codec = self.codec.cuda(device)

        self.inferenceBox.device = self.dboxes.device

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
            predicts: localization and confidence Tensor, shape is (batch, total_dbox_num, 4+class_labels)
        """
        if not self.isBuilt:
            raise NotImplementedError(
                "Not initialized, implement \'build_feature\', \'build_classifier\', \'build_addon\'")

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
            predicts: localization and confidence Tensor, shape is (batch, total_dbox_num, 4+class_labels)
            targets: Tensor, matched targets. shape = (batch num, dbox num, 4 + class num)
        """
        if not self.isBuilt:
            raise NotImplementedError(
                "Not initialized, implement \'build_feature\', \'build_classifier\', \'build_addon\'")
        if not self.training:
            raise NotImplementedError("call \'train()\' first")

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
        img = check_image(image, self.device)

        # normed_img, orig_img: Tensor, shape = (b, c, h, w)
        normed_img, orig_img = get_normed_and_origin_img(img, self.rgb_means, self.rgb_stds, toNorm, self.device)

        if list(img.shape[1:]) != [self.input_channel, self.input_height, self.input_width]:
            raise ValueError('image shape was not same as input shape: {}, but got {}'.format([self.input_channel, self.input_height, self.input_width], list(img.shape[1:])))


        if conf_threshold is None:
            conf_threshold = self.vis_conf_threshold if visualize else self.val_conf_threshold

        with torch.no_grad():

            # predict
            predicts = self(normed_img)

            predicts = self.decoder(predicts, self.dboxes)

            # list of tensor, shape = (box num, 6=(class index, confidence, cx, cy, w, h))
            infers = self.inferenceBox(predicts, conf_threshold)

            img_num = normed_img.shape[0]
            if visualize:
                return infers, [toVisualizeRGBImg(orig_img[i], locs=infers[i][:, 2:], inf_labels=infers[i][:, 0],
                                                  inf_confs=infers[i][:, 1], classe_labels=self.class_labels, verbose=False) for i in range(img_num)]
            else:
                return infers

    def load_for_finetune(self, path):
        """
        load weights from input to extra features weights for fine tuning
        :param path: str
        :return: self
        """
        pretrained_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_state_dict = self.state_dict()

        # rename
        pre_keys, mod_keys = list(pretrained_state_dict.keys()), list(model_state_dict.keys())
        renamed = [(pre_key, pretrained_state_dict[pre_key]) for pre_key in pre_keys if not ('conf' in pre_key or 'loc' in pre_key)]

        # set vgg layer's parameters
        model_state_dict.update(OrderedDict(renamed))
        self.load_state_dict(model_state_dict, strict=False)

        logging.info("model loaded")

class SSDvggBase(SSDBase):

    def __init__(self, train_config, val_config, defaultBox, codec=None, predictor=None, inferenceBox=None, **build_kwargs):
        """
        :param train_config: SSDTrainConfig
        :param val_config: SSDValonfig
        :param defaultBox: instance inheriting DefaultBoxBase
        :param codec: Codec, if it's None, use default Codec
        :param predictor: Predictor, if it's None, use default Predictor
        :param inferenceBox: InferenceBox, if it's None, use default InferenceBox
        """
        self._vgg_index = -1

        super().__init__(train_config, val_config, defaultBox,
                         codec=codec, predictor=predictor, inferenceBox=inferenceBox, **build_kwargs)

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
        out_channels = tuple(dbox_num * self.class_nums_with_background for dbox_num in _dbox_num_per_fpixel)
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
    # to avoid Error regarding num_batches_tracked
    pre_ind = 0
    for mod_ind in range(model._vgg_index):
        pre_key, mod_key = pre_keys[pre_ind], mod_keys[mod_ind]
        if 'num_batches_tracked' in mod_key:
            continue
        renamed += [(mod_key, pretrained_state_dict[pre_key])]
        pre_ind += 1
    """
    for (pre_key, mod_key) in zip(pre_keys[:model._vgg_index], mod_keys[:model._vgg_index]):
        renamed += [(mod_key, pretrained_state_dict[pre_key])]
    """


    # set vgg layer's parameters
    model_state_dict.update(OrderedDict(renamed))
    model.load_state_dict(model_state_dict, strict=False)

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

def check_image(image, device):
    """
    :param image: ndarray or Tensor of list or tuple, or ndarray, or Tensor. Note that each type will be handled as;
            ndarray of list or tuple, ndarray: (?, h, w, c). channel order will be handled as RGB
            Tensor of list or tuple, Tensor: (?, c, h, w). channel order will be handled as RGB
    :param device: torch.device
    :return:
        img: Tensor, shape = (b, c, h, w)
    """
    if isinstance(image, (list, tuple)):
        img = []
        for im in image:
            if isinstance(im, np.ndarray):
                im = torch.tensor(im, requires_grad=False)
                img += [im.permute((2, 0, 1))]
            elif isinstance(im, torch.Tensor):
                img += [im]
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

    return img.to(device)

def get_normed_and_origin_img(img, rgb_means, rgb_stds, toNorm, device):
    """
    :param img: Tensor, shape = (b, c, h, w)
    :param rgb_means: tuple or float
    :param rgb_stds: tuple or float
    :param toNorm: Bool
    :param device: torch.device
    :return:
        normed_img: Tensor, shape = (b, c, h, w)
        orig_img: Tensor, shape = (b, c, h, w). Order is rgb
    """
    rgb_means = _check_norm('rgb_means', rgb_means)
    rgb_stds = _check_norm('rgb_stds', rgb_stds)

    img = img.to(device)

    # shape = (1, 3, 1, 1)
    rgb_means = rgb_means.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
    rgb_stds = rgb_stds.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
    if toNorm:
        normed_img = (img / 255. - rgb_means) / rgb_stds
        orig_img = img / 255. # divide 255. for tensor2cvrgbimg
    else:
        normed_img = img
        orig_img = img * rgb_stds + rgb_means

    return normed_img, orig_img