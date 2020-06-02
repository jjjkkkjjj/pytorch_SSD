from torchvision.models.utils import load_state_dict_from_url
import abc
import logging

from .._utils import check_instance, _check_ins
from ..core.boxes.dbox import *
from ..core.layers import *
from ..core.inference import InferenceBox
from ..core.boxes.codec import Codec
from ssd._utils import weights_path
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

        self._called_learn_infer = True

        return normed_img, orig_img


    def load_weights(self, path):
        """
        :param path: str
        :return:
        """
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))


    def init_weights(self):
        raise NotImplementedError()

class SSDConfig(object):
    def __init__(self, **kwargs):
        self.class_nums = kwargs.get('class_nums')

        input_shape = kwargs.get('input_shape')
        assert len(input_shape) == 3, "input dimension must be 3"
        assert input_shape[0] == input_shape[1], "input must be square size"
        self.input_shape = input_shape

        self.batch_norm = kwargs.get('batch_norm')

        self.aspect_ratios = kwargs.get('aspect_ratios')
        self.classifier_source_names = kwargs.get('classifier_source_names')
        self.addon_source_names = kwargs.get('addon_source_names')

        self.norm_means = kwargs.get('norm_means')
        self.norm_stds = kwargs.get('norm_stds')

    @property
    def input_height(self):
        return self.input_shape[0]
    @property
    def input_width(self):
        return self.input_shape[1]
    @property
    def input_channel(self):
        return self.input_shape[2]

class SSDBase(ObjectDetectionModelBase):
    defaultBox: DefaultBoxBase
    _config: SSDConfig

    feature_layers: nn.ModuleDict
    localization_layers: nn.ModuleDict
    confidence_layers: nn.ModuleDict
    addon_layers: nn.ModuleDict

    def __init__(self, config, defaultBox):
        """
        :param config: SSDvggConfig
        :param defaultBox: instance inheriting DefaultBoxBase
        """
        self._config = _check_ins('config', config, SSDConfig)
        super().__init__(config.class_nums, config.input_shape, config.batch_norm)

        self.codec = Codec(norm_means=self.norm_means, norm_stds=self.norm_stds)
        self.defaultBox = _check_ins('defaultBox', defaultBox, DefaultBoxBase)

        self.predictor = Predictor(self.class_nums)

    @property
    def isBuilt(self):
        return not hasattr(self, 'feature_layers') and\
               not hasattr(self, 'localization_layers') and\
               not hasattr(self, 'confidence_layers')

    ### build ###
    @abc.abstractmethod
    def build_feature(self, *args, **kwargs):
        pass
    @abc.abstractmethod
    def build_addon(self, *args, **kwargs):
        pass
    @abc.abstractmethod
    def build_classifier(self, *args, **kwargs):
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

    ### config ###
    @property
    def aspect_ratios(self):
        return self._config.aspect_ratios
    @property
    def classifier_source_names(self):
        return self._config.classifier_source_names
    @property
    def addon_source_names(self):
        return self._config.addon_source_names
    @property
    def norm_means(self):
        return self._config.norm_means
    @property
    def norm_stds(self):
        return self._config.norm_stds

    # device management
    def to(self, *args, **kwargs):
        self.defaultBox.dboxes = self.defaultBox.dboxes.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        self.defaultBox.dboxes = self.defaultBox.dboxes.cuda(device)

        return super().cuda(device)


    def build(self, vgg_layers, extra_layers):
        """
        :param vgg_layers: nn.ModuleDict
        :param extra_layers: nn.ModuleDict
        :return:
        """
        ### feature layers ###
        self.build_feature(vgg_layers, extra_layers)

        ### addon layers ###
        self.build_addon()

        ### classifier layers ###
        self.build_classifier()

        ### default box ###
        self.defaultBox = self.defaultBox.build(self.feature_layers, self._config.classifier_source_names,
                                                self.localization_layers)

        self.init_weights()

        return self

    def forward(self, x):
        """
        :param x: Tensor, input Tensor whose shape is (batch, c, h, w)
        :return:
            predicts: localization and confidence Tensor, shape is (batch, total_dbox_num, 4+class_nums)
        """
        # feature
        sources = []
        addon_i = 1
        for name, layer in self.feature_layers.items():
            x = layer(x)

            source = x
            if name in self.addon_source_names:
                if name not in self._config.classifier_source_names:
                    logging.warning("No meaning addon: {}".format(name))
                source = self.addon_layers['addon_{}'.format(addon_i)](source)
                addon_i += 1

            # get features by feature map convolution
            if name in self._config.classifier_source_names:
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
        batch_num = x.shape[0]

        pos_indicator, targets = self.encoder(targets, self.dboxes, batch_num)
        predicts = self(x)

        return pos_indicator, predicts, targets

    def init_weights(self):
        pass

class SSDvggBase(SSDBase):

    def __init__(self, config, defaultBox):
        """
        :param config: SSDvggConfig
        :param defaultBox: instance inheriting DefaultBoxBase
        """
        super().__init__(config, defaultBox)

        self._vgg_index = -1


    def build_feature(self, vgg_layers, extra_layers):
        """
        :param vgg_layers: nn.ModuleDict
        :param extra_layers: nn.ModuleDict
        :return:
        """
        feature_layers = []
        vgg_layers = _check_ins('vgg_layers', vgg_layers, nn.ModuleDict)
        for name, module in vgg_layers.items():
            feature_layers += [(name, module)]
        self._vgg_index = len(feature_layers)

        extra_layers = _check_ins('extra_layers', extra_layers, nn.ModuleDict)
        for name, module in extra_layers.items():
            feature_layers += [(name, module)]

        self.feature_layers = nn.ModuleDict(OrderedDict(feature_layers))

    def build_addon(self):
        addon_layers = []
        for i, name in enumerate(self.addon_source_names):
            addon_layers += [
                ('addon_{}'.format(i + 1), L2Normalization(self.feature_layers[name].out_channels, gamma=20))
            ]
        self.addon_layers = nn.ModuleDict(addon_layers)

    def build_classifier(self):
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