import numpy as np
import torch
import abc
from torch import nn

class DefaultBoxBase(nn.Module):
    def __init__(self, img_shape=(300, 300, 3), scale_range=(0.1, 0.9),
                 aspect_ratios=((1, 2), (1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2), (1, 2)), clip=True):
        """
        :param img_shape: tuple, must be 3d
        :param scale_range: tuple of scale range, first element means minimum scale and last is maximum relu_one
        :param aspect_ratios: tuple of aspect ratio, note that all of elements must be greater than 0
        :param clip: bool, whether to force to be 0 to 1
        """
        super().__init__()

        assert len(img_shape) == 3, "input image dimension must be 3"
        assert img_shape[0] == img_shape[1], "input image's height and width must be same"
        self._img_shape = img_shape

        self._scale_range = scale_range

        # check aspect ratio is prper
        for aspect_ratio in aspect_ratios:
            assert np.where(np.array(aspect_ratio) <= 0)[0].size <= 0, "aspect must be greater than 0"
        self.aspect_ratios = aspect_ratios

        self.clip = clip

        self.dbox_num_per_fmap = []
        self.fmap_sizes = []

        self.dboxes = None

    @property
    def img_height(self):
        return self._img_shape[0]
    @property
    def img_width(self):
        return self._img_shape[1]
    @property
    def img_channels(self):
        return self._img_shape[2]

    @property
    def total_dboxes_nums(self):
        if self.dboxes is not None:
            return self.dboxes.shape[0]
        else:
            raise NotImplementedError('must call build')

    @property
    def fmap_num(self):
        return len(self.aspect_ratios)

    @property
    def scale_min(self):
        return self._scale_range[0]
    @property
    def scale_max(self):
        return self._scale_range[1]

    def get_scale(self, k, m):
        return self.scale_min + (self.scale_max - self.scale_min) * (k - 1) / (m - 1)

    @property
    def dbox_num_per_fpixel(self):
        ret = []
        for aspect_ratio in self.aspect_ratios:
            ret += [len(aspect_ratio)*2]
        return ret

    def build(self, feature_layers, classifier_source_names, localization_layers):

        assert len(classifier_source_names) == self.fmap_num and len(localization_layers) == self.fmap_num, 'must be same length'

        # this x is pseudo Tensor to get feature's map size
        x = torch.tensor((), dtype=torch.float, requires_grad=False).new_zeros((1, self.img_channels, self.img_height, self.img_width))

        i = 0
        for name, layer in feature_layers.items():
            x = layer(x)
            # get features by feature map convolution
            if name in classifier_source_names:
                feature = localization_layers['conv_loc_{0}'.format(i + 1)](x)
                _, _, h, w = feature.shape
                self.fmap_sizes += [[h, w]]
                dbox = len(self.aspect_ratios[i]) * 2
                self.dbox_num_per_fmap += [h * w * dbox]
                # print(features[-1].shape)
                i += 1

        self.dboxes = self.forward()
        self.dboxes.requires_grad_(False)

        if self.dboxes is None:
            raise NotImplementedError('must inherit forward')

        return self

    @abc.abstractmethod
    def forward(self):
        NotImplementedError()


class DBoxSSDOriginal(DefaultBoxBase):
    def __init__(self, img_shape, scale_conv4_3=0.1, scale_range=(0.2, 0.9), **kwargs):
        super().__init__(img_shape=img_shape, scale_range=scale_range, **kwargs)
        self.scale_conv4_3 = scale_conv4_3

    def forward(self):
        dboxes = []

        #fsize = []
        #sk = []
        #sk_ = []

        # conv4_3 has different scale
        fmap_h, fmap_w = self.fmap_sizes[0]
        scale_k = self.scale_conv4_3
        scale_k_plus = self.scale_min
        ars = self.aspect_ratios[0]
        dboxes += self._make(fmap_w, fmap_h, scale_k, scale_k_plus, ars)
        #fsize += [fmap_h]
        #sk += [scale_k]
        #sk_ += [scale_k_plus]
        for k in range(1, self.fmap_num):
            fmap_h, fmap_w = self.fmap_sizes[k]
            scale_k = self.get_scale(k, m=self.fmap_num-1)
            scale_k_plus = self.get_scale(k + 1, m=self.fmap_num-1)
            ars = self.aspect_ratios[k]
            dboxes += self._make(fmap_w, fmap_h, scale_k, scale_k_plus, ars)
        """
            fsize += [fmap_h]
            sk += [scale_k]
            sk_ += [scale_k_plus]
        print(fsize, sk, sk_)
        """
        #print(dboxes)

        dboxes = np.concatenate(dboxes, axis=0)
        dboxes = torch.from_numpy(dboxes).float()

        # ret_features = torch.cat(ret_features, dim=1)
        if self.clip:
            dboxes = dboxes.clamp(min=0, max=1)

        return dboxes  # , ret_features

    def _make(self, fmap_w, fmap_h, scale_k, scale_k_plus, ars):
        # get cx and cy
        # (cx, cy) = ((i+0.5)/f_k, (j+0.5)/f_k)

        # / f_k
        step_i, step_j = (np.arange(fmap_w) + 0.5) / fmap_w, (np.arange(fmap_h) + 0.5) / fmap_h
        # ((i+0.5)/f_k, (j+0.5)/f_k) for all i,j
        cx, cy = np.meshgrid(step_i, step_j)
        # cx, cy's shape (fmap_w, fmap_h) to (fmap_w*fmap_h, 1)
        aspect_ratio_num = len(ars)*2

        cx, cy = cx.reshape(-1, 1).repeat(aspect_ratio_num, axis=0), cy.reshape(-1, 1).repeat(aspect_ratio_num, axis=0)
        width, height = np.zeros_like(cx), np.zeros_like(cy)

        for i, ar in enumerate(ars):
            # normal aspect
            aspect = ar
            scale = scale_k

            box_w, box_h = scale * np.sqrt(aspect), scale / np.sqrt(aspect)
            width[i*2::aspect_ratio_num], height[i*2::aspect_ratio_num] = box_w, box_h

            # reciprocal aspect
            aspect = 1.0 / aspect
            if aspect == 1:  # if aspect is 1, scale = sqrt(s_k * s_k+1)
                scale = np.sqrt(scale_k * scale_k_plus)
            box_w, box_h = scale * np.sqrt(aspect), scale / np.sqrt(aspect)
            width[i*2+1::aspect_ratio_num], height[i*2+1::aspect_ratio_num] = box_w, box_h

        return [np.concatenate((cx, cy, width, height), axis=1)]


# deprecated
class _DefaultBox(DefaultBoxBase):
    def __init__(self, **kwargs):
        """
        :param img_shape: tuple, must be 3d
        :param scale_range: tuple of scale range, first element means minimum scale and last is maximum relu_one
        :param aspect_ratios: tuple of aspect ratio, note that all of elements must be greater than 0
        :param clip: bool, whether to force to be 0 to 1
        """
        super().__init__(**kwargs)


    def forward(self):

        from itertools import product
        from math import sqrt
        mean = []
        steps = [8, 16, 32, 64, 100, 300]
        min_sizes = [30, 60, 111, 162, 213, 264]
        max_sizes = [60, 111, 162, 213, 264, 315]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        """
        fsize = []
        sk = []
        sk_ = []
        """
        for k, sizes in enumerate(self.fmap_sizes):
            fmap_h, fmap_w = sizes
            for i, j in product(range(fmap_h), repeat=2):
                f_k = self.img_width / steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = min_sizes[k] / self.img_width
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (max_sizes[k] / self.img_width))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        #print(mean)
        """
            fsize += [f_k]
            sk += [s_k]
            sk_ += [s_k_prime]
        print(fsize, sk, sk_)
        [37.5, 18.75, 9.375, 4.6875, 3.0, 1.0]
        [0.1, 0.2, 0.37, 0.54, 0.71, 0.88] 
        [0.14142135623730953, 0.2720294101747089, 0.4469899327725402, 0.6191930232165088, 0.7904429138147802, 0.9612491872558333]
        """
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        k = output.cpu().numpy()
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

