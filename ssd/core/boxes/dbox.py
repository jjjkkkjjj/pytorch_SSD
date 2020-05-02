import numpy as np
import torch
from torch import nn

class DefaultBox(nn.Module):
    def __init__(self, img_shape=(300, 300, 3), scale_range=(0.2, 0.9), aspect_ratios=(1, 2, 3), clip=True):
        """
        :param img_shape: tuple, must be 3d
        :param scale_range: tuple of scale range, first element means minimum scale and last is maximum relu_one
        :param aspect_ratios: tuple of aspect ratio, note that all of elements must be greater than 0
        :param clip: bool, whether to force to be 0 to 1
        """
        super().__init__()
        # self.flatten = Flatten()

        assert len(img_shape) == 3, "input image dimension must be 3"
        assert img_shape[0] == img_shape[1], "input image's height and width must be same"
        self._img_shape = img_shape
        self._scale_range = scale_range
        assert np.where(np.array(aspect_ratios) <= 0)[0].size <= 0, "aspect must be greater than 0"
        self._aspect_ratios = aspect_ratios
        self._clip = clip

        self.dboxes_nums = None
        self.dboxes = None
        self.fmap_sizes = []
        self.boxes_num = []

    @property
    def scale_min(self):
        return self._scale_range[0]

    @property
    def scale_max(self):
        return self._scale_range[1]

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

    def get_scale(self, k, m):
        return self.scale_min + (self.scale_max - self.scale_min) * (k - 1) / (m - 1)

    def build(self, feature_layers, classifier_source_names, localization_layers, dbox_nums):
        # this x is pseudo Tensor to get feature's map size
        x = torch.tensor((), dtype=torch.float, requires_grad=False).new_zeros((1, self.img_channels, self.img_height, self.img_width))

        features = []
        i = 1
        for name, layer in feature_layers.items():
            x = layer(x)
            # get features by feature map convolution
            if name in classifier_source_names:
                feature = localization_layers['conv_loc_{0}'.format(i)](x)
                features.append(feature)
                # print(features[-1].shape)
                i += 1

        self.dboxes_nums = dbox_nums
        self.dboxes = self.forward(features, dbox_nums)
        self.dboxes.requires_grad_(False)
        return self

    def forward(self, features, dbox_nums):
        """
        :param features: list of Tensor, Tensor's shape is (batch, c, h, w)
        :param dbox_nums: list of dbox numbers
        :return: dboxes(Tensor)
                dboxes' shape is (position, cx, cy, w, h)

                bellow is deprecated to LocConf
                features' shape is (position, class)
        """
        from itertools import product
        from math import sqrt
        mean = []
        steps = [8, 16, 32, 64, 100, 300]
        min_sizes = [30, 60, 111, 162, 213, 264]
        max_sizes = [60, 111, 162, 213, 264, 315]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        for k, f in enumerate(features):
            _, _, fmap_h, fmap_w = f.shape
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
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        k = output.cpu().numpy()
        if self._clip:
            output.clamp_(max=1, min=0)
        return output
        """
        dboxes = []
        # ret_features = []
        m = len(features)
        assert m == len(dbox_nums), "default boxes number and feature layers number must be same"

        for k, feature, dbox_num in zip(range(1, m + 1), features, dbox_nums):
            _, _, fmap_h, fmap_w = feature.shape
            assert fmap_w == fmap_h, "feature map's height and width must be same"
            self.fmap_sizes += [[fmap_h, fmap_w]]
            # f_k = np.sqrt(fmap_w * fmap_h)

            # get cx and cy
            # (cx, cy) = ((i+0.5)/f_k, (j+0.5)/f_k)

            # / f_k
            step_i, step_j = (np.arange(fmap_w) + 0.5) / fmap_w, (np.arange(fmap_h) + 0.5) / fmap_h
            # ((i+0.5)/f_k, (j+0.5)/f_k) for all i,j
            cx, cy = np.meshgrid(step_i, step_j)
            # cx, cy's shape (fmap_w, fmap_h) to (fmap_w*fmap_h, 1)
            cx, cy = cx.reshape(-1, 1), cy.reshape(-1, 1)
            total_dbox_num = cx.size
            for i in range(int(dbox_num / 2)):
                # normal aspect
                aspect = self._aspect_ratios[i]
                scale = self.get_scale(k, m)
                box_w, box_h = scale * np.sqrt(aspect), scale / np.sqrt(aspect)
                box_w, box_h = np.broadcast_to([box_w], (total_dbox_num, 1)), np.broadcast_to([box_h],
                                                                                              (total_dbox_num, 1))
                dboxes += [np.concatenate((cx, cy, box_w, box_h), axis=1)]

                # reciprocal aspect
                aspect = 1.0 / aspect
                if aspect == 1:  # if aspect is 1, scale = sqrt(s_k * s_k+1)
                    scale = np.sqrt(scale * self.get_scale(k + 1, m))
                box_w, box_h = scale * np.sqrt(aspect), scale / np.sqrt(aspect)
                box_w, box_h = np.broadcast_to([box_w], (total_dbox_num, 1)), np.broadcast_to([box_h],
                                                                                              (total_dbox_num, 1))
                dboxes += [np.concatenate((cx, cy, box_w, box_h), axis=1)]
            self.boxes_num += [total_dbox_num * dbox_num]
            # ret_features += [self.flatten(feature)]

        dboxes = np.concatenate(dboxes, axis=0)
        dboxes = torch.from_numpy(dboxes).float()

        # ret_features = torch.cat(ret_features, dim=1)
        if self._clip:
            dboxes = dboxes.clamp(min=0, max=1)

        return dboxes  # , ret_features
        """