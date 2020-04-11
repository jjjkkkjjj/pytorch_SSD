from torch import nn
import torch
import numpy as np

from .layers import Flatten

class L2Normalization(nn.Module):
    def __init__(self, channels, gamma=20):
        super().__init__()
        self.gamma = gamma
        self.in_channels = channels
        self.out_channels = channels

    # Note that pytorch's dimension order is batch_size, channels, height, width
    def forward(self, x):
        # |x|_2
        # square element-wise, sum along channel and square element-wise
        norm_x = torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt()
        # normalize (x^)
        x = torch.div(x, norm_x)
        return self.gamma * x


class DefaultBox(nn.Module):
    def __init__(self, img_size=(300, 300), scale_range=(0.2, 0.9), aspect_ratios=(1, 2, 3)):
        super().__init__()
        self.flatten = Flatten()

        assert img_size[0] == img_size[1], "input image's height and width must be same"
        self._img_size = img_size
        self._scale_range = scale_range
        assert np.where(np.array(aspect_ratios) < 0)[0].size >= 0, "aspect must be greater than 0"
        self._aspect_ratios = aspect_ratios


    @property
    def scale_min(self):
        return self._scale_range[0]
    @property
    def scale_max(self):
        return self._scale_range[1]
    @property
    def img_height(self):
        return self._img_size[0]
    @property
    def img_width(self):
        return self._img_size[1]

    def forward(self, features, dbox_nums):
        """
        :param features: list of Tensor, Tensor's shape is (batch, c, h, w)
        :param dbox_nums: list of dbox numbers
        :return: dboxes(Tensor), features(Tensor).
                dboxes' shape is (position, cx, cy, w, h)
                features' shape is (position, class)
        """
        dboxes = []
        ret_features = []
        m = len(features)
        for k, feature, dbox_num in zip(range(1, m + 1), features, dbox_nums):
            _, _, fmap_h, fmap_w = feature.shape
            assert fmap_w == fmap_h, "feature map's height and width must be same"
            #f_k = np.sqrt(fmap_w * fmap_h)

            """
            get cx and cy
            (cx, cy) = ((i+0.5)/f_k, (j+0.5)/f_k)
            """
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
                box_w, box_h = np.broadcast_to([box_w], (total_dbox_num, 1)), np.broadcast_to([box_h], (total_dbox_num, 1))
                dboxes += [np.concatenate((cx, cy, box_w, box_h), axis=1)]

                # reciprocal aspect
                aspect = 1 / aspect
                if aspect == 1:  # if aspect is 1, scale = sqrt(s_k * s_k+1)
                    scale = np.sqrt(scale * self.get_scale(k + 1, m))
                box_w, box_h = scale * np.sqrt(aspect), scale / np.sqrt(aspect)
                box_w, box_h = np.broadcast_to([box_w], (total_dbox_num, 1)), np.broadcast_to([box_h], (total_dbox_num, 1))
                dboxes += [np.concatenate((cx, cy, box_w, box_h), axis=1)]

            ret_features += [self.flatten(feature)]

        dboxes = np.concatenate(dboxes, axis=0)
        dboxes = torch.from_numpy(dboxes).float()
        ret_features = torch.cat(ret_features, dim=1)
        return dboxes, ret_features


    """
            def get_centers(fmap_h, fmap_w):
            
            #get cx and cy
            #(cx, cy) = ((i+0.5)/f_k, (j+0.5)/f_k)
            
            # / f_k
            step_x, step_y = self.img_width / fmap_w, self.img_height / fmap_h
            # ((i+0.5)/f_k, (j+0.5)/f_k) for all i,j
            cx, cy = np.linspace(0.5 * step_x, self.img_width - 0.5 * step_x), np.linspace(0.5 * step_y, self.img_height - 0.5 * step_y)

            return np.meshgrid(cx, cy)

        _, _, fmap_h, fmap_w = feature.shape
        assert fmap_w == fmap_h, "feature map's height and width must be same"

        boxes_H, boxes_W = [], []
        for i in range(int(dbox_num / 2)):
            # normal aspect
            aspect = self._aspect_ratios[i]
            scale = self.get_scale(k, m)
            box_h, box_w = scale * np.sqrt(aspect), scale / np.sqrt(aspect)
            boxes_H.append(box_h)
            boxes_W.append(box_w)

            # reciprocal aspect
            aspect = 1 / aspect
            if aspect == 1:  # if aspect is 1, scale = sqrt(s_k * s_k+1)
                scale = np.sqrt(scale * self.get_scale(k + 1, m))
            box_h, box_w = scale * np.sqrt(aspect), scale / np.sqrt(aspect)
            boxes_H.append(box_h)
            boxes_W.append(box_w)
        boxes_CX, boxes_CY = get_centers(fmap_h, fmap_w)
        boxes_CX, boxes_CY = boxes_CX.reshape(-1, 1), boxes_CY.reshape(-1, 1)

        # defaultBoxes.shape = ()
        defaultBoxes = np.concatenate((boxes_CX, boxes_CY), axis=1)
        defaultBoxes = np.tile(defaultBoxes, (1, 2*dbox_num))
        defaultBoxes[:, ::4] -= boxes_W
        defaultBoxes[:, 1::4] -= boxes_H
        defaultBoxes[:, 2::4] += boxes_W
        defaultBoxes[:, 3::4] += boxes_H
        defaultBoxes[:, ::2] /= self.img_width
        defaultBoxes[:, 1::2] /= self.img_height
    """

    def get_scale(self, k, m):
        return self.scale_min + (self.scale_max - self.scale_min) * (k - 1) / (m - 1)