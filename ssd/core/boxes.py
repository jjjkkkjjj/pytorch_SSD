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


def matching_strategy(gts, dboxes, **kwargs):
    """
    :param gts: Tensor, shape is (batch*object num(batch), 1+4+class_nums)
    :param dboxes: shape is (default boxes num, 4)
    IMPORTANT: Note that means (cx, cy, w, h)
    :param kwargs:
        threshold: (Optional) float, threshold for returned indicator
        batch_num: (Required) int, batch size
    :return:
        indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
        gt_loc: Tensor, shape = (batch, default box num, 4)
        gt_conf: Tensor, shape = (batch, default box num, class_num)
    """
    threshold = kwargs.pop('threshold', 0.5)
    batch_num = kwargs.pop('batch_num')
    device = dboxes.device

    # get box number per image
    gt_boxnum_per_image = gts[:, 0]

    dboxes_num = dboxes.shape[0]
    # minus 'box number per image' and 'localization=(cx, cy, w, h)'
    class_num = gts.shape[1] - 1 - 4

    # convert centered coordinated to minmax coordinates
    dboxes_mm = centroids2minmax(dboxes)

    # create returned empty Tensor
    pos_indicator, gt_loc, gt_conf = torch.empty((batch_num, dboxes_num), device=device, dtype=torch.bool), torch.empty((batch_num, dboxes_num, 4), device=device), torch.empty((batch_num, dboxes_num, class_num), device=device)

    # matching for each batch
    index = 0
    for b in range(batch_num):
        box_num = int(gt_boxnum_per_image[index].item())
        gt_loc_per_img, gt_conf_per_img = gts[index:index + box_num, 1:5], gts[index:index + box_num, 5:]

        # overlaps' shape = (object num, default box num)
        overlaps = iou(centroids2minmax(gt_loc_per_img), dboxes_mm.clone())
        """
        best_overlap_per_object, best_dbox_ind_per_object = overlaps.max(dim=1)
        best_overlap_per_dbox, best_object_ind_per_dbox = overlaps.max(dim=0)
        for object_ind, dbox_ind in enumerate(best_dbox_ind_per_object):
            best_object_ind_per_dbox[dbox_ind] = object_ind
        best_overlap_per_dbox.index_fill_(0, best_dbox_ind_per_object, 999)

        pos_ind = best_overlap_per_dbox > threshold
        pos_indicator[b] = pos_ind
        gt_loc[b], gt_conf[b] = gt_loc_per_img[best_object_ind_per_dbox], gt_conf_per_img[best_object_ind_per_dbox]

        neg_ind = torch.logical_not(pos_ind)
        gt_conf[b, neg_ind] = 0
        gt_conf[b, neg_ind, -1] = 1
        """
        # get maximum overlap value for each default box
        # shape = (batch num, dboxes num)
        overlaps_per_dbox, object_indices = overlaps.max(dim=0)
        #object_indices = object_indices.long() # for fancy indexing

        # get maximum overlap values for each object
        # shape = (batch num, object num)
        overlaps_per_object, dbox_indices = overlaps.max(dim=1)
        for obj_ind, dbox_ind in enumerate(dbox_indices):
            object_indices[dbox_ind] = obj_ind
        overlaps_per_dbox.index_fill_(0, dbox_indices, threshold + 1)# ensure N!=0

        pos_ind = overlaps_per_dbox > threshold

        # assign boxes
        gt_loc[b], gt_conf[b] = gt_loc_per_img[object_indices], gt_conf_per_img[object_indices]
        pos_indicator[b] = pos_ind

        # set background flag
        neg_ind = torch.logical_not(pos_ind)
        gt_conf[b, neg_ind] = 0
        gt_conf[b, neg_ind, -1] = 1

        index += box_num



    return pos_indicator.bool(), gt_loc, gt_conf

def iou(a, b):
    """
    :param a: Box Tensor, shape is (nums, 4)
    :param b: Box Tensor, shape is (nums, 4)
    IMPORTANT: Note that 4 means (xmin, ymin, xmax, ymax)
    :return:
        iou: Tensor, shape is (a_num, b_num)
             formula is
             iou = intersection / union = intersection / (A + B - intersection)
    """

    # get intersection's xmin, ymin, xmax, ymax
    # xmin = max(a_xmin, b_xmin)
    # ymin = max(a_ymin, b_ymin)
    # xmax = min(a_xmax, b_xmax)
    # ymax = min(a_ymax, b_ymax)
    """
    >>> b
    tensor([2., 6.])
    >>> c
    tensor([1., 5.])
    >>> torch.cat((b.unsqueeze(1),c.unsqueeze(1)),1)
    tensor([[2., 1.],
            [6., 5.]])
    """
    # convert for broadcast
    # a's shape = (a_num, 1, 4), b's shape = (1, b_num, 4)
    a, b = a.unsqueeze(1), b.unsqueeze(0)
    intersection = torch.cat((torch.max(a[:, :, 0], b[:, :, 0]).unsqueeze(2),
                              torch.max(a[:, :, 1], b[:, :, 1]).unsqueeze(2),
                              torch.min(a[:, :, 2], b[:, :, 2]).unsqueeze(2),
                              torch.min(a[:, :, 3], b[:, :, 3]).unsqueeze(2)), dim=2)
    # get intersection's area
    # (w, h) = (xmax - xmin, ymax - ymin)
    intersection_w, intersection_h = intersection[:, :, 2] - intersection[:, :, 0], intersection[:, :, 3] - intersection[:, :, 1]
    # if intersection's width or height is negative, those will be converted to zero
    intersection_w, intersection_h = torch.clamp(intersection_w, min=0), torch.clamp(intersection_h, min=0)

    intersectionArea = intersection_w * intersection_h

    # get a and b's area
    # area = (xmax - xmin) * (ymax - ymin)
    A, B = (a[:, :, 2] - a[:, :, 0]) * (a[:, :, 3] - a[:, :, 1]), (b[:, :, 2] - b[:, :, 0]) * (b[:, :, 3] - b[:, :, 1])

    return intersectionArea / (A + B - intersectionArea)

def centroids2minmax(a):
    """
    :param a: Box Tensor, shape is (nums, 4=(cx, cy, w, h))
    :return:
        a: Box Tensor, shape is (nums, 4=(xmin, ymin, xmax, ymax))
    """
    return torch.cat((a[:, :2] - a[:, 2:]/2, a[:, :2] + a[:, 2:]/2), dim=1)

def minmax2centroids(a):
    """
    :param a: Box Tensor, shape is (nums, 4=(xmin, ymin, xmax, ymax))
    :return:
        a: Box Tensor, shape is (nums, 4=(cx, cy, w, h))
    """
    return torch.cat(((a[:, 2:] + a[:, :2])/2, a[:, 2:] - a[:, :2]), dim=1)


"""
repeat_interleave is similar to numpy.repeat
>>> a = torch.Tensor([[1,2,3,4],[5,6,7,8]])
>>> a
tensor([[1., 2., 3., 4.],
        [5., 6., 7., 8.]])
>>> torch.repeat_interleave(a, 3, dim=0)
tensor([[1., 2., 3., 4.],
        [1., 2., 3., 4.],
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [5., 6., 7., 8.],
        [5., 6., 7., 8.]])
>>> torch.cat(3*[a])
tensor([[1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [1., 2., 3., 4.],
        [5., 6., 7., 8.]])
"""
def tensor_tile(a, repeat, dim=0):
    return torch.cat([a]*repeat, dim=dim)

class Encoder(nn.Module):
    def __init__(self, norm_means=(0, 0, 0, 0), norm_stds=(0.1, 0.1, 0.2, 0.2)):
        super().__init__()
        # shape = (1, 1, 4=(cx, cy, w, h)) or (1, 1, 1)
        self.norm_means = torch.tensor(norm_means, requires_grad=False).unsqueeze(0).unsqueeze(0)
        self.norm_stds = torch.tensor(norm_stds, requires_grad=False).unsqueeze(0).unsqueeze(0)


    def forward(self, gt_boxes, default_boxes):
        """
        :param gt_boxes: Tensor, shape = (batch, default boxes num, 4)
        :param default_boxes: Tensor, shape = (default boxes num, 4)
        Note that 4 means (cx, cy, w, h)
        :return:
            encoded_boxes: Tensor, calculate ground truth value considering default boxes. The formula is below;
                           gt_cx = (gt_cx - dbox_cx)/dbox_w, gt_cy = (gt_cy - dbox_cy)/dbox_h,
                           gt_w = train(gt_w / dbox_w), gt_h = train(gt_h / dbox_h)
                           shape = (batch, default boxes num, 4)
        """
        assert gt_boxes.shape[1:] == default_boxes.shape, "gt_boxes and default_boxes must be same shape"

        gt_cx = (gt_boxes[:, :, 0] - default_boxes[:, 0]) / default_boxes[:, 2]
        gt_cy = (gt_boxes[:, :, 1] - default_boxes[:, 1]) / default_boxes[:, 3]
        gt_w = torch.log(gt_boxes[:, :, 2] / default_boxes[:, 2])
        gt_h = torch.log(gt_boxes[:, :, 3] / default_boxes[:, 3])

        encoded_boxes = torch.cat((gt_cx.unsqueeze(2),
                          gt_cy.unsqueeze(2),
                          gt_w.unsqueeze(2),
                          gt_h.unsqueeze(2)), dim=2)

        # normalization
        return (encoded_boxes - self.norm_means.to(gt_boxes.device)) / self.norm_stds.to(gt_boxes.device)


class Decoder(nn.Module):
    def __init__(self, norm_means=(0, 0, 0, 0), norm_stds=(0.1, 0.1, 0.2, 0.2)):
        super().__init__()
        # shape = (1, 1, 4=(cx, cy, w, h)) or (1, 1, 1)
        self.norm_means = torch.tensor(norm_means, requires_grad=False).unsqueeze(0).unsqueeze(0)
        self.norm_stds = torch.tensor(norm_stds, requires_grad=False).unsqueeze(0).unsqueeze(0)


    def forward(self, pred_boxes, default_boxes):
        """
        Opposite to above procession
        :param pred_boxes: Tensor, shape = (batch, default boxes num, 4)
        :param default_boxes: Tensor, shape = (default boxes num, 4)
        Note that 4 means (cx, cy, w, h)
        :return:
            inf_boxes: Tensor, calculate ground truth value considering default boxes. The formula is below;
                      inf_cx = pred_cx * dbox_w + dbox_cx, inf_cy = pred_cy * dbox_h + dbox_cy,
                      inf_w = exp(pred_w) * dbox_w, inf_h = exp(pred_h) * dbox_h
                      shape = (batch, default boxes num, 4)
        """
        assert pred_boxes.shape[1:] == default_boxes.shape, "pred_boxes and default_boxes must be same shape"

        pred_unnormalized = pred_boxes * self.norm_stds + self.norm_means

        inf_cx = pred_unnormalized[:, :, 0] * default_boxes[:, 2] + default_boxes[:, 0]
        inf_cy = pred_unnormalized[:, :, 1] * default_boxes[:, 3] + default_boxes[:, 1]
        inf_w = torch.exp(pred_unnormalized[:, :, 2]) * default_boxes[:, 2]
        inf_h = torch.exp(pred_unnormalized[:, :, 3]) * default_boxes[:, 3]

        return torch.cat((inf_cx.unsqueeze(2),
                          inf_cy.unsqueeze(2),
                          inf_w.unsqueeze(2),
                          inf_h.unsqueeze(2)), dim=2)