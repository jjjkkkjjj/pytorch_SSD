import numpy as np
import torch
from torch import nn


def matching_strategy(gts, dboxes, **kwargs):
    """
    :param gts: Tensor, shape is (batch*object num(batch), 1+4+class_nums)
    :param dboxes: shape is (default boxes num, 4)
    IMPORTANT: Note that means (cx, cy, w, h)
    :param kwargs:
        threshold: (Optional) float, threshold for returned indicator
        batch_num: (Required) int, batch size
    :return:
        pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
        matched_gts: Tensor, shape = (batch, default box num, 4+class_num)
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
    pos_indicator, matched_gts = torch.empty((batch_num, dboxes_num), device=device, dtype=torch.bool), torch.empty((batch_num, dboxes_num, 4 + class_num), device=device)

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

        # assign gts
        matched_gts[b, :, :4], matched_gts[b, :, 4:] = gt_loc_per_img[object_indices], gt_conf_per_img[object_indices]
        pos_indicator[b] = pos_ind

        # set background flag
        neg_ind = torch.logical_not(pos_ind)
        matched_gts[b, neg_ind, 4:] = 0
        matched_gts[b, neg_ind, -1] = 1

        index += box_num



    return pos_indicator, matched_gts

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

def iou_numpy(a, b):
    """
    :param a: Box ndarray, shape is (nums, 4)
    :param b: Box ndarray, shape is (nums, 4)
    IMPORTANT: Note that 4 means (xmin, ymin, xmax, ymax)
    :return:
        iou: ndarray, shape is (a_num, b_num)
             formula is
             iou = intersection / union = intersection / (A + B - intersection)
    """

    # get intersection's xmin, ymin, xmax, ymax
    # xmin = max(a_xmin, b_xmin)
    # ymin = max(a_ymin, b_ymin)
    # xmax = min(a_xmax, b_xmax)
    # ymax = min(a_ymax, b_ymax)

    # convert for broadcast
    # a's shape = (a_num, 1, 4), b's shape = (1, b_num, 4)
    a, b = np.expand_dims(a, 1), np.expand_dims(b, 0)
    intersection = np.concatenate((np.expand_dims(np.maximum(a[:, :, 0], b[:, :, 0]), 2),
                                   np.expand_dims(np.maximum(a[:, :, 1], b[:, :, 1]), 2),
                                   np.expand_dims(np.minimum(a[:, :, 2], b[:, :, 2]), 2),
                                   np.expand_dims(np.minimum(a[:, :, 3], b[:, :, 3]), 2)), axis=2)
    # get intersection's area
    # (w, h) = (xmax - xmin, ymax - ymin)
    intersection_w, intersection_h = intersection[:, :, 2] - intersection[:, :, 0], intersection[:, :, 3] - intersection[:, :, 1]
    # if intersection's width or height is negative, those will be converted to zero
    intersection_w, intersection_h = np.clip(intersection_w, a_min=0, a_max=None), np.clip(intersection_h, a_min=0, a_max=None)

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

def centroids2minmax_numpy(a):
    """
    :param a: Box Tensor, shape is (nums, 4=(cx, cy, w, h))
    :return:
        a: Box Tensor, shape is (nums, 4=(xmin, ymin, xmax, ymax))
    """
    return np.concatenate((a[:, :2] - a[:, 2:]/2, a[:, :2] + a[:, 2:]/2), axis=1)

def minmax2centroids_numpy(a):
    """
    :param a: Box Tensor, shape is (nums, 4=(xmin, ymin, xmax, ymax))
    :return:
        a: Box Tensor, shape is (nums, 4=(cx, cy, w, h))
    """
    return np.concatenate(((a[:, 2:] + a[:, :2])/2, a[:, 2:] - a[:, :2]), axis=1)


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

