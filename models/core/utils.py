from torch import nn
import torch

def matching_strategy(gts, dboxes, **kwargs):
    """
    :param gts: Tensor, shape is (batch, object num(batch), 1+4+class_nums)
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
    class_num = gts.shape[1] - 1 - 4 # minus 'box number per image' and 'localization=(cx, cy, w, h)'

    # convert centered coordinated to minmax coordinates
    dboxes_mm = center2minmax(dboxes)

    # create returned empty Tensor
    pos_indicator, gt_loc, gt_conf = torch.empty((batch_num, dboxes_num), device=device), torch.empty((batch_num, dboxes_num, 4), device=device), torch.empty((batch_num, dboxes_num, class_num), device=device)

    # matching for each batch
    index = 0
    for b in range(batch_num):
        box_num = int(gt_boxnum_per_image[index].item())
        gt_loc_per_img, gt_conf_per_img = gts[index:index + box_num, 1:5], gts[index:index + box_num, 5:]

        # overlaps' shape = (object num, default box num)
        overlaps = iou(center2minmax(gt_loc_per_img), dboxes_mm)

        # get maximum overlap value for each default box
        overlaps, object_indices = overlaps.max(dim=0)
        object_indices = object_indices.long() # for fancy indexing
        pos_ind = overlaps > threshold

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

def center2minmax(a):
    """
    :param a: Box Tensor, shape is (nums, 4=(cx, cy, w, h))
    :return:
        a: Box Tensor, shape is (nums, 4=(xmin, ymin, xmax, ymax))
    """
    return torch.cat((a[:, :2] - a[:, 2:]/2, a[:, :2] + a[:, 2:]/2), dim=1)

def minmax2center(a):
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

def align_predicts(predicts, gt_boxnum_per_image):
    """
    :param predicts: Tensor, shape is (batch, total_dbox_nums * 4+class_nums=(cx, cy, w, h, p_class,...)
    :param gt_boxnum_per_image: Tensor, shape is (batch*bbox_nums(batch))
            e.g.) gt_boxnum_per_image = (2,2,1,3,3,3,2,2,...)
            shortly, box number value is arranged for each box number
    :return:
        predicts: Tensor, shape = (total_dbox_nums * total_obox_nums, 4+class_nums)
    """
    ret_predicts = []
    batch_num = predicts.shape[0]
    index = 0
    for b in range(batch_num):
        box_num = int(gt_boxnum_per_image[index].item())
        ret_predicts += [tensor_tile(predicts[b], box_num, dim=0)]

        index += box_num

    return torch.cat(ret_predicts, dim=0)


def align_gts(gts, dbox_total_num):
    return torch.repeat_interleave(gts, dbox_total_num, dim=0)


def align_dboxes(dboxes, gt_total_objects_num):
    return tensor_tile(dboxes, gt_total_objects_num, dim=0)


def gt_loc_converter(gt_boxes, default_boxes):
    """
    :param gt_boxes: Tensor, shape = (batch, default boxes num, 4)
    :param default_boxes: Tensor, shape = (default boxes num, 4)
    :return:
        gt_boxes: Tensor, calculate ground truth value considering default boxes. The formula is below;
                  gt_cx = (gt_cx - dbox_cx)/dbox_w, gt_cy = (gt_cy - dbox_cy)/dbox_h,
                  gt_w = log(gt_w / dbox_w), gt_h = log(gt_h / dbox_h)
    """
    assert gt_boxes.shape[1:] == default_boxes.shape, "gt_boxes and default_boxes must be same shape"

    gt_cx = (gt_boxes[:, :, 0] - default_boxes[:, 0])/default_boxes[:, 2]
    gt_cy = (gt_boxes[:, :, 1] - default_boxes[:, 1])/default_boxes[:, 3]
    gt_w = torch.log(gt_boxes[:, :, 2] / default_boxes[:, 2])
    gt_h = torch.log(gt_boxes[:, :, 3] / default_boxes[:, 3])

    return torch.cat((gt_cx.unsqueeze(2),
                      gt_cy.unsqueeze(2),
                      gt_w.unsqueeze(2),
                      gt_h.unsqueeze(2)), dim=2)
