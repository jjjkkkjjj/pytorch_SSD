from torch import nn
import torch

def matching_strategy(gt_boxes, default_boxes, threshold=0.5):
    assert gt_boxes.shape == default_boxes.shape, "gt_boxes and default_boxes must be same shape"

    overlaps = iou(gt_boxes, default_boxes)

    return overlaps > threshold

def iou(a, b):
    """
    :param a: Box Tensor, shape is (nums, 4)
    :param b: Box Tensor, shape is (nums, 4)
    Note that the box's number of a and b must be same. 4 means (cx, cy, w, h)
    :return:
        iou: Tensor, shape is (a_num, b_num)
             formula is
             iou = intersection / union = intersection / (A + B - intersection)
    """
    assert a.shape == b.shape, "a and b must be same shape"
    # convert centered coordinates to minmax coordinates
    a, b = center2minmax(a), center2minmax(b)

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
    intersection = torch.cat((torch.max(a[:, 0], b[:, 0]).unsqueeze(1),
                              torch.max(a[:, 1], b[:, 1]).unsqueeze(1),
                              torch.min(a[:, 2], b[:, 2]).unsqueeze(1),
                              torch.min(a[:, 3], b[:, 3]).unsqueeze(1)), dim=1)
    # get intersection's area
    # (w, h) = (xmax - xmin, ymax - ymin)
    intersection_w, intersection_h = intersection[:, 2] - intersection[:, 0], intersection[:, 3] - intersection[:, 1]
    # if intersection's width or height is negative, those will be converted to zero
    intersection_w, intersection_h = torch.clamp(intersection_w, min=0), torch.clamp(intersection_h, min=0)

    intersectionArea = intersection_w * intersection_h

    # get a and b's area
    # area = (xmax - xmin) * (ymax - ymin)
    A, B = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

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
    return torch.cat((a[:, 2:] + a[:, :2]/2, a[:, 2:] - a[:, :2]/2), dim=1)

def gt_boxes_converter(gt_boxes, default_boxes):
    """
    :param gt_boxes:
    :param default_boxes:
    :return:
        gt_boxes: Tensor, calculate ground truth value considering default boxes. The formula is below;
                  gt_cx = (gt_cx - dbox_cx)/dbox_w, gt_cy = (gt_cy - dbox_cy)/dbox_h,
                  gt_w = log(gt_w / dbox_w), gt_h = log(gt_h / dbox_h)
    """
    assert gt_boxes.shape == default_boxes.shape, "gt_boxes and default_boxes must be same shape"

    gt_cx = (gt_boxes[:, 0] - default_boxes[:, 0])/default_boxes[:, 2]
    gt_cy = (gt_boxes[:, 1] - default_boxes[:, 1])/default_boxes[:, 3]
    gt_w = torch.log(gt_boxes[:, 2] / default_boxes[:, 2])
    gt_h = torch.log(gt_boxes[:, 3] / default_boxes[:, 3])

    return torch.cat((gt_cx.unsqueeze(1),
                      gt_cy.unsqueeze(1),
                      gt_w.unsqueeze(1),
                      gt_h.unsqueeze(1)), dim=1)


