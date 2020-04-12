from torch import nn
import torch

from .utils import matching_strategy, gt_boxes_converter

class DefaultBoxLoss(nn.Module):
    def __init__(self, alpha=1, matching_func=None, loc_loss=None, conf_loss=None):
        super().__init__()

        self.alpha = alpha
        self.matching_strategy = matching_strategy if matching_func is None else matching_func
        self.loc_loss = LocalizationLoss() if loc_loss is None else loc_loss
        self.conf_loss = ConfidenceLoss() if conf_loss is None else conf_loss


    def forward(self, predicts, gts, dboxes):
        """
        :param predicts: Tensor, shape is (batch, total_dbox_nums, 4+class_nums=(cx, cy, w, h, p_class,...)
        :param gts: Tensor, shape is (batch*bbox_nums(batch), 1+4+class_nums) = [[img's_ind, cx, cy, w, h, p_class,...],..
        :param dboxes: Tensor, shape is (total_dbox_nums, 4=(cx,cy,w,h))
        :return:
            loss: float
        """

        # get box number per image
        gt_boxnum_per_image = gts[:, 0]
        # remove above information from gts
        gts = gts[:, 1:]

        # align shape of predicts, gts and dboxes respectively by using gt_boxnum_per_image.
        gt_total_objects_num = gts.shape[0]
        dbox_total_num = dboxes.shape[0]

        predicts = _align_predicts(predicts, gt_boxnum_per_image)
        gts = _align_gts(gts, dbox_total_num)
        dboxes = _align_dboxes(dboxes, gt_total_objects_num)

        # get localization and confidence for predicts and gts respectively
        pred_loc, pred_conf = predicts[:, :4], predicts[:, 4:]
        gt_loc, gt_conf = gts[:, :4], gts[:, 4:]

        # matching
        indicator = self.matching_strategy(gt_loc, dboxes, threshold=0.5)
        N = torch.sum(indicator).item()

        # calculate ground truth value considering default boxes
        gt_loc = gt_boxes_converter(gt_loc, dboxes)

        # Localization loss
        loc_lossval = self.loc_loss(indicator, pred_loc, gt_loc)

        # Confidence loss
        conf_lossval = self.conf_loss(indicator, pred_conf, gt_conf)

        return (conf_lossval + self.alpha*loc_lossval) / N


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


def _align_predicts(predicts, gt_boxnum_per_image):
    """
    :param predicts: Tensor, shape is (batch, total_dbox_nums, 4+class_nums=(cx, cy, w, h, p_class,...)
    :param gt_boxnum_per_image: Tensor, shape is (batch*bbox_nums(batch))
            e.g.) gt_boxnum_per_image = (2,2,1,3,3,3,2,2,...)
            shortly, box number value is arranged for each box number
    :return:
        predicts: Tensor, shape
    """
    ret_predicts = []
    batch_num = predicts.shape[0]
    index = 0
    for b in range(batch_num):
        box_num = int(gt_boxnum_per_image[index].item())
        ret_predicts += [torch.cat(box_num * [predicts[b]])]

        index += box_num

    return torch.cat(ret_predicts, dim=0)


def _align_gts(gts, dbox_total_num):
    return torch.repeat_interleave(gts, dbox_total_num, dim=0)


def _align_dboxes(dboxes, gt_total_objects_num):
    return torch.cat(gt_total_objects_num * [dboxes])


class LocalizationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smoothL1Loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, indicator, predicts, gts):
        loss = self.smoothL1Loss(predicts[indicator], gts[indicator])

        return loss

class ConfidenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logsoftmaxLoss = nn.LogSoftmax(dim=1)# need to check

    def forward(self, indicator, predicts, gts):

        -self.logsoftmaxLoss() - self.logsoftmaxLoss()
