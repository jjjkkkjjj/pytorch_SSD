from torch import nn
import torch
import math
import torch.nn.functional as F


class SSDLoss(nn.Module):
    def __init__(self, alpha=1, loc_loss=None, conf_loss=None):
        super().__init__()

        self.alpha = alpha
        self.loc_loss = LocalizationLoss() if loc_loss is None else loc_loss
        self.conf_loss = ConfidenceLoss() if conf_loss is None else conf_loss

    def forward(self, pos_indicator, predicts, gts):
        """
        :param pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
        :param predicts: Tensor, shape is (batch, total_dbox_nums, 4+class_nums=(cx, cy, w, h, p_class,...)
        :param gts: Tensor, shape is (batch, total_dbox_nums, 4+class_nums=(cx, cy, w, h, p_class,...)
        :return:
            loss: float
        """
        # get localization and confidence from predicts and gts respectively
        pred_loc, pred_conf = predicts[:, :, :4], predicts[:, :, 4:]
        gt_loc, gt_conf = gts[:, :, :4], gts[:, :, 4:]

        # Localization loss
        loc_loss = self.loc_loss(pos_indicator, pred_loc, gt_loc)

        # Confidence loss
        conf_loss = self.conf_loss(pos_indicator, pred_conf, gt_conf)

        return conf_loss, loc_loss


class LocalizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_indicator, predicts, gts):
        N = pos_indicator.int().sum()

        loss = F.smooth_l1_loss(predicts, gts, reduction='none').sum(dim=-1) # shape = (batch num, dboxes num)
        loss = loss.masked_select(pos_indicator)

        return loss.sum() / N


class ConfidenceLoss(nn.Module):
    def __init__(self, neg_factor=3, hnm_batch=True):
        """
        :param neg_factor: int, the ratio(1(pos): neg_factor) to learn pos and neg for hard negative mining
        :param hnm_batch: bool, whether to do hard negative mining for each batch
        """
        super().__init__()
        self._neg_factor = neg_factor
        self.hnm_batch = hnm_batch

    def forward(self, pos_indicator, predicts, gts):
        if self.hnm_batch:
            background_loss = -F.log_softmax(predicts, dim=-1)[:, :, -1] # shape = (batch num, dboxes num)
            background_loss.masked_fill_(pos_indicator, -math.inf)

            pos_num = pos_indicator.sum(dim=-1) # shape = (dboxes num)
            N = pos_num.sum()

            neg_num = predicts.shape[1] - pos_num
            neg_num = torch.min(neg_num, self._neg_factor * pos_num) # shape = (dboxes num)

            _, indices = torch.sort(background_loss, dim=-1, descending=True)
            _, rank = torch.sort(indices, dim=-1)
            neg_indicator = rank < neg_num.unsqueeze(1).expand_as(rank)
            mask = pos_indicator | neg_indicator

            labels = gts.argmax(dim=-1)

            predicts = predicts[mask]
            labels = labels[mask]

            loss = F.cross_entropy(predicts, labels, reduction='sum')

            return loss / N

        else:
            neg_indicator = torch.logical_not(pos_indicator)

            # all loss
            loss = -gts * F.log_softmax(predicts, dim=-1)  # shape = (batch num, dboxes num, class_num)
            # get positive loss
            pos_loss = loss[pos_indicator].sum()
            N = pos_indicator.float().sum()

            # calculate negative loss
            neg_loss = loss[neg_indicator][:, -1]
            neg_num = neg_indicator.float().sum()

            # hard negative mining
            neg_num = int(min(N * self._neg_factor, neg_num).item())
            # -1 means last index. last index represents background which is negative
            neg_loss, neg_loss_max_indices = neg_loss.topk(neg_num)

            return (pos_loss.sum() + neg_loss.sum()) / N
