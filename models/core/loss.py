from torch import nn
import torch
import logging, math
import torch.nn.functional as F

from .boxes import *

class SSDLoss(nn.Module):
    def __init__(self, alpha=1, matching_func=None, loc_loss=None, conf_loss=None, encoder=None):
        super().__init__()

        self.alpha = alpha
        self.matching_strategy = matching_strategy if matching_func is None else matching_func
        self.loc_loss = LocalizationLoss() if loc_loss is None else loc_loss
        self.conf_loss = ConfidenceLoss() if conf_loss is None else conf_loss
        self.encoder = Encoder() if encoder is None else encoder

    def forward(self, predicts, gts, dboxes):
        """
        :param predicts: Tensor, shape is (batch, total_dbox_nums, 4+class_nums=(cx, cy, w, h, p_class,...)
        :param gts: Tensor, shape is (batch*bbox_nums(batch), 1+4+class_nums) = [[img's_ind, cx, cy, w, h, p_class,...],..
        :param dboxes: Tensor, shape is (total_dbox_nums, 4=(cx,cy,w,h))
        :return:
            loss: float
        """
        # get predict's localization and confidence
        pred_loc, pred_conf = predicts[:, :, :4], predicts[:, :, 4:]

        # matching
        pos_indicator, gt_loc, gt_conf = self.matching_strategy(gts, dboxes, batch_num=predicts.shape[0], threshold=0.5)

        # calculate ground truth value considering default boxes
        gt_loc = self.encoder(gt_loc, dboxes)

        # Localization loss
        loc_loss = self.loc_loss(pos_indicator, pred_loc, gt_loc)

        # Confidence loss
        conf_loss = self.conf_loss(pos_indicator, pred_conf, gt_conf)

        return conf_loss, loc_loss


class LocalizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_indicator, predicts, gts):
        N = pos_indicator.float().sum()

        predicts = predicts[pos_indicator]
        gts = gts[pos_indicator]
        loss = F.smooth_l1_loss(predicts, gts, reduction='sum') # shape = (batch num, dboxes num)

        return loss / N
        """
        batch_num = predicts.shape[0]

        total_loss = self.smoothL1Loss(predicts, gts).sum(dim=-1) # shape = (batch num, dboxes num)

        loss = torch.zeros((batch_num), device=predicts.device, requires_grad=False)
        for b in range(batch_num):
            mask = pos_indicator[b] # shape = (dboxes num)
            N = mask.sum()

            if N.item() == 0:
                logging.warning('cannot assign object boxes!!!')
                loss[b] = torch.zeros((1))
                continue

            pn_loss = total_loss[b]
            loss[b] = pn_loss.masked_select(mask).sum() / N

        return loss
        """

class ConfidenceLoss(nn.Module):
    def __init__(self, neg_factor=3):
        """
        :param neg_factor: int, the ratio(1(pos): neg_factor) to learn pos and neg for hard negative mining
        """
        super().__init__()
        self._neg_factor = neg_factor

    def forward(self, pos_indicator, predicts, gts):
        """
        background_loss = -F.log_softmax(predicts, dim=-1)[:, :, -1] # shape = (batch num, dboxes num)
        background_loss[pos_indicator] = -math.inf

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
        """

        """
        batch_num = predicts.shape[0]

        total_loss, _  = (-gts * self.logsoftmax(predicts)).max(dim=-1) # shape = (batch num, dboxes num)

        loss = torch.zeros((batch_num), device=predicts.device, requires_grad=False)
        for b in range(batch_num):
            p_mask = pos_indicator[b]  # shape = (dboxes num)
            n_mask = torch.logical_not(p_mask)
            N = p_mask.sum()

            if N.item() == 0:
                logging.warning('cannot assign object boxes!!!')
                loss[b] = torch.zeros((1))
                continue

            pn_loss = total_loss[b]
            p_loss = pn_loss.masked_select(p_mask)
            n_loss = pn_loss.masked_select(n_mask)

            # hard negative mining
            # -1 means negative which represents background
            neg_num = n_loss.shape[0]
            _, neg_indices = n_loss.sort(descending=True)
            _, rank = neg_indices.sort()

            neg_num = min(neg_num, N * self._neg_factor)
            neg_mask = rank < neg_num

            loss[b] = (p_loss.sum() + n_loss.masked_select(neg_mask).sum()) / N

        return loss
        """
        neg_indicator = torch.logical_not(pos_indicator)

        # all loss
        loss = -gts * F.log_softmax(predicts, dim=-1) # shape = (batch num, dboxes num, class_num)
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


class LogSoftmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, predicts, targets):
        exp = torch.exp(predicts.clamp(min=1e-15, max=1-1e-15))
        softmax = exp / torch.sum(exp, dim=self.dim, keepdim=True)

        #softmax = softmax.clamp(min=1e-15, max=1-1e-15)

        return targets * torch.log(softmax)
