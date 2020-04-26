from torch import nn
import torch
import logging, math
import torch.nn.functional as F

from .boxes import *

class SSDLoss(nn.Module):
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
        # get predict's localization and confidence
        pred_loc, pred_conf = predicts[:, :, :4], predicts[:, :, 4:]

        # matching
        pos_indicator, gt_loc, gt_conf = self.matching_strategy(gts, dboxes, batch_num=predicts.shape[0], threshold=0.5)

        # calculate ground truth value considering default boxes
        gt_loc = gt_loc_converter(gt_loc, dboxes)

        # Localization loss
        loc_loss = self.loc_loss(pos_indicator, pred_loc, gt_loc)

        # Confidence loss
        conf_loss = self.conf_loss(pos_indicator, pred_conf, gt_conf)

        return conf_loss + self.alpha * loc_loss


class LocalizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_indicator, predicts, gts):
        N = pos_indicator.sum()

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

        background_loss = -F.log_softmax(predicts, dim=-1)[:, :, -1] # shape = (batch num, dboxes num)
        background_loss[pos_indicator] = -math.inf

        N = pos_indicator.sum()

        neg_num = predicts.shape[0] - N
        neg_num = min(neg_num, self._neg_factor * N)

        _, indices = torch.sort(background_loss)
        _, rank = torch.sort(indices)
        neg_indicator = rank < neg_num
        mask = pos_indicator | neg_indicator

        labels = gts.argmax(dim=-1)

        predicts = predicts[mask]
        labels = labels[mask]

        loss = F.cross_entropy(predicts, labels, reduction='sum')

        return loss / N


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
        """
        # get positive loss
        pos_loss = -self.logsoftmax(predicts[pos_indicator], gts[pos_indicator]) # shape = (-1, class_num)
        pos_num = pos_loss.shape[0]
        # get class
        # print(gts[pos_indicator][0].bool())
        # tensor([False, False, False, False, False, False, False, False, False, False,
        #         False, False, False, False, False, False, False, False,  True, False,
        #         False], device='cuda:0')
        pos_mask = gts[pos_indicator].bool()

        # calculate negative loss
        neg_indicator = torch.logical_not(pos_indicator)
        neg_loss = -self.logsoftmax(predicts[neg_indicator], gts[neg_indicator]) # shape = (-1, class_num)
        neg_num = neg_loss.shape[0]

        # hard negative mining
        neg_num = min(pos_num * self._neg_factor, neg_num)
        # -1 means last index. last index represents background which is negative
        _, neg_loss_max_indices = neg_loss[:, -1].sort(dim=0, descending=True)
        neg_topk_mask = neg_loss_max_indices[:neg_num] # shape = (neg_num)

        #_, topk_mask = torch.topk(neg_loss, min(pos_num * self._neg_factor, neg_num), dim=1, sorted=False)
        # error...
        # RuntimeError: invalid argument 5: k not in range for dimension at /opt/conda/conda-bld/pytorch_1579022034529/work/aten/src/THC/generic/THCTensorTopK.cu:23

        # -1 means last index. last index represents background which is negative
        return pos_loss[pos_mask].sum() + neg_loss[neg_topk_mask, -1].sum()
        """

class LogSoftmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, predicts, targets):
        exp = torch.exp(predicts.clamp(min=1e-15, max=1-1e-15))
        softmax = exp / torch.sum(exp, dim=self.dim, keepdim=True)

        #softmax = softmax.clamp(min=1e-15, max=1-1e-15)

        return targets * torch.log(softmax)
