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

    def forward(self, pos_indicator, predicts, targets):
        """
        :param pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
        :param predicts: Tensor, shape is (batch, total_dbox_nums, 4+class_labels=(cx, cy, w, h, p_class,...)
        :param targets: Tensor, shape is (batch, total_dbox_nums, 4+class_labels=(cx, cy, w, h, p_class,...)
        :return:
            loss: float
        """
        # get localization and confidence from predicts and targets respectively
        pred_loc, pred_conf = predicts[:, :, :4], predicts[:, :, 4:]
        targets_loc, targets_conf = targets[:, :, :4], targets[:, :, 4:]

        # Localization loss
        loc_loss = self.loc_loss(pos_indicator, pred_loc, targets_loc)

        # Confidence loss
        conf_loss = self.conf_loss(pos_indicator, pred_conf, targets_conf)

        return conf_loss, loc_loss


class LocalizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_indicator, predicts, targets):
        """
        :param pos_indicator: Tensor, shape = (batch num, dbox num)
        :param predicts: Tensor, shape = (batch num, dbox num, 4=(cx, cy, w, h))
        :param targets: Tensor, shape = (batch num, dbox num, 4=(cx, cy, w, h))
        :return:
        """
        """
        neg_indicator = torch.logical_not(pos_indicator)

        N = pos_indicator.sum(dim=-1) # shape = (batch num)
        N = torch.max(N, torch.ones_like(N))

        loss = F.smooth_l1_loss(predicts, targets, reduction='none').sum(dim=-1)  # shape = (batch num, dboxes num)
        loss.masked_fill_(neg_indicator, 0)

        return (loss.sum(dim=-1) / N).mean()
        """
        N = pos_indicator.sum().float()

        loss = F.smooth_l1_loss(predicts, targets, reduction='none').sum(dim=-1)  # shape = (batch num, dbox num)

        return loss.masked_select(pos_indicator).sum() / N


class ConfidenceLoss(nn.Module):
    def __init__(self, neg_factor=3, hnm_batch=True):
        """
        :param neg_factor: int, the ratio(1(pos): neg_factor) to learn pos and neg for hard negative mining
        :param hnm_batch: bool, whether to do hard negative mining for each batch
        """
        super().__init__()
        self._neg_factor = neg_factor
        self.hnm_batch = hnm_batch

    def forward(self, pos_indicator, predicts, targets):
        """
        :param pos_indicator: Tensor, shape = (batch num, dbox num)
        :param predicts: Tensor, shape = (batch num, dbox num, class num) including background
        :param targets: Tensor, shape = (batch num, dbox num, class num) including background
        :return:
        """
        if self.hnm_batch:
            background_loss = -F.log_softmax(predicts, dim=-1)[:, :, -1] # shape = (batch num, dboxes num)
            background_loss.masked_fill_(pos_indicator, -math.inf)

            pos_num = pos_indicator.sum(dim=-1) # shape = (batch num)
            N = pos_num.sum().float()

            neg_num = predicts.shape[1] - pos_num
            neg_num = torch.min(neg_num, self._neg_factor * pos_num) # shape = (batch num)

            _, indices = torch.sort(background_loss, dim=-1, descending=True)
            _, rank = torch.sort(indices, dim=-1)
            neg_indicator = rank < neg_num.unsqueeze(1).expand_as(rank)
            mask = pos_indicator | neg_indicator

            labels = targets.argmax(dim=-1)

            """
            batch_num = pos_num.shape[0]
            loss = torch.empty((batch_num), device=predicts.device, dtype=torch.float) # shape = (batch num)
            for b in range(batch_num):
                loss[b] = F.cross_entropy(predicts[b, mask[b]], labels[b, mask[b]], reduction='sum')

            pos_num = torch.max(pos_num, torch.ones_like(pos_num))

            return (loss / pos_num).mean()
            """

            loss = F.cross_entropy(predicts[mask], labels[mask], reduction='sum')

            return loss / N

        else:
            neg_indicator = torch.logical_not(pos_indicator)

            # all loss
            loss = -targets * F.log_softmax(predicts, dim=-1)  # shape = (batch num, dboxes num, class_num)
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
