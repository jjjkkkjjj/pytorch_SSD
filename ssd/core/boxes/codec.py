from torch import nn
import torch

from .utils import matching_strategy
from ..._utils import _check_norm, _check_ins
import torchvision

class EncoderBase(nn.Module):
    pass

class DecoderBase(nn.Module):
    pass


class CodecBase(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = _check_ins('encoder', encoder, EncoderBase)
        self.decoder = _check_ins('decoder', decoder, DecoderBase)

    def to(self, *args, **kwargs):
        self.encoder = self.encoder.to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        self.encoder = self.encoder.cuda(device)
        self.decoder = self.decoder.cuda(device)

        return super().cuda(device)


class Codec(CodecBase):
    def __init__(self, norm_means=(0.0, 0.0, 0.0, 0.0), norm_stds=(0.1, 0.1, 0.2, 0.2)):
        # shape = (1, 1, 4=(cx, cy, w, h)) or (1, 1, 1)
        self.norm_means = norm_means
        self.norm_stds = norm_stds

        super().__init__(Encoder(self.norm_means, self.norm_stds), Decoder(self.norm_means, self.norm_stds))


class Encoder(EncoderBase):
    def __init__(self, norm_means=(0.0, 0.0, 0.0, 0.0), norm_stds=(0.1, 0.1, 0.2, 0.2)):
        super().__init__()

        norm_means = _check_norm('norm_means', norm_means)
        norm_stds = _check_norm('norm_stds', norm_stds)

        # shape = (1, 1, 4=(cx, cy, w, h)) or (1, 1, 1)
        self.norm_means = norm_means.unsqueeze(0).unsqueeze(0)
        self.norm_stds = norm_stds.unsqueeze(0).unsqueeze(0)


    def forward(self, targets, dboxes, batch_num):
        """
        :param targets: Tensor, shape is (batch*object num(batch), 1+4+class_labels)
        :param dboxes: Tensor, shape is (total_dbox_nums, 4=(cx,cy,w,h))
        :param batch_num: int
        :return:
            pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
            encoded_boxes: Tensor, calculate ground truth value considering default boxes. The formula is below;
                           gt_cx = (gt_cx - dbox_cx)/dbox_w, gt_cy = (gt_cy - dbox_cy)/dbox_h,
                           gt_w = train(gt_w / dbox_w), gt_h = train(gt_h / dbox_h)
                           shape = (batch, default boxes num, 4)
        """
        # matching
        # pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
        # targets: Tensor, shape = (batch, default box num, 4+class_num) including background
        pos_indicator, targets = matching_strategy(targets, dboxes, batch_num=batch_num)

        # encoding
        # targets_loc: Tensor, shape = (batch, default boxes num, 4)
        targets_loc = targets[:, :, :4]

        assert targets_loc.shape[1:] == dboxes.shape, "targets_loc and default_boxes must be same shape"

        gt_cx = (targets_loc[:, :, 0] - dboxes[:, 0]) / dboxes[:, 2]
        gt_cy = (targets_loc[:, :, 1] - dboxes[:, 1]) / dboxes[:, 3]
        gt_w = torch.log(targets_loc[:, :, 2] / dboxes[:, 2])
        gt_h = torch.log(targets_loc[:, :, 3] / dboxes[:, 3])

        encoded_boxes = torch.cat((gt_cx.unsqueeze(2),
                                   gt_cy.unsqueeze(2),
                                   gt_w.unsqueeze(2),
                                   gt_h.unsqueeze(2)), dim=2)

        # normalization
        targets[:, :, :4] = (encoded_boxes - self.norm_means) / self.norm_stds

        return pos_indicator, targets

    def to(self, *args, **kwargs):
        self.norm_means = self.norm_means.to(*args, **kwargs)
        self.norm_stds = self.norm_stds.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        self.norm_means = self.norm_means.cuda(device)
        self.norm_stds = self.norm_stds.cuda(device)

        return super().cuda(device)


class Decoder(DecoderBase):
    def __init__(self, norm_means=(0.0, 0.0, 0.0, 0.0), norm_stds=(0.1, 0.1, 0.2, 0.2)):
        super().__init__()

        norm_means = _check_norm('norm_means', norm_means)
        norm_stds = _check_norm('norm_stds', norm_stds)

        # shape = (1, 1, 4=(cx, cy, w, h)) or (1, 1, 1)
        self.norm_means = norm_means.unsqueeze(0).unsqueeze(0)
        self.norm_stds = norm_stds.unsqueeze(0).unsqueeze(0)

    def forward(self, predicts, default_boxes):
        """
        Opposite to above procession
        :param predicts: Tensor, shape = (batch, default boxes num, 4 + class_nums)
        :param default_boxes: Tensor, shape = (default boxes num, 4)
        Note that 4 means (cx, cy, w, h)
        :return:
            inf_boxes: Tensor, calculate ground truth value considering default boxes. The formula is below;
                      inf_cx = pred_cx * dbox_w + dbox_cx, inf_cy = pred_cy * dbox_h + dbox_cy,
                      inf_w = exp(pred_w) * dbox_w, inf_h = exp(pred_h) * dbox_h
                      shape = (batch, default boxes num, 4)
        """
        pred_locs = predicts[:, :, :4]

        assert pred_locs.shape[1:] == default_boxes.shape, "predicts and default_boxes must be same shape"

        pred_unnormalized = pred_locs * self.norm_stds + self.norm_means

        inf_cx = pred_unnormalized[:, :, 0] * default_boxes[:, 2] + default_boxes[:, 0]
        inf_cy = pred_unnormalized[:, :, 1] * default_boxes[:, 3] + default_boxes[:, 1]
        inf_w = torch.exp(pred_unnormalized[:, :, 2]) * default_boxes[:, 2]
        inf_h = torch.exp(pred_unnormalized[:, :, 3]) * default_boxes[:, 3]

        predicts[:, :, :4] = torch.cat((inf_cx.unsqueeze(2),
                                        inf_cy.unsqueeze(2),
                                        inf_w.unsqueeze(2),
                                        inf_h.unsqueeze(2)), dim=2)

        return predicts

    def to(self, *args, **kwargs):
        self.norm_means = self.norm_means.to(*args, **kwargs)
        self.norm_stds = self.norm_stds.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        self.norm_means = self.norm_means.cuda(device)
        self.norm_stds = self.norm_stds.cuda(device)

        return super().cuda(device)
