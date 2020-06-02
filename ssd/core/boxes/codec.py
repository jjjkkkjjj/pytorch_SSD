from torch import nn
import torch

from .utils import matching_strategy

class Codec(nn.Module):
    def __init__(self, norm_means=(0.0, 0.0, 0.0, 0.0), norm_stds=(0.1, 0.1, 0.2, 0.2)):
        super().__init__()
        # shape = (1, 1, 4=(cx, cy, w, h)) or (1, 1, 1)
        self.norm_means = norm_means
        self.norm_stds = norm_stds

        self.encoder = Encoder(self.norm_means, self.norm_stds)
        self.decoder = Decoder(self.norm_means, self.norm_stds)

class Encoder(nn.Module):
    def __init__(self, norm_means=(0.0, 0.0, 0.0, 0.0), norm_stds=(0.1, 0.1, 0.2, 0.2)):
        super().__init__()
        # shape = (1, 1, 4=(cx, cy, w, h)) or (1, 1, 1)
        self.norm_means = torch.tensor(norm_means, requires_grad=False).unsqueeze(0).unsqueeze(0)
        self.norm_stds = torch.tensor(norm_stds, requires_grad=False).unsqueeze(0).unsqueeze(0)


    def forward(self, gts, dboxes, batch_num):
        """
        :param gts: Tensor, shape is (batch*object num(batch), 1+4+class_nums)
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
        # gts: Tensor, shape = (batch, default box num, 4+class_num) including background
        pos_indicator, gts = matching_strategy(gts, dboxes, batch_num=batch_num)

        # encoding
        # gt_boxes: Tensor, shape = (batch, default boxes num, 4)
        gt_boxes = gts[:, :, :4]

        assert gt_boxes.shape[1:] == dboxes.shape, "gt_boxes and default_boxes must be same shape"

        gt_cx = (gt_boxes[:, :, 0] - dboxes[:, 0]) / dboxes[:, 2]
        gt_cy = (gt_boxes[:, :, 1] - dboxes[:, 1]) / dboxes[:, 3]
        gt_w = torch.log(gt_boxes[:, :, 2] / dboxes[:, 2])
        gt_h = torch.log(gt_boxes[:, :, 3] / dboxes[:, 3])

        encoded_boxes = torch.cat((gt_cx.unsqueeze(2),
                          gt_cy.unsqueeze(2),
                          gt_w.unsqueeze(2),
                          gt_h.unsqueeze(2)), dim=2)

        # normalization
        gts[:, :, :4] = (encoded_boxes - self.norm_means.to(gt_boxes.device)) / self.norm_stds.to(gt_boxes.device)

        return pos_indicator, gts


class Decoder(nn.Module):
    def __init__(self, norm_means=(0.0, 0.0, 0.0, 0.0), norm_stds=(0.1, 0.1, 0.2, 0.2)):
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

