from torch import nn
import torch

class PredictorBase(nn.Module):
    def __init__(self, class_nums):
        super().__init__()

        self._class_nums = class_nums

    @property
    def class_nums(self):
        return self._class_nums


class Predictor(PredictorBase):
    def __init__(self, class_nums):
        super().__init__(class_nums)

    def forward(self, locs, confs):
        """
        :param locs: list of Tensor, Tensor's shape is (batch, c, h, w)
        :param confs: list of Tensor, Tensor's shape is (batch, c, h, w)
        :return: predicts: localization and confidence Tensor, shape is (batch, total_dbox_num * (4+class_labels))
        """
        locs_reshaped, confs_reshaped = [], []
        for loc, conf in zip(locs, confs):
            batch_num = loc.shape[0]

            # original feature => (batch, (class_num or 4)*dboxnum, fmap_h, fmap_w)
            # converted into (batch, fmap_h, fmap_w, (class_num or 4)*dboxnum)
            # contiguous means aligning stored 1-d memory for given array
            loc = loc.permute((0, 2, 3, 1)).contiguous()
            locs_reshaped += [loc.reshape((batch_num, -1))]

            conf = conf.permute((0, 2, 3, 1)).contiguous()
            confs_reshaped += [conf.reshape((batch_num, -1))]



        locs_reshaped = torch.cat(locs_reshaped, dim=1).reshape((batch_num, -1, 4))
        confs_reshaped = torch.cat(confs_reshaped, dim=1).reshape((batch_num, -1, self.class_nums))

        return torch.cat((locs_reshaped, confs_reshaped), dim=2)