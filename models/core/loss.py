from torch import nn

class DefaultBoxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, gt):
        """
        :param predicted: Tensor, shape is (None, 8732)(cx, cy, w, h, p_class,...)
        :param gt: Tensor,
        :return:
        """
        return x