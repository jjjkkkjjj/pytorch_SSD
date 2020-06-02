import torch

def batch_ind_fn(batch):
    """
    :param batch:
    :return:
        imgs: Tensor, shape = (b, c, h, w)
        targets: list of Tensor, whose shape = (object box num, 4 + class num) including background
    """
    imgs, gts = list(zip(*batch))

    return torch.stack(imgs), gts