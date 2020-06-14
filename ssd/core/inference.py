from .boxes import iou, centroids2corners
from .._utils import _check_ins
from ssd.core.boxes.codec import Decoder

from torch.nn import Module
from torch.nn import functional as F
import torch, cv2
import math
import numpy as np

class InferenceBoxBase(Module):
    def __init__(self, class_nums_with_background, filter_func, val_config):
        super().__init__()
        self.class_nums_with_background = class_nums_with_background
        self.filter_func = filter_func

        from ..models.base import SSDValConfig
        self.val_config = _check_ins('val_config', val_config, SSDValConfig)
        
        self.device = torch.device('cpu')

class InferenceBox(InferenceBoxBase):
    def __init__(self, class_nums_with_background, filter_func, val_config):
        super().__init__(class_nums_with_background, filter_func, val_config)

    @property
    def conf_threshold(self):
        return self.val_config.conf_threshold

    def forward(self, predicts, conf_threshold=None):
        """
        :param predicts: Tensor, shape = (batch number, default boxes number, 4 + class_num)
        :param conf_threshold: float or None, if it's None, passed default value with 0.01
        :return:
            ret_boxes: list of tensor, shape = (box num, 5=(class index, cx, cy, w, h))
        """
        # alias
        batch_num = predicts.shape[0]
        class_num = self.class_nums_with_background
        ret_num = predicts.shape[2] - class_num + 1 + 1 # loc num + 1=(class index) + 1=(conf)

        predicts[:, :, -class_num:] = F.softmax(predicts[:, :, -class_num:], dim=-1)

        conf_threshold = conf_threshold if conf_threshold else self.conf_threshold

        ret_boxes = []
        for b in range(batch_num):
            ret_box = []
            pred = predicts[b] # shape = (default boxes number, *)
            for c in range(class_num - 1): # last index means background
                # filter out less than threshold
                indicator = pred[:, -class_num+c] > conf_threshold
                if indicator.sum().item() == 0:
                    continue

                # shape = (filtered default boxes num, *=loc+1=conf)
                filtered_pred = torch.cat((pred[indicator, :-class_num], pred[indicator, -class_num+c].unsqueeze(1)), dim=1)

                # inferred_indices: Tensor, shape = (inferred boxes num,)
                # inferred_confs: Tensor, shape = (inferred boxes num,)
                inferred_indices, inferred_confs, inferred_locs = self.filter_func(filtered_pred, self.val_config)
                if inferred_indices.nelement() == 0:
                    continue
                else:
                    # append class flag
                    # shape = (inferred boxes num, 1)
                    flag = np.broadcast_to([c], shape=(inferred_indices.nelement(), 1))
                    flag = torch.from_numpy(flag).float().to(self.device)

                    # shape = (inferred box num, 2+loc=(class index, confidence, *))
                    ret_box += [torch.cat((flag, inferred_confs.unsqueeze(1), inferred_locs), dim=1)]

            if len(ret_box) == 0:
                ret_boxes += [torch.from_numpy(np.ones((1, ret_num))*np.nan)]
            else:
                ret_boxes += [torch.cat(ret_box, dim=0)]

        # list of tensor, shape = (box num, ret_num=(class index, confidence, *=loc))
        return ret_boxes


def non_maximum_suppression(pred, val_config):
    """
    :param pred: tensor, shape = (filtered default boxes num, 4=loc + 1=conf)
    Note that filtered default boxes number must be more than 1
    :param val_config: SSDValConfig
    :return:
        inferred_indices: Tensor, shape = (inferred box num,)
        inferred_confs: Tensor, shape = (inferred box num,)
        inferred_locs: Tensor, shape = (inferred box num, 4)
    """
    loc, conf = pred[:, :-1], pred[:, -1]
    iou_threshold = val_config.iou_threshold
    topk = val_config.topk

    # sort confidence and default boxes with descending order
    c, conf_des_inds = conf.sort(dim=0, descending=True)
    # get topk indices
    conf_des_inds = conf_des_inds[:topk]
    # converted into minmax coordinates
    loc_mm = centroids2corners(loc)

    inferred_indices = []
    while conf_des_inds.nelement() > 0:
        largest_conf_index = conf_des_inds[0]

        largest_conf_loc = loc[largest_conf_index, :].unsqueeze(0)  # shape = (1, 4=(xmin, ymin, xmax, ymax))
        # append to result
        inferred_indices.append(largest_conf_index)

        # remove largest element
        conf_des_inds = conf_des_inds[1:]

        if conf_des_inds.nelement() == 0:
            break

        # get iou, shape = (1, loc_des num)
        overlap = iou(centroids2corners(largest_conf_loc), loc_mm[conf_des_inds])
        # filter out overlapped boxes for box with largest conf, shape = (loc_des num)
        indicator = overlap.reshape((overlap.nelement())) <= iou_threshold

        conf_des_inds = conf_des_inds[indicator]

    inferred_indices = torch.Tensor(inferred_indices).long()
    return inferred_indices, conf[inferred_indices], loc[inferred_indices]

def tensor2cvrgbimg(img, to8bit=True):
    if to8bit:
        img = img * 255.
    return img.cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)

def toVisualizeRectangleRGBimg(img, locs, thickness=2, rgb=(255, 0, 0), verbose=False):
    """
    :param img: Tensor, shape = (c, h, w)
    :param locs: Tensor, centered coordinates, shape = (box num, 4=(cx, cy, w, h)).
    :param thickness: int
    :param rgb: tuple of int, order is rgb and range is 0~255
    :param verbose: bool, whether to show information
    :return:
        img: RGB order
    """
    # convert (c, h, w) to (h, w, c)
    img = tensor2cvrgbimg(img, to8bit=True).copy()
    #cv2.imshow('a', img)
    #cv2.waitKey()
    # print(locs)
    locs_mm = centroids2corners(locs).cpu().numpy()

    h, w, c = img.shape
    locs_mm[:, 0::2] *= w
    locs_mm[:, 1::2] *= h
    locs_mm[:, 0::2] = np.clip(locs_mm[:, 0::2], 0, w).astype(int)
    locs_mm[:, 1::2] = np.clip(locs_mm[:, 1::2], 0, h).astype(int)
    locs_mm = locs_mm.astype(int)

    if verbose:
        print(locs_mm)
    for bnum in range(locs_mm.shape[0]):
        topleft = locs_mm[bnum, :2]
        bottomright = locs_mm[bnum, 2:]

        if verbose:
            print(tuple(topleft), tuple(bottomright))

        cv2.rectangle(img, tuple(topleft), tuple(bottomright), rgb, thickness)

    return img

def toVisualizeRGBImg(img, locs, inf_labels, classe_labels, inf_confs=None, verbose=False):
    """
    :param img: Tensor, shape = (c, h, w)
    :param locs: Tensor, centered coordinates, shape = (box num, 4=(cx, cy, w, h)).
    :param inf_labels:
    :param classe_labels: list of str
    :param inf_confs: Tensor, (box_num,)
    :param verbose:
    :return:
    """
    # convert (c, h, w) to (h, w, c)
    img = tensor2cvrgbimg(img)

    class_num = len(classe_labels)
    box_num = locs.shape[0]
    assert box_num == inf_labels.shape[0], 'must be same boxes number'
    if inf_confs is not None:
        if isinstance(inf_confs, torch.Tensor):
            inf_confs = inf_confs.cpu().numpy()
        elif not isinstance(inf_confs, np.ndarray):
            raise ValueError(
                'Invalid \'inf_confs\' argment were passed. inf_confs must be ndarray or Tensor, but got {}'.format(
                    type(inf_confs).__name__))
        assert inf_confs.ndim == 1 and inf_confs.size == box_num, "Invalid inf_confs"

    # color
    angles = np.linspace(0, 255, class_num).astype(np.uint8)
    # print(angles.shape)
    hsvs = np.array((0, 255, 255))[np.newaxis, np.newaxis, :].astype(np.uint8)
    hsvs = np.repeat(hsvs, class_num, axis=0)
    # print(hsvs.shape)
    hsvs[:, 0, 0] += angles
    rgbs = cv2.cvtColor(hsvs, cv2.COLOR_HSV2RGB).astype(np.int)

    # Line thickness of 2 px
    thickness = 1



    h, w, c = img.shape
    # print(locs)
    locs_mm = centroids2corners(locs).cpu().numpy()
    locs_mm[:, ::2] *= w
    locs_mm[:, 1::2] *= h
    locs_mm = locs_mm
    locs_mm[:, 0::2] = np.clip(locs_mm[:, 0::2], 0, w).astype(int)
    locs_mm[:, 1::2] = np.clip(locs_mm[:, 1::2], 0, h).astype(int)
    locs_mm = locs_mm.astype(int)

    if verbose:
        print(locs_mm)
    for bnum in range(box_num):# box num
        img = img.copy()

        rect_topleft = locs_mm[bnum, :2]
        rect_bottomright = locs_mm[bnum, 2:]

        if verbose:
            print(tuple(rect_topleft), tuple(rect_bottomright))

        index = inf_labels[bnum].item()
        if math.isnan(index):
            continue
        index = int(index)

        if inf_confs is not None:
            text = classe_labels[index] + ':{:.2f}'.format(inf_confs[bnum])
        else:
            text = classe_labels[index]

        labelSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)

        rect_bottomright = tuple(rect_bottomright)
        rect_topleft = tuple(rect_topleft)
        rgb = tuple(rgbs[index, 0].tolist())

        # text area
        text_bottomleft = (rect_topleft[0], rect_topleft[1] + int(labelSize[0][1] * 1.5))
        text_topright = (rect_topleft[0] + labelSize[0][0], rect_topleft[1])
        cv2.rectangle(img, text_bottomleft, text_topright, rgb, cv2.FILLED)

        text_bottomleft = (rect_topleft[0], rect_topleft[1] + labelSize[0][1])
        cv2.putText(img, text, text_bottomleft, cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1)

        # rectangle
        cv2.rectangle(img, rect_topleft, rect_bottomright, rgb, thickness)

    return img