from .boxes import iou, centroids2corners
from ssd.core.boxes.codec import Decoder

from torch.nn import Module
from torch.nn import functional as F
import torch, cv2
import numpy as np

class InferenceBox(Module):
    def __init__(self, conf_threshold=0.01, iou_threshold=0.45, topk=200):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.topk = topk
        self.device = torch.device('cpu')

    def forward(self, inf_cand_loc, inf_cand_conf, conf_threshold=None):
        """
        :param inf_cand_loc: Tensor, shape = (batch number, default boxes number, 4)
        :param inf_cand_conf: Tensor, shape = (batch number, default boxes number, class number)
        :param conf_threshold: float or None, if it's None, passed default value with 0.01
        :return:
            ret_boxes: list of tensor, shape = (box num, 5=(class index, cx, cy, w, h))
        """
        batch_num = inf_cand_loc.shape[0]
        class_num = inf_cand_conf.shape[2]

        conf_threshold = conf_threshold if conf_threshold else self.conf_threshold

        ret_boxes = []
        for b in range(batch_num):
            ret_box = []
            inf_conf = inf_cand_conf[b] # shape = (default boxes number, class number)
            inf_loc = inf_cand_loc[b]
            for c in range(class_num - 1): # last index means background
                # filter out less than threshold
                indicator = inf_conf[:, c] > conf_threshold
                conf = inf_conf[indicator, c] # shape = (filtered default boxes num)
                if conf.nelement() == 0:
                    continue
                loc = inf_loc[indicator, :] # shape = (filtered default boxes num, 4)

                # list of Tensor, shape = (1, 5=(confidence, cx, cy, w, h))
                inferred_boxes = non_maximum_suppression(conf, loc, self.iou_threshold, self.topk)
                if len(inferred_boxes) == 0:
                    continue
                else:
                    # shape = (inferred boxes num, 5)
                    inferred_boxes = torch.cat(inferred_boxes, dim=0)

                    # append class flag
                    # shape = (inferred boxes num, 1)
                    flag = np.broadcast_to([c], shape=(len(inferred_boxes), 1))
                    flag = torch.from_numpy(flag).float().to(self.device)

                    # shape = (inferred box num, 6=(class index, confidence, cx, cy, w, h))
                    ret_box += [torch.cat((flag, inferred_boxes), dim=1)]

            if len(ret_box) == 0:
                ret_boxes += [torch.from_numpy(np.ones((1, 6))*-1)]
            else:
                ret_boxes += [torch.cat(ret_box, dim=0)]

        # list of tensor, shape = (box num, 6=(class index, confidence, cx, cy, w, h))
        return ret_boxes


def non_maximum_suppression(conf, loc, iou_threshold=0.45, topk=200):
    """
    :param conf: tensor, shape = (filtered default boxes num)
    :param loc: tensor, shape = (filtered default boxes num, 4)
    Note that filtered default boxes number must be more than 1
    :param iou_threshold: int
    :return: inferred_boxes: list of inferred boxes(Tensor). inferred boxes' Tensor shape = (inferred boxes number, 5=(conf, cx, cy, w, h))
    """
    # sort confidence and default boxes with descending order
    c, conf_des_inds = conf.sort(dim=0, descending=True)
    # get topk indices
    conf_des_inds = conf_des_inds[:topk]
    # converted into minmax coordinates
    loc_mm = centroids2corners(loc)

    inferred_boxes = []
    while conf_des_inds.nelement() > 0:
        largest_conf_index = conf_des_inds[0]
        # conf[largest_conf_index]'s shape = []
        largest_conf = conf[largest_conf_index].unsqueeze(0).unsqueeze(0) # shape = (1, 1)
        largest_conf_loc = loc[largest_conf_index, :].unsqueeze(0)  # shape = (1, 4=(xmin, ymin, xmax, ymax))
        # append to result
        inferred_boxes.append(torch.cat((largest_conf, largest_conf_loc), dim=1)) # shape = (1, 5)

        # remove largest element
        conf_des_inds = conf_des_inds[1:]

        if conf_des_inds.nelement() == 0:
            break

        # get iou, shape = (1, loc_des num)
        overlap = iou(centroids2corners(largest_conf_loc), loc_mm[conf_des_inds])
        # filter out overlapped boxes for box with largest conf, shape = (loc_des num)
        indicator = overlap.reshape((overlap.nelement())) <= iou_threshold

        conf_des_inds = conf_des_inds[indicator]

    return inferred_boxes

def tensor2cvrgbimg(img, to8bit=True):
    if to8bit:
        img = img * 255.
    return img.cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)

def toVisualizeRectangleRGBimg(img, locs, thickness=2, rgb=(255, 0, 0), verbose=False):
    """
    :param img: Tensor, shape = ()
    :param locs: Tensor, centered coordinates, shape = ()
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
    locs_mm = centroids2corners(locs).detach().numpy()

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

def toVisualizeRGBImg(img, locs, conf_indices, classes, confs=None, verbose=False):
    # convert (c, h, w) to (h, w, c)
    img = tensor2cvrgbimg(img)

    class_num = len(classes)
    box_num = locs.shape[0]
    assert box_num == conf_indices.shape[0], 'must be same boxes number'
    if confs is not None:
        if isinstance(confs, torch.Tensor):
            confs = confs.cpu().numpy()
        elif not isinstance(confs, np.ndarray):
            raise ValueError(
                'Invalid \'confs\' argment were passed. confs must be ndarray or Tensor, but got {}'.format(
                    type(confs).__name__))
        assert confs.ndim == 1 and confs.size == box_num, "Invalid confs"

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

        index = int(conf_indices[bnum].item())
        if index == -1:
            continue

        text = classes[index] + ':{:.2f}'.format(confs[bnum])

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