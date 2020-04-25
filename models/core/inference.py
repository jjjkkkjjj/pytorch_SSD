from .boxes import iou, center2minmax, minmax2center, pred_loc_converter

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

        self.softmax = F.softmax

    def forward(self, predicts, dboxes):
        """
        :param predicts: localization and confidence Tensor, shape is (batch, total_dbox_num, 4+class_nums)
        :param dboxes: Tensor, default boxes Tensor whose shape is (total_dbox_nums, 4)`
        :return:
        """

        """
        pred_loc, inf_cand_loc: shape = (batch number, default boxes number, 4)
        pred_conf: shape = (batch number, default boxes number, class number)
        """
        pred_loc, pred_conf = predicts[:, :, :4], predicts[:, :, 4:]
        inf_cand_loc, inf_cand_conf = pred_loc_converter(pred_loc, dboxes), self.softmax(pred_conf, dim=2)

        batch_num = predicts.shape[0]
        class_num = pred_conf.shape[2]

        ret_boxes = []
        for b in range(batch_num):
            ret_box = []
            inf_conf = inf_cand_conf[b] # shape = (default boxes number, class number)
            inf_loc = inf_cand_loc[b]
            for c in range(class_num - 1): # last index means background
                # filter out less than threshold
                indicator = inf_conf[:, c] > self.conf_threshold
                conf = inf_conf[indicator, c] # shape = (filtered default boxes num)
                if conf.nelement() == 0:
                    continue
                loc = inf_loc[indicator, :] # shape = (filtered default boxes num, 4)

                # list of Tensor, shape = (1, 4=(cx, cy, w, h))
                inferred_boxes = non_maximum_suppression(conf, loc, self.iou_threshold, self.topk)
                if len(inferred_boxes) == 0:
                    continue
                else:
                    # shape = (inferred boxes num, 4)
                    inferred_boxes = torch.cat(inferred_boxes, dim=0)

                    # append class flag
                    # shape = (inferred boxes num, 1)
                    flag = np.broadcast_to([c], shape=(len(inferred_boxes), 1))
                    flag = torch.from_numpy(flag).float()

                    # shape = (inferred box num, 5=(class index, cx, cy, w, h))
                    ret_box += [torch.cat((flag, inferred_boxes), dim=1)]

            ret_boxes += [torch.cat(ret_box, dim=0)]

        # list of tensor, shape = (box num, 5=(class index, cx, cy, w, h))
        return ret_boxes


def non_maximum_suppression(conf, loc, iou_threshold=0.45, topk=200):
    """
    :param conf: tensor, shape = (filtered default boxes num)
    :param loc: tensor, shape = (filtered default boxes num, 4)
    Note that filtered default boxes number must be more than 1
    :param iou_threshold: int
    :return: inferred_boxes: list of inferred boxes(Tensor). inferred boxes' Tensor shape = (inferred boxes number, 4)
    """
    # sort confidence and default boxes with descending order
    c, conf_des_inds = conf.sort(dim=0, descending=True)
    # get topk indices
    conf_des_inds = conf_des_inds[:topk]
    # converted into minmax coordinates
    loc_mm = center2minmax(loc)

    inferred_boxes = []
    while conf_des_inds.nelement() > 0:
        largest_conf_index = conf_des_inds[0]
        largest_conf_loc = loc[largest_conf_index, :].unsqueeze(0)  # shape = (1, 4=(xmin, ymin, xmax, ymax))
        # append to result
        inferred_boxes.append(largest_conf_loc)

        # remove largest element
        conf_des_inds = conf_des_inds[1:]

        if conf_des_inds.nelement() == 0:
            break

        # get iou, shape = (1, loc_des num)
        overlap = iou(center2minmax(largest_conf_loc), loc_mm[conf_des_inds])
        # filter out overlapped boxes for box with largest conf, shape = (loc_des num)
        indicator = overlap.reshape((overlap.nelement())) <= iou_threshold

        conf_des_inds = conf_des_inds[indicator]

    return inferred_boxes

def tensor2cvimg(img):
    return img.numpy().transpose((1, 2, 0)).astype(np.uint8)

def toVisualizeRectangleimg(img, locs, thickness=2, rgb=(255, 0, 0), verbose=False):
    # convert (c, h, w) to (h, w, c)
    img = tensor2cvimg(img)

    # print(locs)
    locs_mm = center2minmax(locs).numpy()

    h, w, c = img.shape
    locs_mm[:, ::2] *= w
    locs_mm[:, 1::2] *= h
    locs_mm = np.clip(locs_mm, 0, w).astype(np.int)

    if verbose:
        print(locs_mm)
    for bnum in range(locs_mm.shape[0]):
        topleft = locs_mm[bnum, :2]
        bottomright = locs_mm[bnum, 2:]

        if verbose:
            print(tuple(topleft), tuple(bottomright))

        cv2.rectangle(img, tuple(topleft), tuple(bottomright), rgb, thickness)

    return img

def toVisualizeImg(img, locs, conf_indices, classes, verbose=False):
    # convert (c, h, w) to (h, w, c)
    img = tensor2cvimg(img)

    class_num = len(classes)
    box_num = locs.shape[0]
    assert box_num == conf_indices.shape[0], 'must be same boxes number'

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
    locs_mm = center2minmax(locs).numpy()
    locs_mm[:, ::2] *= w
    locs_mm[:, 1::2] *= h
    locs_mm = locs_mm
    locs_mm = np.clip(locs_mm, 0, w).astype(np.int)

    if verbose:
        print(locs_mm)
    for bnum in range(box_num):# box num
        topleft = locs_mm[bnum, :2]
        bottomright = locs_mm[bnum, 2:]

        if verbose:
            print(tuple(topleft), tuple(bottomright))

        index = int(conf_indices[bnum].item())

        labelSize = cv2.getTextSize(classes[index], cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)
        _x2 = topleft[0] + labelSize[0][0]
        _y2 = topleft[1] - int(labelSize[0][1])

        topleft = tuple(topleft)
        bottomright = tuple(bottomright)
        rgb = tuple(rgbs[index, 0].tolist())

        cv2.rectangle(img, topleft, (_x2, _y2), rgb, cv2.FILLED)
        cv2.putText(img, classes[index], topleft, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(img, topleft, bottomright, rgb, thickness)

    return img