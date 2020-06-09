import abc
import sys
import torch
import math
from torch.utils.data import DataLoader
import numpy as np

sys.path.append('...')
from ssd_data.datasets.base import _DatasetBase
from ..models.base import ObjectDetectionModelBase
from .._utils import _check_ins
from ..core.boxes.utils import centroids2corners_numpy, corners2centroids_numpy, iou_numpy

# mAP: https://datascience.stackexchange.com/questions/25119/how-to-calculate-map-for-detection-task-for-the-pascal-voc-challenge
# https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
# https://towardsdatascience.com/implementation-of-mean-average-precision-map-with-non-maximum-suppression-f9311eb92522
class EvaluatorBase(object):
    def __init__(self, dataloader, iteration_interval=5000, verbose=True, **eval_kwargs):
        self.dataloader = _check_ins('dataloader', dataloader, DataLoader)
        self.dataset = _check_ins('dataloader.dataset', dataloader.dataset, _DatasetBase)
        self.iteration_interval = _check_ins('iteration_interval', iteration_interval, int)
        self.verbose = verbose
        self.device = None
        self.eval_kwargs = eval_kwargs

        self._result = {}

    @property
    def class_labels(self):
        return self.dataset.class_labels
    @property
    def class_nums(self):
        return self.dataset.class_nums

    def __call__(self, model):
        model = _check_ins('model', model, ObjectDetectionModelBase)
        self.device = model.device

        # targets_loc: list of ndarray, whose shape = (targets box num, 4)
        # targets_label: list of ndarray, whose shape = (targets box num, 1)
        targets_loc, targets_label = [], []
        # infers_loc: list of ndarray, whose shape = (inferred box num, 4)
        # infers_label: list of ndarray, whose shape = (inferred box num, 1)
        # infers_conf: list of ndarray, whose shape = (inferred box num, 1)
        infers_loc, infers_label, infers_conf = [], [], []


        if self.verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

        # predict
        #dataloader = iter(self.dataloader)
        #for i in range(10): # debug
        #    images, targets = next(dataloader)
        for i, (images, targets) in enumerate(self.dataloader):
            images = images.to(self.device)

            # infer is list of Tensor, shape = (box num, 6=(class index, confidence, cx, cy, w, h))
            infer = model.infer(images, visualize=False)

            targets_loc += [target.cpu().numpy()[:, :4] for target in targets]
            targets_label += [np.argmax(target.cpu().numpy()[:, 4:], axis=1) for target in targets]

            infers_loc += [inf.cpu().numpy()[:, 2:] for inf in infer]
            infers_label += [inf.cpu().numpy()[:, 0] for inf in infer]
            infers_conf += [inf.cpu().numpy()[:, 1] for inf in infer]
            """slower
            for target in targets:
                target = target.cpu().numpy()
                targets_loc += [target[:, :4]]
                targets_label += [np.argmax(target[:, 4:], axis=1)]

            for inf in infer:
                inf = inf.cpu().numpy()
                infers_loc += [inf[:, 2:]]
                infers_label += [inf[:, 0]]
                infers_conf += [inf[:, 1]]
            """

            if self.verbose:
                sys.stdout.write('\rinferring...\t{}/{}:\t{}%'.format(i+1, len(self.dataloader), int(100.*(i+1.)/len(self.dataloader))))
                sys.stdout.flush()

        if self.verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

        # debug
        #_save_debugfile(targets_loc, targets_label, infers_loc, infers_label, infers_conf)

        return self.eval(targets_loc, targets_label, infers_loc, infers_label, infers_conf, **self.eval_kwargs)

    @abc.abstractmethod
    def eval(self, targets_loc, targets_label, infers_loc, infers_label, infers_conf, **kwargs):
        """
        :param targets_loc: list of ndarray, whose shape = (targets box num, 4)
        :param targets_label: list of ndarray, whose shape = (targets box num,)
        :param infers_loc: list of ndarray, whose shape = (inferred box num, 4)
        :param infers_label: list of ndarray, whose shape = (inferred box num,)
        :param infers_conf: list of ndarray, whose shape = (inferred box num,)
        Note that above len(list) == image number
        :param kwargs:
        :return:
        """
        raise NotImplementedError()



class VOC2007Evaluator(EvaluatorBase):
    def __init__(self, dataset, iteration_interval=5000, verbose=True, iou_threshold=0.5):
        super().__init__(dataset, iteration_interval, verbose, iou_threshold=iou_threshold)

    def eval(self, targets_loc, targets_label, infers_loc, infers_label, infers_conf, **kwargs):
        """
        :param targets_loc: list of ndarray, whose shape = (targets box num, 4)
        :param targets_label: list of ndarray, whose shape = (targets box num,)
        :param infers_loc: list of ndarray, whose shape = (inferred box num, 4)
        :param infers_label: list of ndarray, whose shape = (inferred box num,)
        :param infers_conf: list of ndarray, whose shape = (inferred box num,)
        Note that above len(list) == image number
        :param iou_threshold: float, default is 0.5
        :return:
            AP: ndarray, shape = (dataset class nums), average precision for each class
        """
        iou_threshold = _check_ins('iou_threshold', kwargs.get('iou_threshold', 0.5), float)

        precisions, recalls = calc_PR(targets_loc, targets_label, infers_loc, infers_label,
                                      infers_conf, iou_threshold, self.class_nums)

        AP = {}
        for i, label in enumerate(self.class_labels):
            if recalls[i] is None:
                AP[label] = 0.
                continue

            # use 11 points
            ap = 0
            for pt in np.arange(0, 1.1, 0.1):
                mask = recalls[i] >= pt
                if mask.sum() == 0:
                    continue
                ap += np.max(precisions[i][mask])
            AP[label] = ap / 11.
        AP['mAP'] = np.mean(tuple(AP.values()))
        #print(AP)
        return AP


class VOCStyleEvaluator(EvaluatorBase):
    def __init__(self, dataset, iteration_interval=5000, verbose=True, iou_threshold=0.5):
        super().__init__(dataset, iteration_interval, verbose, iou_threshold=iou_threshold)

    def eval(self, targets_loc, targets_label, infers_loc, infers_label, infers_conf, **kwargs):
        """
        :param targets_loc: list of ndarray, whose shape = (targets box num, 4)
        :param targets_label: list of ndarray, whose shape = (targets box num,)
        :param infers_loc: list of ndarray, whose shape = (inferred box num, 4)
        :param infers_label: list of ndarray, whose shape = (inferred box num,)
        :param infers_conf: list of ndarray, whose shape = (inferred box num,)
        Note that above len(list) == image number
        :param iou_threshold: float, default is 0.5
        :return:
        """
        iou_threshold = _check_ins('iou_threshold', kwargs.get('iou_threshold', 0.5), float)

        precisions, recalls = calc_PR(targets_loc, targets_label, infers_loc, infers_label,
                                      infers_conf, iou_threshold, self.class_nums)

        raise NotImplementedError('Not Supported')


def calc_PR(targets_loc, targets_label, infers_loc, infers_label, infers_conf, iou_threshold, class_nums):
    """
    :param targets_loc: list of ndarray, whose shape = (targets box num, 4)
    :param targets_label: list of ndarray, whose shape = (targets box num,)
    :param infers_loc: list of ndarray, whose shape = (inferred box num, 4)
    :param infers_label: list of ndarray, whose shape = (inferred box num,)
    :param infers_conf: list of ndarray, whose shape = (inferred box num,)
    Note that above len(list) == image number
    :param iou_threshold: float, default is 0.5
    :param class_nums: int
    :return:
        precisions: list of ndarray, whose shape = (inferred box num)
        recalls: list of ndarray, whose shape = (inferred box num)
    """

    # list of list of ndarray, whose shape = (inferred box)
    # ref: https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
    confidence_per_class = [[] for _ in range(class_nums)]
    # list of list of bool ndarray, whose shape = (inferred box). correct consists of tp(=True) or fp(=False)
    correct_per_class = [[] for _ in range(class_nums)]

    """
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    TP = number of detections with IoU>0.5
    FP = number of detections with IoU<=0.5 or detected more than once
    FN = number of objects that not detected or detected with IoU<=0.5
    """

    # calculate iou for each label
    for t_locs, t_labels, i_locs, i_labels, i_confs in zip(targets_loc, targets_label, infers_loc, infers_label, infers_conf):
        # above t_* are ndarray, whose shape = (target box num, ?)
        # above i_* are ndarray, whose shape = (inferred box num, ?)

        # check existence inferred box
        if np.isnan(i_confs).any():
            continue

        # sort inferred box, conf and label with descending conf
        i_rank = np.argsort(i_confs)[::-1]
        i_locs = i_locs[i_rank]
        i_labels = i_labels[i_rank]
        i_confs = i_confs[i_rank]

        # shape = (targets box num, inferred box num)
        overlap = iou_numpy(centroids2corners_numpy(t_locs), centroids2corners_numpy(i_locs))

        # shape = (inferred box num,)
        max_overlap_to_target, max_target_box_ind = np.max(overlap, axis=0), np.argmax(overlap, axis=0)

        # store detected target index
        detected_t_ind = []
        for inferred_box_ind, (i_label, i_conf) in enumerate(zip(i_labels.astype(np.int), i_confs)):
            # overlap (IoU) > threshold and not duplicated box -> true positive
            t_ind = max_target_box_ind[inferred_box_ind]
            if max_overlap_to_target[inferred_box_ind] > iou_threshold and t_ind not in detected_t_ind:
                correct_per_class[i_label] += [True]
                confidence_per_class[i_label] += [i_conf]

                detected_t_ind += [t_ind]

            else: # false positive
                correct_per_class[i_label] += [False]
                confidence_per_class[i_label] += [i_conf]


    precisions, recalls = [], []
    for i in range(class_nums):
        if len(confidence_per_class[i]) == 0:
            precisions += [None]
            recalls += [None]
            continue

        # concatenate all ndarray -> flatten ndarray
        # shape = (inferred box num)
        confs = np.array(confidence_per_class[i])
        corrects = np.array(correct_per_class[i])

        # sort with descending order
        rank = confs.argsort()[::-1]
        corrects = corrects[rank]


        # calculate TP
        tp = np.cumsum(corrects)
        fp = np.cumsum(~corrects)

        # if divide 0, nan will be passed
        # prec = tp / all detection, rec = tp / all ground truth
        # ref > https://github.com/rafaelpadilla/Object-Detection-Metrics#different-competitions-different-metrics
        # ref > https://github.com/Cartucho/mAP
        prec = tp / np.maximum(tp + fp, 1e-15)
        rec = tp / tp[-1]

        precisions += [np.nan_to_num(prec)]
        recalls += [np.nan_to_num(rec)]

    return precisions, recalls
    """
    # list of list of ndarray, whose shape = (inferred box)
    overlap_per_class = [[] for _ in range(class_nums)]
    match_per_class = [[] for _ in range(class_nums)]
    conf_per_class = [[] for _ in range(class_nums)]
    # calculate iou for each label
    for t_locs, t_labels, i_locs, i_labels, i_conf in zip(targets_loc, targets_label, infers_loc, infers_label, infers_conf):
        # above 5 values are ndarray, whose shape = (inferred box num, ?)

        # shape = (targets box num, inferred box num)
        overlap = iou_numpy(centroids2corners_numpy(t_locs), centroids2corners_numpy(i_locs))

        # shape = (targets box num, inferred box num)
        match = np.expand_dims(t_labels, axis=1) == np.expand_dims(i_labels, axis=0)
        
        for target_box_ind, class_ind in enumerate(t_labels):
            overlap_per_class[class_ind] += [overlap[target_box_ind]]
            match_per_class[class_ind] += [match[target_box_ind]]
            conf_per_class[class_ind] += [i_conf]

    precisions, recalls = [], []
    for i in range(class_nums):
        if len(overlap_per_class[i]) == 0:
            precisions += [None]
            recalls += [None]
            continue

        # concatenate all ndarray -> flatten ndarray
        # shape = (inferred box num)
        overlap = np.concatenate(overlap_per_class[i])
        match = np.concatenate(match_per_class[i])
        conf = np.concatenate(conf_per_class[i])

        # sort with descending order
        rank = conf.argsort()[::-1]
        overlap = overlap[rank]
        target_true = match[rank]
        pred_true = overlap > iou_threshold

        # calculate TP, FP and FN
        tp = np.cumsum(target_true & pred_true)
        #fp = np.cumsum(np.logical_not(target_true) & pred_true)
        #fn = np.cumsum(target_true & np.logical_not(pred_true))

        # if divide 0, nan will be passed
        # prec = tp / all detection, rec = tp / all ground truth
        # ref > https://github.com/rafaelpadilla/Object-Detection-Metrics#different-competitions-different-metrics
        # ref > https://github.com/Cartucho/mAP
        prec = tp / np.arange(1, tp.size + 1)
        rec = tp / tp[-1]

        precisions += [np.nan_to_num(prec)]
        recalls += [np.nan_to_num(rec)]
    
    return precisions, recalls
    """

import os

def _save_debugfile(targets_loc, targets_label, infers_loc, infers_label, infers_conf):
    """
    :param targets_loc: list of ndarray, whose shape = (targets box num, 4)
    :param targets_label: list of ndarray, whose shape = (targets box num,)
    :param infers_loc: list of ndarray, whose shape = (inferred box num, 4)
    :param infers_label: list of ndarray, whose shape = (inferred box num,)
    :param infers_conf: list of ndarray, whose shape = (inferred box num,)
    Note that above len(list) == image number
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.debug')

    box_nums = []
    for t, i in zip(targets_label, infers_label):
        box_nums += [[t.shape[0], i.shape[0]]]

    with open(os.path.join(path, 'targets_loc.npy'), 'wb') as f:
        np.save(f, np.concatenate(targets_loc, axis=0))
    with open(os.path.join(path, 'targets_label.npy'), 'wb') as f:
        np.save(f, np.concatenate(targets_label, axis=0))
    with open(os.path.join(path, 'infers_loc.npy'), 'wb') as f:
        np.save(f, np.concatenate(infers_loc, axis=0))
    with open(os.path.join(path, 'infers_label.npy'), 'wb') as f:
        np.save(f, np.concatenate(infers_label, axis=0))
    with open(os.path.join(path, 'infers_conf.npy'), 'wb') as f:
        np.save(f, np.concatenate(infers_conf, axis=0))

    with open(os.path.join(path, 'box_nums.npy'), 'wb') as f:
        np.save(f, np.array(box_nums))

def _load_debugfile():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.debug')

    with open(os.path.join(path, 'box_nums.npy'), 'rb') as f:
        box_nums = np.load(f)

    with open(os.path.join(path, 'targets_loc.npy'), 'rb') as f:
        t_loc = np.load(f)
    with open(os.path.join(path, 'targets_label.npy'), 'rb') as f:
        t_label = np.load(f)
    with open(os.path.join(path, 'infers_loc.npy'), 'rb') as f:
        i_loc = np.load(f)
    with open(os.path.join(path, 'infers_label.npy'), 'rb') as f:
        i_label = np.load(f)
    with open(os.path.join(path, 'infers_conf.npy'), 'rb') as f:
        i_conf = np.load(f)

    t, i = 0, 0
    targets_loc, targets_label = [], []
    infers_loc, infers_label, infers_conf = [], [], []
    for b_num in box_nums:
        t_bnum, i_bnum = b_num

        targets_loc += [t_loc[t:t+t_bnum]]
        targets_label += [t_label[t:t + t_bnum]]

        infers_loc += [i_loc[i:i+i_bnum]]
        infers_label += [i_label[i:i+i_bnum]]
        infers_conf += [i_conf[i:i+i_bnum]]

        t += t_bnum
        i += i_bnum

    assert len(targets_label) == len(infers_loc)
    assert t == t_loc.shape[0]
    assert i == i_loc.shape[0]

    return targets_loc, targets_label, infers_loc, infers_label, infers_conf
