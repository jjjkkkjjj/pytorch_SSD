import abc
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np

sys.path.append('...')
from data.datasets.base import _DatasetBase
from ..models.base import ObjectDetectionModelBase
from .._utils import _check_ins
from ..core.boxes.utils import centroids2corners_numpy, corners2centroids_numpy, iou_numpy

# mAP: https://datascience.stackexchange.com/questions/25119/how-to-calculate-map-for-detection-task-for-the-pascal-voc-challenge
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
        infers_loc, infers_label = [], []

        if self.verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

        # predict
        #for i in range(2000): # debug
        for i, (images, targets) in enumerate(self.dataloader):
            images = images.to(self.device)

            # infer is list of Tensor, shape = (box num, 5=(class index, cx, cy, w, h))
            infer = model.infer(images, visualize=False)

            targets_loc += [target[:, :4].cpu().numpy() for target in targets]
            targets_label += [np.argmax(target[:, 4:].cpu().numpy(), axis=1) for target in targets]

            infers_loc += [inf[:, 1:].cpu().numpy() for inf in infer]
            infers_label += [inf[:, 0].cpu().numpy().astype(np.int) for inf in infer]

            if self.verbose:
                sys.stdout.write('\rinferring...\t{}/{}:\t{}%'.format(i+1, len(self.dataloader), int(100.*(i+1.)/len(self.dataloader))))
                sys.stdout.flush()

        if self.verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

        return self.eval(targets_loc, targets_label, infers_loc, infers_label, **self.eval_kwargs)

    @abc.abstractmethod
    def eval(self, targets_loc, targets_label, infers_loc, infers_label, **kwargs):
        """
        :param targets_loc: list of ndarray, whose shape = (targets box num, 4)
        :param targets_label: list of ndarray, whose shape = (targets box num, 1)
        :param infers_loc: list of ndarray, whose shape = (inferred box num, 4)
        :param infers_label: list of ndarray, whose shape = (inferred box num, 1)
        :param kwargs:
        :return:
        """
        raise NotImplementedError()



class VOC2007Evaluator(EvaluatorBase):
    def __init__(self, dataset, iteration_interval=5000, verbose=True, iou_threshold=0.5):
        super().__init__(dataset, iteration_interval, verbose, iou_threshold=iou_threshold)

    def eval(self, targets_loc, targets_label, infers_loc, infers_label, **kwargs):
        """
        :param targets_loc: list of ndarray, whose shape = (targets box num, 4)
        :param targets_label: list of ndarray, whose shape = (targets box num, 1)
        :param infers_loc: list of ndarray, whose shape = (inferred box num, 4)
        :param infers_label: list of ndarray, whose shape = (inferred box num, 1)
        :param iou_threshold: float, default is 0.5
        :return:
            AP: ndarray, shape = (dataset class nums), average precision for each class
        """
        iou_threshold = _check_ins('iou_threshold', kwargs.get('iou_threshold', 0.5), float)

        precisions, recalls = calc_PR(targets_loc, targets_label, infers_loc, infers_label,
                                      iou_threshold, self.class_nums)

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

    def eval(self, targets_loc, targets_label, infers_loc, infers_label, **kwargs):
        """
        :param targets_loc: list of ndarray, whose shape = (targets box num, 4)
        :param targets_label: list of ndarray, whose shape = (targets box num, 1)
        :param infers_loc: list of ndarray, whose shape = (inferred box num, 4)
        :param infers_label: list of ndarray, whose shape = (inferred box num, 1)
        :param iou_threshold: float, default is 0.5
        :return:
        """
        iou_threshold = _check_ins('iou_threshold', kwargs.get('iou_threshold', 0.5), float)

        precisions, recalls = calc_PR(targets_loc, targets_label, infers_loc, infers_label,
                                      iou_threshold, self.class_nums)


def calc_PR(targets_loc, targets_label, infers_loc, infers_label, iou_threshold, class_nums):
    """
    :param targets_loc: list of ndarray, whose shape = (targets box num, 4)
    :param targets_label: list of ndarray, whose shape = (targets box num, 1)
    :param infers_loc: list of ndarray, whose shape = (inferred box num, 4)
    :param infers_label: list of ndarray, whose shape = (inferred box num, 1)
    :param iou_threshold: float, default is 0.5
    :param class_nums: int
    :return:
        precisions: list of ndarray, whose shape = (inferred box num)
        recalls: list of ndarray, whose shape = (inferred box num)
    """
    # list of list of ndarray
    overlap_per_class = [[] for _ in range(class_nums)]
    match_per_class = [[] for _ in range(class_nums)]
    # calculate iou for each label
    for t_locs, t_labels, i_locs, i_labels in zip(targets_loc, targets_label, infers_loc, infers_label):
        # above 4 values are ndarray

        # shape = (targets box num, inferred box num)
        overlap = iou_numpy(centroids2corners_numpy(t_locs), centroids2corners_numpy(i_locs))

        # shape = (targets box num, inferred box num)
        match = np.expand_dims(t_labels, axis=1) == np.expand_dims(i_labels, axis=0)

        for target_box_ind, class_ind in enumerate(t_labels):
            overlap_per_class[class_ind] += [overlap[target_box_ind]]
            match_per_class[class_ind] += [match[target_box_ind]]

    precisions, recalls = [], []
    for i in range(class_nums):
        # concatenate all ndarray -> flatten ndarray
        # shape = (inferred box num)
        if len(overlap_per_class[i]) == 0:
            precisions += [None]
            recalls += [None]
            continue

        overlap = np.concatenate(overlap_per_class[i])
        match = np.concatenate(match_per_class[i])

        # sort with descending order
        rank = overlap.argsort()[::-1]
        overlap = overlap[rank]
        target_true = match[rank]

        # calculate TP, FP and FN
        pred_true = overlap > iou_threshold
        tp = np.cumsum(target_true & pred_true)
        fp = np.cumsum(np.logical_not(target_true) & pred_true)
        fn = np.cumsum(target_true & np.logical_not(pred_true))

        # if divide 0, nan will be passed
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        precisions += [np.nan_to_num(prec)]
        recalls += [np.nan_to_num(rec)]

    return precisions, recalls