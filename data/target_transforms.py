import numpy as np
import torch

from ._utils import _one_hot_encode

class Compose(object):
    def __init__(self, target_transforms):
        self.target_transforms = target_transforms

    def __call__(self, bboxes, labels, flags):
        for t in self.target_transforms:
            bboxes, labels, flags = t(bboxes, labels, flags)
        return bboxes, labels, flags

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.target_transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToTensor(object):
    def __call__(self, bboxes, labels, flags):
        return torch.from_numpy(bboxes), torch.from_numpy(labels), flags

class ToCentroids(object):
    def __call__(self, bboxes, labels, flags):
        # bbox = [xmin, ymin, xmax, ymax]
        bboxes = np.concatenate(((bboxes[:, 2:] + bboxes[:, :2]) / 2,
                                 (bboxes[:, 2:] - bboxes[:, :2])), axis=1)

        return bboxes, labels, flags

class ToMinMax(object):
    def __call__(self, bboxes, labels, flags):
        # bbox = [cx, cy, w, h]
        bboxes = np.concatenate((bboxes[:, :2] - bboxes[:, 2:]/2,
                                 bboxes[:, :2] + bboxes[:, 2:]/2), axis=1)

        return bboxes, labels, flags

class Ignore(object):
    def __init__(self, difficult=True, **kwargs):
        """
        :param difficult: if true, difficult bbox will be ignored, otherwise the relu_one will be kept
        :param kwargs: if true, specific keyword will be ignored
        """
        self.difficult = difficult
        self.kwargs = kwargs
        if len(kwargs) > 0:
            import logging
            logging.warning('Unsupported arguments: {}'.format(self.kwargs.keys()))

    def __call__(self, bboxes, labels, flags):
        ret_bboxes = []
        ret_labels = []
        ret_flags = []

        for bbox, label, flag in zip(bboxes, labels, flags):
            if self.difficult and flag['difficult']:
                continue
            """
            isIgnore = False
            for key, value in self.kwargs.items():
                if value and key in flag and flag[key]:
                    isIgnore = True
                    break
            if isIgnore:
                continue
            #if self._ignore_partial and flag['partial']:
            #    continue
            """
            # normalize
            # bbox = [xmin, ymin, xmax, ymax]
            ret_bboxes += [bbox]
            ret_labels += [label]
            ret_flags += [flag]

        ret_bboxes = np.array(ret_bboxes, dtype=np.float32)
        ret_labels = np.array(ret_labels, dtype=np.float32)

        return ret_bboxes, ret_labels, ret_flags

class OneHot(object):
    def __init__(self, class_nums, add_background=True):
        self._class_nums = class_nums
        self._add_background = add_background
        if add_background:
            self._class_nums += 1

    def __call__(self, bboxes, labels, flags):
        if labels.ndim != 1:
            raise ValueError('labels might have been already relu_one-hotted or be invalid shape')

        labels = _one_hot_encode(labels.astype(np.int), self._class_nums)
        labels = np.array(labels, dtype=np.float32)

        return bboxes, labels, flags
