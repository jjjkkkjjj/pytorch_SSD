from .utils import *

import torch
import cv2

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes, labels, flags):
        for t in self.transforms:
            img, bboxes, labels, flags = t(img, bboxes, labels, flags)
        return img, bboxes, labels, flags

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

"""
bellow classes are consisted of
    :param img: Tensor
    :param bboxes: ndarray of bboxes
    :param labels: ndarray of bboxes' indices
    :param flags: list of flag's dict
    :return: Tensor of img, ndarray of bboxes, ndarray of labels, dict of flags
"""

class ToTensor(object):
    def __call__(self, img, bboxes, labels, flags):
        # convert ndarray into Tensor
        return torch.from_numpy(img), torch.from_numpy(bboxes), torch.from_numpy(labels), flags

class Resize(object):
    def __init__(self, size):
        """
        :param size: 2d-array-like, (height, width)
        """
        self._size = size

    def __call__(self, img, bboxes, labels, flags):
        return cv2.resize(img, self._size), bboxes, labels, flags

class Normalize(object):

    def __call__(self, img, bboxes, labels, flags):
        height, width, channel = img.shape

        # normalize
        # bbox = [xmin, ymin, xmax, ymax]
        # [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
        bboxes[:, 0::2] /= float(width)
        bboxes[:, 1::2] /= float(height)


        return img, bboxes, labels, flags

class Centered(object):
    def __call__(self, img, bboxes, labels, flags):
        # bbox = [xmin, ymin, xmax, ymax]
        bboxes = np.concatenate(((bboxes[:, 2:] + bboxes[:, :2]) / 2,
                                 (bboxes[:, 2:] - bboxes[:, :2])), axis=1)

        return img, bboxes, labels, flags

class Ignore(object):
    def __init__(self, ignore_difficult=True, ignore_partial=False):
        """
        :param ignore_difficult: if true, difficult bbox will be ignored, otherwise the one will be kept
        :param ignore_partial: if true, an object being visible partially will be ignored
        """
        self._ignore_difficult = ignore_difficult
        self._ignore_partial = ignore_partial

    def __call__(self, img, bboxes, labels, flags):
        ret_bboxes = []
        ret_labels = []
        ret_flags = []

        for bbox, label, flag in zip(bboxes, labels, flags):
            if self._ignore_difficult and flag['difficult']:
                continue
            if self._ignore_partial and flag['partial']:
                continue

            # normalize
            # bbox = [xmin, ymin, xmax, ymax]
            ret_bboxes += [bbox]
            ret_labels += [label]
            ret_flags += [flag]

        ret_bboxes = np.array(ret_bboxes, dtype=np.float32)
        ret_labels = np.array(ret_labels, dtype=np.float32)

        return img, ret_bboxes, ret_labels, ret_flags

class OneHot(object):
    def __init__(self, class_nums):
        self._class_nums = class_nums

    def __call__(self, img, bboxes, labels, flags):
        if labels.ndim != 1:
            raise ValueError('labels might have been already one-hotted or be invalid shape')

        labels = one_hot_encode(labels.astype(np.int), self._class_nums)
        labels = np.array(labels, dtype=np.float32)

        return img, bboxes, labels, flags
