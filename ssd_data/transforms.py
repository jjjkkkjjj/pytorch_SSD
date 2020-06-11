import numpy as np
import torch
import cv2
import logging

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes, labels, flags, *args):
        for t in self.transforms:
            transformed = t(img, bboxes, labels, flags, *args)
            img, bboxes, labels, flags = transformed[:4]
            args = transformed[4:]
        return img, bboxes, labels, flags, args

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
    """
    Note that convert ndarray to tensor and [0-255] to [0-1]
    """
    def __call__(self, img, *args):
        # convert ndarray into Tensor
        # transpose img's tensor (h, w, c) to pytorch's format (c, h, w). (num, c, h, w)
        img = np.transpose(img, (2, 0, 1))
        return (torch.from_numpy(img).float() / 255., *args)

class Resize(object):
    def __init__(self, size):
        """
        :param size: 2d-array-like, (height, width)
        """
        self._size = size

    def __call__(self, img, *args):
        return (cv2.resize(img, self._size), *args)


class Normalize(object):
    #def __init__(self, rgb_means=(103.939, 116.779, 123.68), rgb_stds=(1.0, 1.0, 1.0)):
    def __init__(self, rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225)):
        self.means = np.array(rgb_means, dtype=np.float32).reshape((-1, 1, 1))
        if np.any(np.abs(self.means) > 1):
            logging.warning("In general, mean value should be less than 1 because img's range is [0-1]")

        self.stds = np.array(rgb_stds, dtype=np.float32).reshape((-1, 1, 1))

    def __call__(self, img, *args):
        if isinstance(img, torch.Tensor):
            return ((img.float() - torch.from_numpy(self.means)) / torch.from_numpy(self.stds), *args)
        else:
            return ((img.astype(np.float32) - self.means) / self.stds, *args)
