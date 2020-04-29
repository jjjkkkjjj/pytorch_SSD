import numpy as np
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
    def __call__(self, img, *args):
        # convert ndarray into Tensor
        # transpose img's tensor (h, w, c) to pytorch's format (c, h, w). (num, c, h, w)
        img = np.transpose(img, (2, 0, 1))
        return (torch.from_numpy(img).float(), *args)

class Resize(object):
    def __init__(self, size):
        """
        :param size: 2d-array-like, (height, width)
        """
        self._size = size

    def __call__(self, img, *args):
        return (cv2.resize(img, self._size), *args)


class Normalize(object):
    def __init__(self, bgr_means=(123.68, 116.779, 103.939), bgr_stds=(1.0, 1.0, 1.0)):
        self.means = np.array(bgr_means, dtype=np.float32)
        self.stds = np.array(bgr_stds, dtype=np.float32)

    def __call__(self, img, *args):

        return ((img.astype(np.float32) - self.means) / self.stds, *args)
