from ..utils import *
from .base import _voc_classes

class Normalize(object):
    def __init__(self, normalize=True, one_hot=True, ignore_difficult=True, ignore_partial=False):
        """
        :param normalize: bool, if true, bbox will be normalized 0 to 1 by own's width and height
        :param one_hot: bool, if true, label will be converted to one-hot vector, otherwise label's index will be kept
        :param ignore_difficult: if true, difficult bbox will be ignored, otherwise the one will be kept
        :param ignore_partial: if true, an object being visible partially will be ignored
        """
        self._normalize = normalize
        self._one_hot = one_hot
        self._ignore_difficult = ignore_difficult
        self._ignore_partial = ignore_partial

    def __call__(self, img, bboxes, labels, flags):
        """
        :param img: Tensor
        :param bboxes: list of bboxes
        :param labels: list of bboxes' indices
        :param flags: list of flag's dict
        :return: Tensor of img, ndarray of bboxes, ndarray of labels, dict of flags
        """
        ret_bboxes = []
        ret_labels = []
        height, width, channel = img.shape
        for bbox, label, flag in zip(bboxes, labels, flags):
            if self._ignore_difficult and flag['difficult']:
                continue
            if self._ignore_partial and flag['partial']:
                continue

            # normalize
            # bbox = [xmin, ymin, xmax, ymax]
            ret_bboxes += [bbox]
            ret_labels += [label]

        ret_bboxes = np.array(ret_bboxes, dtype=np.float32)
        if self._normalize:
            #[bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
            ret_bboxes[:, 0::2] /= float(width)
            ret_bboxes[:, 1::2] /= float(height)

        if self._one_hot:
            ret_labels = one_hot_encode(ret_labels, len(_voc_classes))
        ret_labels = np.array(ret_labels, dtype=np.float32)

        return img, ret_bboxes, ret_labels, flags

