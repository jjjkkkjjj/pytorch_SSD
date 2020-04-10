import torch
from torch.utils.data import Dataset
import cv2, glob, os
import numpy as np
from xml.etree import ElementTree as ET

from .utils import *


#========== transform ===========#
class VOCDatasetTransform(object):
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


#========== dataset ===========#
"""
ref > https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:

__len__ so that len(dataset) returns the size of the dataset.
__getitem__ to support the indexing such that dataset[i] can be used to get ith sample

"""

_voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']

class VOCDataset(Dataset):
    def __init__(self, voc_dir, transform=None):
        self.transform = transform
        self._voc_dir = voc_dir
        self._annopaths = glob.glob(os.path.join(self._voc_dir, 'Annotations', '*.xml'))

    def _jpgpath(self, filename):
        """
        :param filename: path containing .jpg
        :return: path of jpg
        """
        return os.path.join(self._voc_dir, 'JPEGImages', filename)

    def __getitem__(self, index):
        """
        :param index: int
        :return:
            rgb image(Tensor), list of bboxes, list of bboxes' label index
        """
        img = self._get_image(index)
        bboxes, linds, flags = self._get_bbox_lind(index)

        if self.transform:
            img, bboxes, linds, flags = self.transform(img, bboxes, linds, flags)

        return img, bboxes, linds

    def __len__(self):
        return len(self._annopaths)

    """
    Detail of contents in voc > https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5

    VOC bounding box (xmin, ymin, xmax, ymax)
    """
    def _get_image(self, index):
        """
        :param index: int
        :return:
            rgb image(Tensor)
        """
        root = ET.parse(self._annopaths[index]).getroot()
        img = cv2.imread(self._jpgpath(get_xml_et_value(root, 'filename')))
        # pytorch's image order is rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # convert ndarray into Tensor
        return torch.from_numpy(img)

    def _get_bbox_lind(self, index):
        """
        :param index: int
        :return:
            list of bboxes, list of bboxes' label index, list of flags([difficult, truncated])
        """
        linds = []
        bboxes = []
        flags = []

        root = ET.parse(self._annopaths[index]).getroot()
        for obj in root.iter('object'):
            linds.append(_voc_classes.index(get_xml_et_value(obj, 'name')))

            bndbox = obj.find('bndbox')

            # bbox = [xmin, ymin, xmax, ymax]
            bboxes.append([get_xml_et_value(bndbox, 'xmin', int), get_xml_et_value(bndbox, 'ymin', int), get_xml_et_value(bndbox, 'xmax', int), get_xml_et_value(bndbox, 'ymax', int)])

            flags.append({'difficult': get_xml_et_value(obj, 'difficult', int) == 1,
                          'partial': get_xml_et_value(obj, 'truncated', int) == 1})

        return bboxes, linds, flags


