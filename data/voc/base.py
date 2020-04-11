import torch
from torch.utils.data import Dataset
import cv2, glob, os
import numpy as np
from xml.etree import ElementTree as ET

from ..utils import *


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

class VOCBaseDataset(Dataset):
    def __init__(self, voc_dir, transform=None):
        self.transform = transform
        self.class_nums = len(_voc_classes)
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

