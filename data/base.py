import torch
from torch.utils.data import Dataset
import cv2, glob, os
import numpy as np
from xml.etree import ElementTree as ET

from .utils import _get_xml_et_value

"""
ref > https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:

__len__ so that len(dataset) returns the size of the dataset.
__getitem__ to support the indexing such that dataset[i] can be used to get ith sample

"""
VOC_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']
VOC_class_nums = len(VOC_classes) + 1

class VOCBaseDataset(Dataset):
    class_nums = len(VOC_classes) + 1
    def __init__(self, voc_dir, focus, transform=None, target_transform=None, augmentation=None):
        self.transform = transform
        self.target_transform = target_transform
        self.augmentation = augmentation

        self._voc_dir = voc_dir
        self._focus = focus
        layouttxt_path = os.path.join(self._voc_dir, 'ImageSets', 'Main', self._focus + '.txt')
        if os.path.exists(layouttxt_path):
            with open(layouttxt_path, 'r') as f:
                filenames = f.read().splitlines()
                filenames = [filename.split()[0] for filename in filenames]
                self._annopaths = [os.path.join(self._voc_dir, 'Annotations', '{}.xml'.format(filename)) for filename in filenames]
        else:
            raise FileNotFoundError('layout: {} was invalid arguments'.format(focus))

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
            img : rgb image(Tensor or ndarray)
            gt : Tensor or ndarray of bboxes and labels [box, label]
            = [xmin, ymin, xmamx, ymax, label index(or relu_one-hotted label)]
            or
            = [cx, cy, w, h, label index(or relu_one-hotted label)]
        """
        img = self._get_image(index)
        bboxes, linds, flags = self._get_bbox_lind(index)

        # To Percent mode
        height, width, channel = img.shape
        # bbox = [xmin, ymin, xmax, ymax]
        # [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
        bboxes[:, 0::2] /= float(width)
        bboxes[:, 1::2] /= float(height)

        if self.augmentation:
            img, bboxes, linds, flags = self.augmentation(img, bboxes, linds, flags)

        if self.transform:
            img, bboxes, linds, flags = self.transform(img, bboxes, linds, flags)

        if self.target_transform:
            bboxes, linds, flags = self.target_transform(bboxes, linds, flags)

        # concatenate bboxes and linds
        if isinstance(bboxes, torch.Tensor) and isinstance(linds, torch.Tensor):
            if linds.ndim == 1:
                linds = linds.unsqueeze(1)
            gt = torch.cat((bboxes, linds), dim=1)
        else:
            if linds.ndim == 1:
                linds = linds[:, np.newaxis]
            gt = np.concatenate((bboxes, linds), axis=1)


        return img, gt

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
        img = cv2.imread(self._jpgpath(_get_xml_et_value(root, 'filename')))
        # pytorch's image order is rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)

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
            linds.append(VOC_classes.index(_get_xml_et_value(obj, 'name')))

            bndbox = obj.find('bndbox')

            # bbox = [xmin, ymin, xmax, ymax]
            bboxes.append([_get_xml_et_value(bndbox, 'xmin', int), _get_xml_et_value(bndbox, 'ymin', int), _get_xml_et_value(bndbox, 'xmax', int), _get_xml_et_value(bndbox, 'ymax', int)])

            flags.append({'difficult': _get_xml_et_value(obj, 'difficult', int) == 1})#,
                          #'partial': _get_xml_et_value(obj, 'truncated', int) == 1})

        return np.array(bboxes, dtype=np.float32), np.array(linds, dtype=np.float32), flags

