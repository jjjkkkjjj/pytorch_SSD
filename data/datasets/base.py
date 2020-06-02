import torch
from torch.utils.data import Dataset
import numpy as np
import abc

from .._utils import _check_ins, _contain_ignore
from ..target_transforms import Ignore

"""
ref > https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:

__len__ so that len(dataset) returns the size of the dataset.
__getitem__ to support the indexing such that dataset[i] can be used to get ith sample

"""

class ObjectDetectionDatasetBase(Dataset):
    def __init__(self, ignore=None, transform=None, target_transform=None, augmentation=None):
        """
        :param ignore: target_transforms.Ignore
        :param transform: instance of transforms
        :param target_transform: instance of target_transforms
        :param augmentation:  instance of augmentations
        """
        #ignore, target_transform = _separate_ignore(target_transform)
        self.ignore = _check_ins('ignore', ignore, Ignore, allow_none=True)
        self.transform = transform
        self.target_transform = _contain_ignore(target_transform)
        self.augmentation = augmentation

    @abc.abstractmethod
    def _get_image(self, index):
        """
        :param index: int
        :return:
            rgb image(Tensor)
        """
        raise NotImplementedError('\'_get_image\' must be overridden')

    @abc.abstractmethod
    def _get_bbox_lind(self, index):
        """
        :param index: int
        :return:
            list of bboxes, list of bboxes' label index, list of flags([difficult, truncated])
        """
        raise NotImplementedError('\'_get_bbox_lind\' must be overridden')

    def __getitem__(self, index):
        """
        :param index: int
        :return:
            img : rgb image(Tensor or ndarray)
            targets : Tensor or ndarray of bboxes and labels [box, label]
            = [xmin, ymin, xmamx, ymax, label index(or relu_one-hotted label)]
            or
            = [cx, cy, w, h, label index(or relu_one-hotted label)]
        """
        img = self._get_image(index)
        bboxes, linds, flags = self._get_bbox_lind(index)

        img, bboxes, linds, flags = self.apply_transform(img, bboxes, linds, flags)

        # concatenate bboxes and linds
        if isinstance(bboxes, torch.Tensor) and isinstance(linds, torch.Tensor):
            if linds.ndim == 1:
                linds = linds.unsqueeze(1)
            targets = torch.cat((bboxes, linds), dim=1)
        else:
            if linds.ndim == 1:
                linds = linds[:, np.newaxis]
            targets = np.concatenate((bboxes, linds), axis=1)

        return img, targets

    def apply_transform(self, img, bboxes, linds, flags):
        """
        IMPORTATANT: apply transform function in order with ignore, augmentation, transform and target_transform
        :param img:
        :param bboxes:
        :param linds:
        :param flags:
        :return:
            Transformed img, bboxes, linds, flags
        """
        # To Percent mode
        height, width, channel = img.shape
        # bbox = [xmin, ymin, xmax, ymax]
        # [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
        bboxes[:, 0::2] /= float(width)
        bboxes[:, 1::2] /= float(height)

        if self.ignore:
            bboxes, linds, flags = self.ignore(bboxes, linds, flags)

        if self.augmentation:
            img, bboxes, linds, flags = self.augmentation(img, bboxes, linds, flags)

        if self.transform:
            img, bboxes, linds, flags = self.transform(img, bboxes, linds, flags)

        if self.target_transform:
            bboxes, linds, flags = self.target_transform(bboxes, linds, flags)

        return img, bboxes, linds, flags

    @abc.abstractmethod
    def __len__(self):
        pass



from .voc import VOC2007Dataset, VOC2012Dataset

class Compose(Dataset):
    class_nums = -1
    def __init__(self, class_nums, datasets=(VOC2007Dataset, VOC2012Dataset), **kwargs):
        """
        :param class_nums: int, class number
        :param datasets: tuple of Dataset
        :param kwargs:
            :param ignore:
            :param transform:
            :param target_transform:
            :param augmentation:
        """
        self.transform = kwargs.get('transform', None)
        self.target_transform = kwargs.get('target_transform', None)
        self.augmentation = kwargs.get('augmentation', None)

        datasets = _check_ins('datasets', datasets, (tuple, list))

        _datasets, _lens = [], []
        for dataset in datasets:
            try:
                dataset = dataset(**kwargs)
            except Exception as e:
                raise ValueError('Invalid arguments were passed. {} could not be initialized because\n{}'.format(dataset.__name__, e))
            dataset = _check_ins('element of datasets', dataset, Dataset)
            # initialization
            _datasets += [dataset]

            _lens += [len(_datasets[-1])]

        self.datasets = _datasets
        self.lens = _lens
        Compose.class_nums = class_nums


    def __getitem__(self, index):
        for i in range(len(self.lens)):
            if index < sum(self.lens[:i+1]):
                return self.datasets[i][index - sum(self.lens[:i])]

        raise ValueError('Index out of range')

    def __len__(self):
        return sum(self.lens)


