from torch.utils.data import Dataset
from .base import VOCBaseDataset

from .utils import _thisdir

VOC_class_nums = VOCBaseDataset.class_nums

class VOC2007Dataset(Dataset):
    class_nums = VOCBaseDataset.class_nums
    def __init__(self, transform=None):
        self.trainval = VOC2007_TrainValDataset(transform)
        self.test = VOC2007_TestDataset(transform)

    def __getitem__(self, index):
        if index < len(self.trainval):
            return self.trainval[index]
        else:
            return self.test[index - len(self.trainval)]

    def __len__(self):
        return len(self.trainval) + len(self.test)


class VOC2007_TrainValDataset(VOCBaseDataset):
    def __init__(self, transform=None):
        super().__init__(_thisdir + '/voc/voc2007/trainval/VOCdevkit/VOC2007', transform)


class VOC2007_TestDataset(VOCBaseDataset):
    def __init__(self, transform=None):
        super().__init__(_thisdir + '/voc/voc2007/test/VOCdevkit/VOC2007', transform)


class VOC2012Dataset(Dataset):
    class_nums = VOCBaseDataset.class_nums
    def __init__(self, transform=None):
        self.trainval = VOC2012_TrainValDataset(transform)
        self.test = VOC2012_TestDataset(transform)

    def __getitem__(self, index):
        if index < len(self.trainval):
            return self.trainval[index]
        else:
            return self.test[index - len(self.trainval)]

    def __len__(self):
        return len(self.trainval) + len(self.test)

class VOC2012_TrainValDataset(VOCBaseDataset):
    def __init__(self, transform=None):
        super().__init__(_thisdir + '/voc/voc2012/trainval/VOCdevkit/VOC2012', transform)


class VOC2012_TestDataset(VOCBaseDataset):
    def __init__(self, transform=None):
        super().__init__(_thisdir + '/voc/voc2012/test/VOCdevkit/VOC2012', transform)


class Compose(Dataset):
    class_nums = -1
    def __init__(self, class_nums, datasets=(VOC2007Dataset, VOC2012Dataset), transform=None):
        self.transform = transform

        _datasets, _lens = [], []
        for dataset in datasets:
            # initialization
            _datasets += [dataset(transform)]

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