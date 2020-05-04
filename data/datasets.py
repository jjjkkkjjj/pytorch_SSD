from torch.utils.data import Dataset
from .base import VOCBaseDataset, VOC_classes, VOC_class_nums

from .utils import DATA_ROOT


class VOC2007Dataset(Dataset):
    class_nums = VOCBaseDataset.class_nums
    def __init__(self, transform=None, target_transform=None, augmentation=None):
        self.trainval = VOC2007_TrainValDataset(transform=transform, target_transform=target_transform, augmentation=augmentation)
        self.test = VOC2007_TestDataset(transform=transform, target_transform=target_transform, augmentation=augmentation)

    def __getitem__(self, index):
        if index < len(self.trainval):
            return self.trainval[index]
        else:
            return self.test[index - len(self.trainval)]

    def __len__(self):
        return len(self.trainval) + len(self.test)


class VOC2007_TrainValDataset(VOCBaseDataset):
    def __init__(self, focus='trainval', transform=None, target_transform=None, augmentation=None):
        super().__init__(DATA_ROOT + '/voc/voc2007/trainval/VOCdevkit/VOC2007', focus, transform, target_transform, augmentation)


class VOC2007_TestDataset(VOCBaseDataset):
    def __init__(self, focus='test', transform=None, target_transform=None, augmentation=None):
        super().__init__(DATA_ROOT + '/voc/voc2007/test/VOCdevkit/VOC2007', focus, transform, target_transform, augmentation)


class VOC2012Dataset(Dataset):
    class_nums = VOCBaseDataset.class_nums
    def __init__(self, transform=None, target_transform=None, augmentation=None):
        self.trainval = VOC2012_TrainValDataset(transform=transform, target_transform=target_transform, augmentation=augmentation)
        self.test = VOC2012_TestDataset(transform=transform, target_transform=target_transform, augmentation=augmentation)

    def __getitem__(self, index):
        if index < len(self.trainval):
            return self.trainval[index]
        else:
            return self.test[index - len(self.trainval)]

    def __len__(self):
        return len(self.trainval) + len(self.test)

class VOC2012_TrainValDataset(VOCBaseDataset):
    def __init__(self, focus='trainval', transform=None, target_transform=None, augmentation=None):
        super().__init__(DATA_ROOT + '/voc/voc2012/trainval/VOCdevkit/VOC2012', focus,
                         transform, target_transform, augmentation)


class VOC2012_TestDataset(VOCBaseDataset):
    def __init__(self, focus='test', transform=None, target_transform=None, augmentation=None):
        super().__init__(DATA_ROOT + '/voc/voc2012/test/VOCdevkit/VOC2012', focus,
                         transform, target_transform, augmentation)


class Compose(Dataset):
    class_nums = -1
    def __init__(self, class_nums, datasets=(VOC2007Dataset, VOC2012Dataset), transform=None, target_transform=None, augmentation=None):
        self.transform = transform
        self.target_transform = target_transform
        self.augmentation = augmentation

        _datasets, _lens = [], []
        for dataset in datasets:
            # initialization
            _datasets += [dataset(transform=transform, target_transform=target_transform, augmentation=augmentation)]

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