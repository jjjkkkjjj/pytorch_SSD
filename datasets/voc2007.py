from .dataset import VOCDataset, Dataset
from .dataset import VOCDatasetTransform

class VOC2007Dataset(Dataset):
    def __init__(self, transform=None):
        self._train = VOC2007_TrainDataset(transform)
        self._test = VOC2007_TestDataset(transform)

    def __getitem__(self, index):
        if index < len(self._train):
            return self._train[index]
        else:
            return self._test[index - len(self._test)]

    def __len__(self):
        return len(self._train) + len(self._test)

class VOC2007_TrainDataset(VOCDataset):
    def __init__(self, transform=None):
        super().__init__('./datasets/voc2007/train/VOCdevkit/VOC2007', transform)

class VOC2007_TestDataset(VOCDataset):
    def __init__(self, transform=None):
        super().__init__('./datasets/voc2007/test/VOCdevkit/VOC2007', transform)