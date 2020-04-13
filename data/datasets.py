from torch.utils.data import Dataset
from .base import VOCBaseDataset

class VOC2007Dataset(Dataset):
    class_nums = VOCBaseDataset.class_nums
    def __init__(self, transform=None):
        self.train = VOC2007_TrainDataset(transform)
        self.test = VOC2007_TestDataset(transform)

    def __getitem__(self, index):
        if index < len(self.train):
            return self.train[index]
        else:
            return self.test[index - len(self.train)]

    def __len__(self):
        return len(self.train) + len(self.test)

class VOC2007_TrainDataset(VOCBaseDataset):
    def __init__(self, transform=None):
        super().__init__('./data/voc/voc2007/train/VOCdevkit/VOC2007', transform)

class VOC2007_TestDataset(VOCBaseDataset):
    def __init__(self, transform=None):
        super().__init__('./data/voc/voc2007/test/VOCdevkit/VOC2007', transform)