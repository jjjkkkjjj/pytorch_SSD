from data.voc.datasets import VOC2007Dataset
from data.voc.transforms import Normalize

from models.ssd300 import SSD300
from models.train.trainer import Trainer

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD

if __name__ == '__main__':
    transform = transforms.Compose(
        [Normalize()]
    )
    train_dataset = VOC2007Dataset(transform=transform).train
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True)

    model = SSD300(class_nums=train_dataset.class_nums)
    print(model)

    optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    #trainer = Trainer(model, loss_func=, optimizer=optimizer)