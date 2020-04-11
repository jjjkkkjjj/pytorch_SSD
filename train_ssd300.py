from data.datasets import VOC2007Dataset
from data import transforms

from models.ssd300 import SSD300

#from torchvision import transforms > not import!!
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Ignore(ignore_difficult=True),
         transforms.Normalize(),
         transforms.OneHot(class_nums=VOC2007Dataset.class_nums)]
    )
    train_dataset = VOC2007Dataset(transform=transform).train
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True)
    train_dataset[0]
    model = SSD300(class_nums=train_dataset.class_nums)
    print(model)

    optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    #trainer = Trainer(model, loss_func=, optimizer=optimizer)