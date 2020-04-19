from data import datasets
from data import transforms, utils

from models.ssd300 import SSD300
from models.core.loss import SSDLoss
from models.core.train import *

#from torchvision import transforms > not import!!
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Ignore(difficult=True),
         transforms.Normalize(),
         transforms.Centered(),
         transforms.Resize((300, 300)), # if resizing first, can't be normalized
         transforms.OneHot(class_nums=datasets.VOC_class_nums),
         transforms.ToTensor()]
    )
    train_dataset = datasets.Compose(datasets.VOC_class_nums, datasets=(datasets.VOC2007Dataset, datasets.VOC2012_TrainValDataset), transform=transform)
    #train_dataset = datasets.VOC2007Dataset(transform=transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              collate_fn=utils.batch_ind_fn)

    model = SSD300(class_nums=train_dataset.class_nums, batch_norm=False)
    model.load_vgg_weights()
    print(model)

    optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    #iter_sheduler = SSDIterMultiStepLR(optimizer, milestones=(10, 20, 30), gamma=0.1, verbose=True)
    iter_sheduler = SSDIterStepLR(optimizer, step_size=20, gamma=0.1, verbose=True)
    trainer = Trainer(model, loss_func=SSDLoss(), optimizer=optimizer, iter_sheduler=iter_sheduler, gpu=True)
    trainer.train(70, train_loader, checkpoints_interval=10)