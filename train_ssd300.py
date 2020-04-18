from data.datasets import VOC2007Dataset
from data import transforms, utils

from models.ssd300 import SSD300
from models.core.loss import SSDLoss
from models.core.train import Trainer, SSDIterSchedulerLR

#from torchvision import transforms > not import!!
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Ignore(ignore_difficult=True),
         transforms.Normalize(),
         transforms.Centered(),
         transforms.Resize((300, 300)), # if resizing first, can't be normalized
         transforms.OneHot(class_nums=VOC2007Dataset.class_nums),
         transforms.ToTensor()]
    )
    train_dataset = VOC2007Dataset(transform=transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              collate_fn=utils.batch_ind_fn)

    model = SSD300(class_nums=train_dataset.class_nums, batch_norm=False)
    model.load_vgg_weights()
    print(model)

    optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    iter_sheduler = SSDIterSchedulerLR(optimizer, milestones=(10000, 200000, 300000), gamma=0.1, verbose=True)
    trainer = Trainer(model, loss_func=SSDLoss(), optimizer=optimizer, iter_sheduler=iter_sheduler, gpu=True)
    trainer.train(30, train_loader)