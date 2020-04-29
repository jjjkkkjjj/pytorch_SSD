from data import datasets
from data import transforms, utils

from ssd.models.ssd300 import SSD300
from ssd.core.loss import SSDLoss
from ssd.train import *
from ssd.core.scheduler import *

#from torchvision import transforms > not import!!
from torch.utils.data import DataLoader
from torch.optim.adam import Adam

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Ignore(difficult=True),
         transforms.Normalize(),
         transforms.Centered(),
         transforms.Resize((300, 300)), # if resizing first, can't be normalized
         transforms.SubtractMean((123.68, 116.779, 103.939)),
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
    """
    imgs, gts = utils.batch_ind_fn((train_dataset[2000],))
    p, d = model(imgs)
    from ssd.core.boxes import matching_strategy
    matching_strategy(gts, d, batch_num=1)
    """
    #optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    #iter_sheduler = SSDIterMultiStepLR(optimizer, milestones=(10, 20, 30), gamma=0.1, verbose=True)
    iter_sheduler = SSDIterStepLR(optimizer, step_size=10000, gamma=0.1, verbose=True)

    save_manager = SaveManager(modelname='ssd300', interval=10, max_checkpoints=3)
    log_manager = LogManager(interval=10, save_manager=save_manager, live_graph=None)
    trainer = TrainLogger(model, loss_func=SSDLoss(), optimizer=optimizer, scheduler=iter_sheduler, log_manager=log_manager, gpu=True)

    trainer.train(70, train_loader)
