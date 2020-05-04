from data import datasets
from data import transforms, target_transforms, augmentations, utils

from ssd.models.ssd300 import SSD300
from ssd.train import *

#from torchvision import transforms > not import!!
from torch.utils.data import DataLoader
from torch.optim.adam import Adam

if __name__ == '__main__':
    """
    augmentation = augmentations.Compose(
        []
    )"""
    augmentation = augmentations.AugmentationOriginal()

    transform = transforms.Compose(
        [transforms.Normalize(rgb_means=(103.939, 116.779, 123.68), rgb_stds=1),
         transforms.Resize((300, 300)),
         transforms.ToTensor()]
    )
    target_transform = target_transforms.Compose(
        [target_transforms.Ignore(difficult=True),
         target_transforms.ToCentroids(),
         target_transforms.OneHot(class_nums=datasets.VOC_class_nums),
         target_transforms.ToTensor()]
    )


    train_dataset = datasets.Compose(datasets.VOC_class_nums, datasets=(datasets.VOC2007Dataset, datasets.VOC2012_TrainValDataset),
                                     transform=transform, target_transform=target_transform, augmentation=augmentation)
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
    log_manager = LogManager(interval=10, save_manager=save_manager, loss_interval=10, live_graph=None)
    trainer = TrainLogger(model, loss_func=SSDLoss(), optimizer=optimizer, scheduler=iter_sheduler, log_manager=log_manager, gpu=True)

    trainer.train(30, train_loader)
