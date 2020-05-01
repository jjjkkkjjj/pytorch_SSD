from data import datasets
from data import transforms, target_transforms, augmentations, utils

from ssd.models.ssd300 import SSD300

from torch.utils.data import DataLoader

if __name__ == '__main__':
    augmentaion = augmentations.Compose(
        []
    )

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

    test_dataset = datasets.Compose(datasets.VOC_class_nums,
                                     datasets=(datasets.VOC2007Dataset, datasets.VOC2012_TrainValDataset),
                                     transform=transform, target_transform=target_transform, augmentation=augmentaion)

    test_loader = DataLoader(test_dataset,
                              batch_size=32,
                              shuffle=False,
                              collate_fn=utils.batch_ind_fn)

    model = SSD300(class_nums=test_dataset.class_nums, batch_norm=False)
    model.load_weights('weights/checkpoints/ssd300-voc2007_i-0018000_checkpoints20200430.pth')
    model.eval()

    images = [test_dataset[i][0] for i in range(2)]
    model.inference(images)
