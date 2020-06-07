from data import datasets
from data import transforms, target_transforms, augmentations, utils

from ssd.models.ssd300 import SSD300
from ssd.train.eval import VOC2007Evaluator
import torch
from torch.utils.data import DataLoader
import cv2

if __name__ == '__main__':
    augmentation = None

    transform = transforms.Compose(
            [transforms.Resize((300, 300)),
             transforms.ToTensor(),
             transforms.Normalize(rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))]
    )
    target_transform = target_transforms.Compose(
        [target_transforms.ToCentroids(),
         target_transforms.OneHot(class_nums=datasets.VOC_class_nums, add_background=True),
         target_transforms.ToTensor()]
    )
    test_dataset = datasets.Compose(datasets=(datasets.VOC2012_TrainValDataset,),
                                    transform=transform, target_transform=target_transform, augmentation=augmentation)

    test_loader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=True,
                             collate_fn=utils.batch_ind_fn,
                             num_workers=4,
                             pin_memory=False)

    model = SSD300(class_labels=datasets.VOC_class_labels, batch_norm=False).to(torch.device('cpu'))
    model.load_weights('./weights/ssd300-voc2007/ssd300-voc2007_i-60000.pth')
    model.eval()
    print(model)

    evaluator = VOC2007Evaluator(test_loader, iteration_interval=5000)
    ap = evaluator(model)
    print(ap)

    image = cv2.cvtColor(cv2.imread('assets/coco_testimg.jpg'), cv2.COLOR_BGR2RGB)
    infers, imgs = model.infer(cv2.resize(image, (300, 300)), visualize=True, toNorm=True)
    for img in imgs:
        cv2.imshow('result', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey()

    images = [test_dataset[i][0] for i in range(20)]
    inf, ret_imgs = model.infer(images, visualize=True, toNorm=False)
    for img in ret_imgs:
        cv2.imshow('result', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey()