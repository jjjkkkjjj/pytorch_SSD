from data import datasets
from data import transforms, utils

from ssd.models.ssd300 import SSD300

from torch.utils.data import DataLoader

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Ignore(difficult=True),
         transforms.Normalize(),
         transforms.Centered(),
         transforms.Resize((300, 300)),  # if resizing first, can't be normalized
         transforms.OneHot(class_nums=datasets.VOC_class_nums),
         transforms.ToTensor()]
    )

    test_dataset = datasets.VOC2007_TrainValDataset(transform=transform)
    test_loader = DataLoader(test_dataset,
                              batch_size=32,
                              shuffle=False,
                              collate_fn=utils.batch_ind_fn)

    model = SSD300(class_nums=test_dataset.class_nums, batch_norm=False).build_test('./weights/ssd300-voc2007-separated_i-60000.pth')
    images = [test_dataset[i][0] for i in range(50)]
    model.inference(images)
