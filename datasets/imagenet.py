import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet, ImageFolder
from torchvision import transforms

def data(batch_size_train, batch_size_test):
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

    train_dataset = ImageFolder(
        './datasets/imagenet/',
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True
    )

    test_dataset = ImageFolder(
        './datasets/imagenet/',
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size_test, shuffle=False
    )

    return train_loader, test_loader