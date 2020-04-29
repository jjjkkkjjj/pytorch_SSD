from ssd.models.vgg import VGG11
from torchvision import models

if __name__ == '__main__':
    #model = VGG11(3, class_nums=1000, load_model='./ssd/weights/vgg11-bbd30ac9.pth')
    model = VGG11(3, class_nums=1000)
    #model.load_weights()
    model.load_weights(path='./weights/vgg11-bbd30ac9.pth')
    print(model)
    print(models.vgg11())


    """
    train_loader, test_loader = data(256, 256)

    loss_func = functional.nll_loss
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = TrainLogger(model, loss_func, optimizer)
    trainer.train(100, train_loader)
    """
