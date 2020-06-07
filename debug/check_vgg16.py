from ssd.models.vgg import VGGBase16
#from ssd.train.trainer import TrainLogger
#from data.mnist import data

if __name__ == '__main__':
    model = VGGBase16(3, class_nums=1000)
    model.load_weights()
    print(model)


    """
    train_images, train_labels, test_images, test_labels = data()

    a = TrainLogger(model=model, loss_func=)
    """