import math
import torch

from .log import LogManager
"""
    ref: https://nextjournal.com/gkoehler/pytorch-mnist
"""
class TrainLogger(object):
    def __init__(self, model, loss_func, optimizer, log_manager, scheduler=None, gpu=True):
        self.gpu = gpu

        self.model = model.cuda() if self.gpu else model
        # convert to float
        self.model = self.model.to(dtype=torch.float)
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.scheduler = scheduler

        self.test_losses = []

        if isinstance(log_manager, LogManager):
            self.log_manager = log_manager
        else:
            raise ValueError('logmanager must be \'Logmanager\' instance')

    """
    @property
    def model_name(self):
        return self.model.__class__.__name__.lower()
    """

    def train(self, max_iterations, train_loader):
        """
        :param max_iterations: int, how many iterations during training
        :param train_loader: Dataloader, must return Tensor of images and ground truthes
        :return:
        """

        # calculate epochs
        iter_per_epoch = math.ceil(len(train_loader.dataset) / float(max_iterations))
        epochs = math.ceil(max_iterations / float(iter_per_epoch))

        self.model.train()

        self.log_manager.initialize(max_iterations)

        for epoch in range(1, epochs + 1):
            if self.log_manager.isFinish:
                break

            for _iteration, (images, targets) in enumerate(train_loader):
                self.optimizer.zero_grad()

                if self.gpu:
                    images = images.cuda()
                    targets = targets.cuda()

                # set variable
                # images.requires_grad = True
                # gts.requires_grad = True

                predicts, dboxes = self.model(images)

                if self.gpu:
                    dboxes = dboxes.cuda()

                confloss, locloss = self.loss_func(predicts, targets, dboxes=dboxes)
                loss = confloss + self.loss_func.alpha * locloss
                loss.backward()  # calculate gradient for value with requires_grad=True, shortly back propagation
                # print(self.model.feature_layers.conv1_1.weight.grad)

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                # update train
                self.log_manager.update_iteration(self.model, epoch, _iteration + 1, batch_num=len(images), data_num=len(train_loader.dataset),
                                                  iter_per_epoch=len(train_loader), loclossval=locloss.item(), conflossval=confloss.item())

                if self.log_manager.isFinish:
                    break


        print('\nTraining finished')
        self.log_manager.finish(self.model)