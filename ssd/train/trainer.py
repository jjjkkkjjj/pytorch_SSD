import math
import torch

from .log import LogManager
from .._utils import check_instance
from ..models.base import SSDBase
"""
    ref: https://nextjournal.com/gkoehler/pytorch-mnist
"""
class TrainLogger(object):
    model: SSDBase

    def __init__(self, model, loss_func, optimizer, log_manager, scheduler=None, gpu=True):
        self.gpu = gpu

        self.model = check_instance('model', model, SSDBase)
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
                    targets = [target.cuda() for target in targets]

                # set variable
                # images.requires_grad = True
                # targets.requires_grad = True
                pos_indicator, predicts, gts = self.model.learn(images, targets)

                confloss, locloss = self.loss_func(pos_indicator, predicts, gts)
                loss = confloss + self.loss_func.alpha * locloss
                loss.backward()  # calculate gradient for value with requires_grad=True, shortly back propagation
                #print(self.model.feature_layers.convRL1_1.conv.weight.grad)
                #print(self.model.localization_layers.conv_loc_1.weight.grad)
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