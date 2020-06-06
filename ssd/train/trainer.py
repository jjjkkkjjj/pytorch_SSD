import math
import torch
import time

from .log import LogManager
from .._utils import _check_ins
from ..models.base import SSDBase
from .eval import EvaluatorBase
"""
    ref: https://nextjournal.com/gkoehler/pytorch-mnist
"""
class TrainLogger(object):
    model: SSDBase

    def __init__(self, model, loss_func, optimizer, log_manager, scheduler=None, gpu=True):
        self.gpu = gpu

        self.model = _check_ins('model', model, SSDBase)
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

    def train(self, max_iterations, train_loader):#, evaluator=None):
        """
        :param max_iterations: int, how many iterations during training
        :param train_loader: Dataloader, must return Tensor of images and ground truthes
        :param evaluator: EvaluatorBase, if it's None, Evaluation will not be run
        :return:
        """

        #evaluator = _check_ins('evaluator', evaluator, EvaluatorBase, allow_none=True)

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
                start = time.time()

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

                end = time.time()

                # update train
                self.log_manager.update_iteration(self.model, epoch, _iteration + 1, batch_num=len(images), data_num=len(train_loader.dataset),
                                                  iter_per_epoch=len(train_loader), loclossval=locloss.item(), conflossval=confloss.item(), iter_time=end-start)
                """
                too slow...
                if evaluator and self.log_manager.now_iteration % evaluator.iteration_interval:
                    self.model.eval()
                    print(evaluator(self.model))
                    self.model.train()
                """
                if self.log_manager.isFinish:
                    break


        print('\nTraining finished')
        self.log_manager.finish(self.model)