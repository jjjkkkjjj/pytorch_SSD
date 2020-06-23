import math
import torch
from torch import nn
import time, logging

from .log import LogManager
from .._utils import _check_ins
from ..models.base import SSDBase
from .eval import EvaluatorBase
"""
    ref: https://nextjournal.com/gkoehler/pytorch-mnist
"""
class TrainLogger(object):
    model: SSDBase

    def __init__(self, model, loss_func, optimizer, log_manager, scheduler=None):

        self._model = _check_ins('model', model, (SSDBase, nn.DataParallel))

        if torch.cuda.is_available():
            if not 'cuda' in self.model.device.type:
                logging.warning('You can use CUDA device but you didn\'t set CUDA device.'
                                'To use CUDA, call \"model.cuda()\"')
        self.device = self.model.device
        torch.set_default_tensor_type(torch.FloatTensor)

        # convert to float
        #self.model = self.model.to(dtype=torch.float)
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.scheduler = scheduler

        self.test_losses = []

        if isinstance(log_manager, LogManager):
            self.log_manager = log_manager
        else:
            raise ValueError('logmanager must be \'Logmanager\' instance')


    @property
    def model(self):
        if isinstance(self._model, nn.DataParallel):
            return self._model.module
        else:
            return self._model


    def train(self, max_iterations, train_loader, start_iteration=0):#, evaluator=None):
        """
        :param max_iterations: int, how many iterations during training
        :param train_loader: Dataloader, must return Tensor of images and ground truthes
        :param start_iteration: int
        :param evaluator: EvaluatorBase, if it's None, Evaluation will not be run
        :return:
        """

        #evaluator = _check_ins('evaluator', evaluator, EvaluatorBase, allow_none=True)

        # calculate epochs
        iter_per_epoch = math.ceil(len(train_loader.dataset) / float(max_iterations))
        epochs = math.ceil(max_iterations / float(iter_per_epoch))

        self.model.train()

        self.log_manager.initialize(max_iterations, start_iteration=start_iteration)

        for epoch in range(1, epochs + 1):
            if self.log_manager.isFinish:
                break

            for _iteration, (images, targets) in enumerate(train_loader):
                self.optimizer.zero_grad()

                images = images.to(self.device)
                targets = [target.to(self.device) for target in targets]
                start = time.time()

                # set variable
                # images.requires_grad = True
                # targets.requires_grad = True
                pos_indicator, predicts, gts = self.model(images, targets)

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
                                                  iter_per_epoch=len(train_loader), loclossval=locloss.item(), conflossval=confloss.item(),
                                                  lossval=loss.item(), iter_time=end-start)
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