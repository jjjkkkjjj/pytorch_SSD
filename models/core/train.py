__all__ = ['Trainer']

from .utils import _weights_path
from .scheduler import SSDIterMultiStepLR, _IterLRScheduler
from .graph import LiveGraph

import torch
import time
import logging
import os, re, sys
import matplotlib.pyplot as plt
import math
import numpy as np
from glob import glob
from datetime import date
"""
    ref: https://nextjournal.com/gkoehler/pytorch-mnist
"""


class Trainer(object):
    def __init__(self, model, loss_func, optimizer, scheduler=None, gpu=True, log_interval=100):
        self.gpu = gpu

        self.model = model.cuda() if self.gpu else model
        # convert to float
        self.model = self.model.to(dtype=torch.float)
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.scheduler = scheduler

        self.test_losses = []

        self.log_interval = log_interval

    """
    @property
    def model_name(self):
        return self.model.__class__.__name__.lower()
    """

    def train(self, iterations, train_loader, savemodelname='ssd300', checkpoints_iteration_interval=5000, max_checkpoints=15, live_graph=None):
        """
        :param iterations: int, how many iterations during training
        :param train_loader: Dataloader, must return Tensor of images and ground truthes
        :param savemodelname: (Optional) str or None, saved model name. if it's None, model will not be saved after finishing training.
        :param checkpoints_iteration_interval: (Optional) int or None, Whether to save for each designated iteration or not. if it's None, model will not be saved.
        :param max_checkpoints: (Optional) int, how many models will be saved during training.
        :return:
        """

        # calculate epochs
        iter_per_epoch = math.ceil(len(train_loader.dataset) / float(iterations))
        epochs = math.ceil(iterations / float(iter_per_epoch))

        self.model.train()

        log_manager = _LogManager(savemodelname, checkpoints_iteration_interval, self.log_interval, max_checkpoints, epochs,
                                  live_graph)

        for epoch in range(1, epochs + 1):
            if log_manager.isFinish:
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

                loss = self.loss_func(predicts, targets, dboxes=dboxes)
                loss.backward()  # calculate gradient for value with requires_grad=True, shortly back propagation
                # print(self.model.feature_layers.conv1_1.weight.grad)

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                # update log
                log_manager.update_log(self.model, epoch, _iteration + 1, batch_num=len(images),
                                       data_num=len(train_loader.dataset),
                                       iter_per_epoch=len(train_loader), lossval=loss.item())

                if log_manager.isFinish:
                    break


        print('Training finished')
        log_manager.save_model(self.model)



class _LogManager(object):
    def __init__(self, savemodelname, checkpoints_iteration_interval, log_interval, max_checkpoints, iterations, live_graph):
        if savemodelname is None:
            logging.warning('Training model will not be saved!!!')

        if checkpoints_iteration_interval and max_checkpoints > 15:
            logging.warning('One model size will be about 0.1 GB. Please take care your storage.')

        savedir = _weights_path(__file__, _root_num=2, dirname='weights')
        save_checkpoints_dir = os.path.join(savedir, 'checkpoints')
        today = '{:%Y%m%d}'.format(date.today())

        # check existing checkpoints file
        filepaths = sorted(
            glob(os.path.join(save_checkpoints_dir, savemodelname + '_i[-]*_checkpoints{}.pth'.format(today))))
        if len(filepaths) > 0:
            logging.warning('Today\'s checkpoints is remaining. Remove them?\nInput any key. [n]/y')
            i = input()
            if re.match(r'y|yes', i, flags=re.IGNORECASE):
                for file in filepaths:
                    os.remove(file)
                logging.warning('Removed {}'.format(filepaths))
            else:
                logging.warning('Please rename them.')
                exit()

        if live_graph:
            logging.info("You should use jupyter notebook")
            if not isinstance(live_graph, LiveGraph):
                raise ValueError('live_graph must inherit LivaGraph')

            # initialise the graph and settings
            live_graph.initialize()

        # log's info
        self.savedir = savedir
        self.save_checkpoints_dir = save_checkpoints_dir
        self.savemodelname = savemodelname
        self.today = today

        # parameters
        self.log_interval = log_interval
        self.checkpoints_iteration_interval = checkpoints_iteration_interval
        self.max_checkpoints = max_checkpoints
        self.max_iterations = iterations
        self.live_graph = live_graph

        self.train_losses = []
        self.train_losses_iteration = []
        self.total_iteration = 0

    @property
    def isFinish(self):
        return self.total_iteration == self.max_iterations

    def update_log(self, model, epoch, iteration, batch_num,
                   data_num, iter_per_epoch, lossval):
        #template = 'Epoch {}, Loss: {:.5f}, Accuracy: {:.5f}, Test Loss: {:.5f}, Test Accuracy: {:.5f}, elapsed_time {:.5f}'
        iter_template = '\rTraining... Epoch: {}, Iter: {},\t [{}/{}\t ({:.0f}%)]\tLoss: {:.6f}'

        sys.stdout.write(iter_template.format(
            epoch, iteration, iteration * batch_num, data_num,
                              100. * iteration / iter_per_epoch, lossval))
        sys.stdout.flush()

        self.total_iteration += 1
        self.train_losses_iteration += [self.total_iteration]
        self.train_losses += [lossval]

        self._update_log_iteration(epoch)
        self._save_checkpoints_model(iteration, model)

    def _update_log_iteration(self, epoch):

        if self.total_iteration % self.log_interval == 0 or self.total_iteration == 1:
            if self.live_graph:
                self.live_graph.redraw(epoch, self.total_iteration, self.train_losses_iteration, self.train_losses)
                #print('')
            else:
                print('')
                #iter_template = 'Training... Epoch: {}, Iter: {},\tLoss: {:.6f}'
                #print(iter_template.format(
                #    epoch, self.total_iteration, self.train_losses[-1]))

    def _save_checkpoints_model(self, iteration, model):
        info = ''
        if iteration % self.checkpoints_iteration_interval == 0 and self.savemodelname and iteration != self.max_iterations:
            filepaths = sorted(
                glob(os.path.join(self.save_checkpoints_dir,
                                  self.savemodelname + '_i[-]*_checkpoints{}.pth'.format(self.today))))

            # filepaths = [path for path in os.listdir(save_checkpoints_dir) if re.search(savemodelname + '_i\-*_checkpoints{}.pth'.format(today), path)]
            # print(filepaths)
            removedinfo = ''
            # remove oldest checkpoints
            if len(filepaths) > self.max_checkpoints - 1:
                removedinfo += os.path.basename(filepaths[0])
                os.remove(filepaths[0])


            # save model
            savepath = os.path.join(self.save_checkpoints_dir,
                                    self.savemodelname + '_i-{:07d}_checkpoints{}.pth'.format(iteration, self.today))
            torch.save(model.state_dict(), savepath)

            # append information for verbose
            info += 'Saved model to {}'.format(savepath)
            if removedinfo != '':
                if self.live_graph:
                    removedinfo = '\nRemoved {}'.format(removedinfo)
                    info = '\n' + 'Saved model as {}{}'.format(os.path.basename(savepath), removedinfo)
                    self.live_graph.update_info(info)
                else:
                    removedinfo = ' and removed {}'.format(removedinfo)
                    info = '\n' + 'Saved model as {}{}'.format(os.path.basename(savepath), removedinfo)
                    print(info)

    def save_model(self, model):
        if self.savemodelname:
            # model
            savepath = os.path.join(self.savedir, self.savemodelname + '_i-{}.pth'.format(self.epochs))
            torch.save(model.state_dict(), savepath)
            print('Saved model to {}'.format(savepath))


            # graph
            savepath = os.path.join(self.savedir, self.savemodelname + '_learning-curve_i-{}.png'.format(self.epochs))
            # initialise the graph and settings
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.ion()
            ax.clear()
            # plot
            ax.plot(self.train_losses_iteration, self.train_losses)
            ax.set_title('Learning curve')
            ax.set_xlabel('iteration')
            ax.set_ylabel('loss')
            #ax.axis(xmin=1, xmax=iterations)
            # save
            fig.savefig(savepath)

            print('Saved graph to {}'.format(savepath))