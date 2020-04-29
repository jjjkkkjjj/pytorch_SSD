import logging
import os, sys

from .save import SaveManager
from .graph import LiveGraph

class LogManager(object):
    def __init__(self, interval, save_manager, live_graph=None):

        if live_graph:
            logging.info("You should use jupyter notebook")
            if not isinstance(live_graph, LiveGraph):
                raise ValueError('live_graph must inherit LivaGraph')

            # initialise the graph and settings
            live_graph.initialize(loss_names=['total', 'loc', 'conf'])

        # parameters
        self.interval = interval
        self.live_graph = live_graph

        if not isinstance(save_manager, SaveManager):
            raise ValueError('save_manager must be \'SaveManager\' instance')
        self.save_manager = save_manager

        self._losses_manager = _SSDLossesManager()


    @property
    def isFinish(self):
        return self._losses_manager.now_iteration == self.max_iterations
    @property
    def cuimode(self):
        return self.live_graph is None
    @property
    def now_iteration(self):
        return self._losses_manager.now_iteration
    @property
    def max_iterations(self):
        return self.save_manager.max_iterations

    def initialize(self, max_iterations):
        self.save_manager.initialize(max_iterations)

    def update_iteration(self, model, epoch, iteration, batch_num,
                         data_num, iter_per_epoch, loclossval, conflossval):

        if self.isFinish:
            return

        self._losses_manager.update_iteration(totalval=loclossval+conflossval, locval=loclossval, confval=conflossval)

        # template = 'Epoch {}, Loss: {:.5f}, Accuracy: {:.5f}, Test Loss: {:.5f}, Test Accuracy: {:.5f}, elapsed_time {:.5f}'
        iter_template = '\rTraining... Epoch: {}, Iter: {},\t [{}/{}\t ({:.0f}%)]\tLoss: {:.6f}, Loc Loss: {:.6f}, Conf Loss: {:.6f}'

        sys.stdout.write(iter_template.format(
            epoch, self.now_iteration, iteration * batch_num, data_num,
                                         100. * iteration / iter_per_epoch, loclossval + conflossval, loclossval,
            conflossval))
        sys.stdout.flush()

        self._update_log_iteration(epoch)
        saved_path, removed_path = self.save_manager.update_iteration(model, iteration)

        if saved_path == '':
            return

        # append information for verbose
        saved_info = '\nSaved model to {}\n'.format(saved_path)
        if removed_path != '':
            if self.cuimode:
                removed_info = ' and removed {}'.format(removed_path)
                saved_info = '\n' + 'Saved model as {}{}\n'.format(os.path.basename(saved_path), removed_info)
                print(saved_info)

            else:
                removed_info = '\nRemoved {}'.format(removed_path)
                saved_info = '\n' + 'Saved model as {}{}'.format(os.path.basename(saved_path), removed_info)
                self.live_graph.update_info(saved_info)
        else:
            print(saved_info)

    def _update_log_iteration(self, epoch):

        if self.now_iteration % self.interval == 0 or self.now_iteration == 1:
            if self.cuimode:
                print('')
                # iter_template = 'Training... Epoch: {}, Iter: {},\tLoss: {:.6f}'
                # print(iter_template.format(
                #    epoch, self.total_iteration, self.train_losses[-1]))
            else:
                self.live_graph.update(epoch, self.now_iteration, self._losses_manager.iterations,
                                       total=self._losses_manager.totals, loc=self._losses_manager.locs, conf=self._losses_manager.confs)
                # print('')



    def finish(self, model):
        self.save_manager.finish(model, self._losses_manager)

class _SSDLossesManager(object):
    def __init__(self):
        self.now_iteration = 0

        self.iterations = []
        self.totals = []
        self.locs = []
        self.confs = []

    def update_iteration(self, totalval, locval, confval):
        self.now_iteration += 1
        self.iterations += [self.now_iteration]
        self.totals += [totalval]
        self.locs += [locval]
        self.confs += [confval]

