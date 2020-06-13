from glob import glob
from datetime import date
import os, logging, re, torch
import matplotlib.pyplot as plt

class SaveManager(object):
    def __init__(self, modelname, interval, max_checkpoints, plot_yrange=(0, 14), savedir='./weights'):
        """
        :param modelname: str, saved model name.
        :param interval: int, save for each designated iteration
        :param max_checkpoints: (Optional) int, how many ssd will be saved during training.
        """
        if max_checkpoints > 15:
            logging.warning('One model size will be about 0.1 GB. Please take care your storage.')
        save_checkpoints_dir = os.path.join(savedir, 'checkpoints')
        today = '{:%Y%m%d}'.format(date.today())

        # check existing checkpoints file
        filepaths = sorted(
            glob(os.path.join(save_checkpoints_dir, modelname + '_i[-]*_checkpoints{}.pth'.format(today))))
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

        if not os.path.isdir(savedir):
            raise FileNotFoundError('{} was not found, please make it'.format(savedir))
        if not os.path.isdir(save_checkpoints_dir):
            raise FileNotFoundError('{} was not found, please make it'.format(save_checkpoints_dir))

        self.savedir = savedir
        self.save_checkpoints_dir = save_checkpoints_dir
        self.modelname = modelname
        self.today = today
        self.interval = interval
        self.plot_yrange = plot_yrange

        self.max_iterations = -1
        self.max_checkpoints = max_checkpoints

    def initialize(self, max_iterations):
        self.max_iterations = max_iterations

    def update_iteration(self, model, now_iteration):
        saved_path = ''
        removed_path = ''
        if now_iteration % self.interval == 0 and self.modelname and now_iteration != self.max_iterations:
            filepaths = sorted(
                glob(os.path.join(self.save_checkpoints_dir,
                                  self.modelname + '_i[-]*_checkpoints{}.pth'.format(self.today))))

            # filepaths = [path for path in os.listdir(save_checkpoints_dir) if re.search(savemodelname + '_i\-*_checkpoints{}.pth'.format(today), path)]
            # print(filepaths)

            # remove oldest checkpoints
            if len(filepaths) > self.max_checkpoints - 1:
                removed_path += os.path.basename(filepaths[0])
                os.remove(filepaths[0])

            # save model
            saved_path = os.path.join(self.save_checkpoints_dir,
                                      self.modelname + '_i-{:07d}_checkpoints{}.pth'.format(now_iteration, self.today))
            torch.save(model.state_dict(), saved_path)


        return saved_path, removed_path

    def finish(self, model, loss_manager):
        # model
        savepath = os.path.join(self.savedir, 'results', self.modelname + '_i-{}.pth'.format(loss_manager.now_iteration))
        torch.save(model.state_dict(), savepath)
        print('Saved model to {}'.format(savepath))

        # graph
        savepath = os.path.join(self.savedir, 'results',
                                self.modelname + '_learning-curve_i-{}.png'.format(loss_manager.now_iteration))
        # initialise the graph and settings
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ion()
        ax.clear()
        # plot
        ax.plot(loss_manager.iterations, loss_manager.totals, label='total')
        ax.plot(loss_manager.iterations, loss_manager.locs, label='loc')
        ax.plot(loss_manager.iterations, loss_manager.confs, label='conf')
        ax.legend()
        if self.plot_yrange:
            ax.axis(ymin=self.plot_yrange[0], ymax=self.plot_yrange[1])

        ax.set_title('Learning curve')
        ax.set_xlabel('iteration')
        ax.set_ylabel('loss')
        # ax.axis(xmin=1, xmax=iterations)
        # save
        fig.savefig(savepath)

        print('Saved graph to {}'.format(savepath))