import matplotlib.pyplot as plt


class LiveGraph(object):
    def __init__(self, yrange=(0, 14)):
        self.yrange = yrange

        self.fig = None
        self.ax = None
        self.train_losses_iteration = []
        self.losses = {}
        self.main_name = ''

    def initialize(self, loss_names):
        # initialise the graph and settings
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        plt.ion()

        self.fig.show()
        self.fig.canvas.draw()

        for name in loss_names:
            self.losses[name] = []
        self.main_name = loss_names[0]

    def update(self, epoch, iteration, losses_iteration, **losses):
        if self.fig is None or self.ax is None:
            raise NotImplementedError('Call initialize first!')

        self.train_losses_iteration = losses_iteration

        self.ax.clear()
        # plot
        for name, _losses in losses.items():
            if not name in self.losses.keys():
                raise KeyError('must pass {} in initialize method'.format(name))
            self.losses[name] = _losses
            self.ax.plot(self.train_losses_iteration, self.losses[name], label=name)

        self.ax.legend()

        # self.ax.axis(xmin=0, xmax=iterations) # too small to see!!
        if self.yrange:
            self.ax.axis(ymin=self.yrange[0], ymax=self.yrange[1])

        self.ax.set_title('Learning curve\nEpoch: {}, Iteration: {}, Loss: {}'.format(epoch, iteration,
                                                                                     self.losses[self.main_name][-1]))

        self.ax.set_xlabel('iteration')
        self.ax.set_ylabel('loss')
        # update
        self.fig.canvas.draw()

    def update_info(self, info):
        if self.fig is None or self.ax is None:
            raise NotImplementedError('Call initialize first!')

        self.ax.clear()
        # plot
        for name, losses in self.losses.items():
            self.ax.plot(self.train_losses_iteration, losses, label=name)

        self.ax.legend()

        # self.ax.axis(xmin=0, xmax=iterations) # too small to see!!
        if self.yrange:
            self.ax.axis(ymin=self.yrange[0], ymax=self.yrange[1])

        self.ax.set_title('Learning curve\n{}'.format(info))
                          #fontsize=self.fontsize)

        self.ax.set_xlabel('iteration')
        self.ax.set_ylabel('loss')
        # update
        self.fig.canvas.draw()