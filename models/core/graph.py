import matplotlib.pyplot as plt


class LiveGraph(object):
    def __init__(self, yrange=(0, 8)):
        self.yrange = yrange

        self.fig = None
        self.ax = None
        self.train_losses_iteration = []
        self.train_losses = []

    def initialize(self):
        # initialise the graph and settings
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        plt.ion()

        self.fig.show()
        self.fig.canvas.draw()

    def redraw(self, epoch, iteration, losses_iteration, losses, **another_vals):
        if self.fig is None or self.ax is None:
            raise NotImplementedError('Call initialize first!')

        self.train_losses_iteration = losses_iteration
        self.train_losses = losses

        self.ax.clear()
        # plot
        self.ax.plot(self.train_losses_iteration, self.train_losses, label='total')
        for name, vals in another_vals.items():
            self.ax.plot(self.train_losses_iteration, vals, label=name)

        self.ax.legend()

        # self.ax.axis(xmin=0, xmax=iterations) # too small to see!!
        if self.yrange:
            self.ax.axis(ymin=self.yrange[0], ymax=self.yrange[1])

        self.ax.set_title('Learning curve\nEpoch: {}, Iteration: {}, Loss: {}'.format(epoch, iteration,
                                                                                     self.train_losses[-1]))

        self.ax.set_xlabel('iteration')
        self.ax.set_ylabel('loss')
        # update
        self.fig.canvas.draw()

    def update_info(self, info):
        if self.fig is None or self.ax is None:
            raise NotImplementedError('Call initialize first!')

        self.ax.clear()
        # plot
        self.ax.plot(self.train_losses_iteration, self.train_losses)
        # self.ax.axis(xmin=0, xmax=iterations) # too small to see!!
        if self.yrange:
            self.ax.axis(ymin=self.yrange[0], ymax=self.yrange[1])

        self.ax.set_title('Learning curve\n{}'.format(info))
                          #fontsize=self.fontsize)

        self.ax.set_xlabel('iteration')
        self.ax.set_ylabel('loss')
        # update
        self.fig.canvas.draw()