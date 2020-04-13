import torch
from torch.nn import functional
from torch.autograd import Variable
import time

"""
    ref: https://nextjournal.com/gkoehler/pytorch-mnist
"""
class Trainer(object):
    def __init__(self, model, loss_func, optimizer, gpu=True):
        self.gpu = gpu

        self.model = model.cuda() if self.gpu else model
        # convert to float
        self.model = self.model.to(dtype=torch.float)
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.train_losses = []
        self.train_counter = []

        self.test_losses = []
        self.test_counter = []

        self.log_interval = 10

    def train(self, epochs, train_loader):
        self.model.train()

        template = 'Epoch {}, Loss: {:.5f}, Accuracy: {:.5f}, Test Loss: {:.5f}, Test Accuracy: {:.5f}, elapsed_time {:.5f}'
        iter_template = 'Training... Epoch: {}, Iter: {},\t [{}/{}\t ({:.0f}%)]\tLoss: {:.6f}'
        for epoch in range(epochs):
            start = time.time()
            for batch_idx, (images, gts) in enumerate(train_loader):
                self.optimizer.zero_grad()
                if self.gpu:
                    images = images.cuda()
                    gts = gts.cuda()

                # set variable
                images.requires_grad = True
                gts.requires_grad = True

                predicts, dboxes = self.model(images)
                if self.gpu:
                    dboxes = dboxes.cuda()
                loss = self.loss_func(predicts, gts, dboxes=dboxes)
                loss.backward() # calculate gradient for value with requires_grad=True, shortly back propagation
                #print(self.model.feature_layers.conv1_1.weight.grad)
                self.optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print(iter_template.format(
                        epoch, batch_idx, batch_idx * len(images), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))
                    self.train_losses.append(loss.item())
                    self.train_counter.append(
                        (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
                    #torch.save(self.model.state_dict(), '/results/model.pth')
                    #torch.save(self.optimizer.state_dict(), '/results/optimizer.pth')
            elapsed_time = time.time() - start
            """
            for test_image, test_label in zip(test_images, test_labels):
                self._test_step(test_image, test_label)

            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.train_acc.result() * 100,
                                  self.test_loss.result(),
                                  self.test_acc.result() * 100,
                                  elapsed_time))
            """