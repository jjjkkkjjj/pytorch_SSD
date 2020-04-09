import torch
from torch.nn import functional
import time

"""
    ref: https://nextjournal.com/gkoehler/pytorch-mnist
"""
class Trainer(object):
    def __init__(self, model, loss_func, optimizer):
        self.model = model
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

        for epoch in range(epochs):
            start = time.time()
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_func(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
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