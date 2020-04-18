import torch
import time

"""
    ref: https://nextjournal.com/gkoehler/pytorch-mnist
"""
class Trainer(object):
    def __init__(self, model, loss_func, optimizer, iter_sheduler=None, gpu=True):
        self.gpu = gpu

        self.model = model.cuda() if self.gpu else model
        # convert to float
        self.model = self.model.to(dtype=torch.float)
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.iter_scheduler = SSDIterSchedulerLR(optimizer) if iter_sheduler is None else iter_sheduler

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
                self.iter_scheduler.step()
                #print([param_group['lr'] for param_group in self.optimizer.param_groups])
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

from torch.optim.lr_scheduler import MultiStepLR

class SSDIterSchedulerLR(MultiStepLR):
    def __init__(self, optimizer, milestones=(40000, 50000, 60000), gamma=0.1, last_iteration=-1, verbose=True):
        super().__init__(optimizer, milestones, gamma, last_epoch=last_iteration)
        self.last_iteration = last_iteration
        self.verbose = verbose
        self._prev_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self):
        ret = super().get_lr()
        self.last_iteration = self.last_epoch
        if self.last_iteration in self.milestones:
            print("Iteration reached milestone: {}. Change lr={} to {}".format(self.last_iteration, self._prev_lr, ret))

        self._prev_lr = ret
        return self._prev_lr


# ref > https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py
"""
from torch.optim.optimizer import Optimizer
import warnings
import weakref
from functools import wraps
from typing import Counter

ITERATION_DEPRECATION_WARNING = (
    "The iteration parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if iteration is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)

class _IterLRScheduler(object):
    def __init__(self, optimizer, last_iteration=-1):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize iteration and base learning rates
        if last_iteration == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_iteration = last_iteration

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def state_dict(self):
"""     """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
"""
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
"""     """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
"""
        self.__dict__.update(state_dict)

    def get_last_lr(self):
"""     """ Return last computed learning rate by current scheduler.
        """
"""
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self, iteration=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            if iteration is None:
                self.last_iteration += 1
                values = self.get_lr()
            else:
                warnings.warn(ITERATION_DEPRECATION_WARNING, DeprecationWarning)
                self.last_iteration = iteration
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

            for param_group, lr in zip(self.optimizer.param_groups, values):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


from bisect import bisect_right

# custom scheduler (same as MultiStepLR)
class SSDIterSchedulerLR(_IterLRScheduler):
    def __init__(self, optimizer, milestones=(40000, 50000, 60000), gamma=0.1, last_iteration=-1, verbose=True):
        super().__init__(optimizer, last_iteration)
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.verbose = verbose

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        if self.last_iteration not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_iteration]
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_iteration)
                for base_lr in self.base_lrs]
"""