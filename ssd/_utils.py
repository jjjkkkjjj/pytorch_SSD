import os
import torch

def weights_path(_file_, _root_num, dirname):
    basepath = os.path.dirname(_file_)
    backs = [".."]*_root_num
    model_dir = os.path.abspath(os.path.join(basepath, *backs, dirname))
    return model_dir


def _check_ins(name, val, cls, allow_none=False, default=None):
    if allow_none and val is None:
        return default

    if not isinstance(val, cls):
        raise ValueError('Argument \'{}\' must be {}, but got {}'.format(name, cls.__name__, type(val).__name__))
    return val

def _check_norm(name, val):
    if isinstance(val, (float, int)):
        val = torch.tensor([float(val)], requires_grad=False)
    elif isinstance(val, (list, tuple)):
        val = torch.tensor(val, requires_grad=False).float()
    elif not isinstance(val, torch.Tensor):
        raise ValueError('{} must be int, float, list, tuple, Tensor, but got {}'.format(name, type(val).__name__))

    return val