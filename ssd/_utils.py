import os

def weights_path(_file_, _root_num, dirname):
    basepath = os.path.dirname(_file_)
    backs = [".."]*_root_num
    model_dir = os.path.abspath(os.path.join(basepath, *backs, dirname))
    return model_dir

def check_instance(name, val, ins):
    assert isinstance(val, ins), '{} must be {}'.format(name, ins.__name__)
    return val

def _check_ins(name, val, cls, allow_none=False):
    if allow_none and val is None:
        return val

    if not isinstance(val, cls):
        raise ValueError('Argument \'{}\' must be {}, but got {}'.format(name, cls.__name__, val.__name__))
    return val