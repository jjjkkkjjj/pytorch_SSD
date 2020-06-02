import torch
import os, fnmatch
import numpy as np

def batch_ind_fn(batch):
    """
    concatenate image's index to gt
    e.g.) gts = [[cx, cy, w, h, p_class,...],...] >  ret_gts = [[img's_box number!!!, cx, cy, w, h, p_class,...],...]

    About img's box number...
    e.g.) ret_gts[0] = (2,2,1,3,3,3,2,2,...)
            shortly, box number value is arranged for each box number
    """
    imgs, gts = list(zip(*batch))

    return torch.stack(imgs), gts

def _get_recurrsive_paths(basedir, ext):
    """
    :param basedir:
    :param ext:
    :return: list of path of files including basedir and ext(extension)
    """
    matches = []
    for root, dirnames, filenames in os.walk(basedir):
        for filename in fnmatch.filter(filenames, '*.{}'.format(ext)):
            matches.append(os.path.join(root, filename))
    return sorted(matches)


def _get_xml_et_value(xml_et, key, rettype=str):
    """
    :param xml_et: Elementtree's element
    :param key:
    :param rettype: class, force to convert it from str
    :return: rettype's value
    """
    if isinstance(rettype, str):
        return xml_et.find(key).text
    else:
        return rettype(xml_et.find(key).text)

def _one_hot_encode(indices, class_num):
    """
    :param indices: list of index
    :param class_num:
    :return: ndarray, relu_one-hot vectors
    """
    size = len(indices)
    one_hot = np.zeros((size, class_num))
    one_hot[np.arange(size), indices] = 1
    return one_hot

def _separate_ignore(target_transform):
    """
    Separate Ignore by target_transform
    :param target_transform:
    :return: ignore, target_transform
    """
    if target_transform:
        from .target_transforms import Ignore, Compose
        if isinstance(target_transform, Ignore):
            return target_transform, None

        if not isinstance(target_transform, Compose):
            return None, target_transform

        # search existing target_transforms.Ignore in target_transform
        new_target_transform = []
        ignore = None
        for t in target_transform.target_transforms:
            if isinstance(t, Ignore):
                ignore = t
            else:
                new_target_transform += [t]
        return ignore, Compose(new_target_transform)

    else:
        return None, target_transform


DATA_ROOT = os.path.join(os.path.expanduser('~'), 'data')