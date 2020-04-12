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

    ret_gts = []
    for ind, gt in enumerate(gts):
        box_num = len(gt)
        ret_gt = np.zeros((box_num, gt.shape[1] + 1))
        ret_gt[:, 1:] = gt
        ret_gt[:, 0] = box_num # concatenate image's index
        ret_gts.append(ret_gt)

    imgs = torch.stack(imgs)
    ret_gts = torch.Tensor(np.concatenate(ret_gts))

    return imgs, ret_gts

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
    :param element: Elementtree's element
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
    :return: ndarray, one-hot vectors
    """
    size = len(indices)
    one_hot = np.zeros((size, class_num))
    one_hot[np.arange(size), indices] = 1
    return one_hot