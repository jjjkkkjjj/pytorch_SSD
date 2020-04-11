import os, fnmatch
import numpy as np

def get_recurrsive_paths(basedir, ext):
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


def get_xml_et_value(xml_et, key, rettype=str):
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

def one_hot_encode(indices, class_num):
    """
    :param indices: list of index
    :param class_num:
    :return: ndarray, one-hot vectors
    """
    size = len(indices)
    one_hot = np.zeros((size, class_num))
    one_hot[np.arange(size), indices] = 1
    return one_hot