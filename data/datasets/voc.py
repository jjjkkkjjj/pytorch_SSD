from torch.utils.data import Dataset
from xml.etree import ElementTree as ET
import cv2, os
import numpy as np

from .base import ObjectDetectionDatasetBase, Compose
from .._utils import DATA_ROOT, _get_xml_et_value, _check_ins

VOC_class_labels = ['aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']
VOC_class_nums = len(VOC_class_labels)

VOC2007_ROOT = os.path.join(DATA_ROOT, 'voc/voc2007/trainval/VOCdevkit/VOC2007')
class VOCDatasetBase(ObjectDetectionDatasetBase):
    def __init__(self, voc_dir, focus, ignore=None, transform=None, target_transform=None, augmentation=None, class_labels=None):
        """
        :param voc_dir: str, voc directory path above 'Annotations', 'ImageSets' and 'JPEGImages'
                e.g.) voc_dir = '~~~~/trainval/VOCdevkit/voc2007'
        :param focus: str, image set name. Assign txt file name under 'ImageSets' directory
        :param ignore: target_transforms.Ignore
        :param transform: instance of transforms
        :param target_transform: instance of target_transforms
        :param augmentation:  instance of augmentations
        :param class_labels: None or list or tuple, if it's None use VOC_class_labels
        """
        super().__init__(ignore=ignore, transform=transform, target_transform=target_transform, augmentation=augmentation)

        self._voc_dir = voc_dir
        self._focus = focus
        self._class_labels = _check_ins('class_labels', class_labels, (list, tuple), allow_none=True)
        if self._class_labels is None:
            self._class_labels = VOC_class_labels

        layouttxt_path = os.path.join(self._voc_dir, 'ImageSets', 'Main', self._focus + '.txt')
        if os.path.exists(layouttxt_path):
            with open(layouttxt_path, 'r') as f:
                filenames = f.read().splitlines()
                filenames = [filename.split()[0] for filename in filenames]
                self._annopaths = [os.path.join(self._voc_dir, 'Annotations', '{}.xml'.format(filename)) for filename in filenames]
        else:
            raise FileNotFoundError('layout: {} was invalid arguments'.format(focus))

    @property
    def class_nums(self):
        return len(self._class_labels)
    @property
    def class_labels(self):
        return self._class_labels

    def _jpgpath(self, filename):
        """
        :param filename: path containing .jpg
        :return: path of jpg
        """
        return os.path.join(self._voc_dir, 'JPEGImages', filename)

    def __len__(self):
        return len(self._annopaths)

    """
    Detail of contents in voc > https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5

    VOC bounding box (xmin, ymin, xmax, ymax)
    """
    def _get_image(self, index):
        """
        :param index: int
        :return:
            rgb image(ndarray)
        """
        root = ET.parse(self._annopaths[index]).getroot()
        img = cv2.imread(self._jpgpath(_get_xml_et_value(root, 'filename')))
        # pytorch's image order is rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)

    def _get_bbox_lind(self, index):
        """
        :param index: int
        :return:
            list of bboxes, list of bboxes' label index, list of flags([difficult, truncated,...])
        """
        linds = []
        bboxes = []
        flags = []

        root = ET.parse(self._annopaths[index]).getroot()
        for obj in root.iter('object'):
            linds.append(self._class_labels.index(_get_xml_et_value(obj, 'name')))

            bndbox = obj.find('bndbox')

            # bbox = [xmin, ymin, xmax, ymax]
            bboxes.append([_get_xml_et_value(bndbox, 'xmin', int), _get_xml_et_value(bndbox, 'ymin', int), _get_xml_et_value(bndbox, 'xmax', int), _get_xml_et_value(bndbox, 'ymax', int)])

            flags.append({'difficult': _get_xml_et_value(obj, 'difficult', int) == 1,
                          'truncated': _get_xml_et_value(obj, 'truncated', int) == 1,
                          'occluded': _get_xml_et_value(obj, 'occluded', int) == 1})

        return np.array(bboxes, dtype=np.float32), np.array(linds, dtype=np.float32), flags


class VOC2007Dataset(Compose):
    def __init__(self, **kwargs):
        super().__init__(datasets=(VOC2007_TrainValDataset, VOC2007_TestDataset), **kwargs)


class VOC2007_TrainValDataset(VOCDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(DATA_ROOT + '/voc/voc2007/trainval/VOCdevkit/VOC2007', focus='trainval', **kwargs)


class VOC2007_TestDataset(VOCDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(DATA_ROOT + '/voc/voc2007/test/VOCdevkit/VOC2007', focus='test', **kwargs)


class VOC2012Dataset(Compose):
    def __init__(self, **kwargs):
        super().__init__(datasets=(VOC2012_TrainValDataset, VOC2012_TestDataset), **kwargs)


class VOC2012_TrainValDataset(VOCDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(DATA_ROOT + '/voc/voc2012/trainval/VOCdevkit/VOC2012',
                         focus='trainval', **kwargs)


class VOC2012_TestDataset(VOCDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(DATA_ROOT + '/voc/voc2012/test/VOCdevkit/VOC2012',
                         focus='test', **kwargs)


