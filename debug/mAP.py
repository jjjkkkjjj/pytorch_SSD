from ssd.train.eval import calc_PR, _load_debugfile

import numpy as np

if __name__ == '__main__':
    iou_threshold = 0.5

    VOC_class_labels = ['aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']
    VOC_class_nums = len(VOC_class_labels)

    precisions, recalls = calc_PR(*_load_debugfile(), iou_threshold, VOC_class_nums)

    AP = {}
    for i, label in enumerate(VOC_class_labels):
        if recalls[i] is None:
            AP[label] = 0.
            continue

        # use 11 points
        ap = 0
        for pt in np.arange(0, 1.1, 0.1):
            mask = recalls[i] >= pt
            if mask.sum() == 0:
                continue
            ap += np.max(precisions[i][mask])
        AP[label] = ap / 11.
    AP['mAP'] = np.mean(tuple(AP.values()))
    print(AP)