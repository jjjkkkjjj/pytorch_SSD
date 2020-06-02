from numpy import random
import numpy as np
import logging

from ._utils import decision
from ssd.core.boxes.utils import iou_numpy, centroids2minmax_numpy, minmax2centroids_numpy
from .base import Compose

class RandomExpand(object):
    def __init__(self, filled_rgb_mean=(103.939, 116.779, 123.68), rmin=1, rmax=4, p=0.5):
        self.filled_rgb_mean = filled_rgb_mean
        self.ratio_min = rmin
        self.ratio_max = rmax
        self.p = p

        assert self.ratio_min >= 0, "must be more than 0"
        assert self.ratio_max >= self.ratio_min, "must be more than factor min"

    def __call__(self, img, bboxes, labels, flags):
        # IMPORTANT: img = rgb order, bboxes: minmax coordinates with PERCENT
        if decision(self.p):
            h, w, c = img.shape
            # get ratio randomly
            ratio = random.uniform(self.ratio_min, self.ratio_max)

            new_h = int(h*ratio)
            new_w = int(w*ratio)

            # get top left coordinates of original image randomly
            topleft_x = int(random.uniform(0, new_w - w))
            topleft_y = int(random.uniform(0, new_h - h))

            # filled with normalized mean value
            new_img = np.ones((new_h, new_w, c)) * np.broadcast_to(self.filled_rgb_mean, shape=(1, 1, c))

            # put original image to selected topleft coordinates
            new_img[topleft_y:topleft_y+h, topleft_x:topleft_x+w] = img
            img = new_img

            # convert box coordinates
            # bbox shape = (*, 4=(xmin, ymin, xmax, ymax))
            # reconstruct original coordinates
            bboxes[:, 0::2] *= float(w)
            bboxes[:, 1::2] *= float(h)
            # move new position
            bboxes[:, 0::2] += topleft_x
            bboxes[:, 1::2] += topleft_y
            # to percent
            bboxes[:, 0::2] /= float(new_w)
            bboxes[:, 1::2] /= float(new_h)

        return img, bboxes, labels, flags


class _SampledPatchOp(object):
    class UnSatisfy(Exception):
        pass

class EntireSample(_SampledPatchOp):
    def __call__(self, img, bboxes, labels, flags):
        return img, bboxes, labels, flags

class RandomIoUSampledPatch(_SampledPatchOp):
    def __init__(self, iou_min=None, iou_max=None, ar_min=0.5, ar_max=2):
        """
        :param iou_min: float or None, if it's None, set iou_min as -inf
        :param iou_max: float or None, if it's None, set iou_max as inf
        """
        self.iou_min = iou_min if iou_min else float('-inf')
        self.iou_max = iou_max if iou_max else float('inf')
        self.aspect_ration_min = ar_min
        self.aspect_ration_max = ar_max

    def __call__(self, img, bboxes, labels, flags):
        # IMPORTANT: img = rgb order, bboxes: minmax coordinates with PERCENT
        h, w, _ = img.shape

        ret_img = img.copy()
        ret_bboxes = bboxes.copy()

        # get patch width and height, and aspect ratio randomly
        patch_w = random.randint(int(0.3 * w), w)
        patch_h = random.randint(int(0.3 * h), h)
        aspect_ratio = patch_h / float(patch_w)

        # aspect ratio constraint b/t .5 & 2
        if not (aspect_ratio >= 0.5 and aspect_ratio <= 2):
            raise _SampledPatchOp.UnSatisfy
        #aspect_ratio = random.uniform(self.aspect_ration_min, self.aspect_ration_max)

        #patch_h, patch_w = int(aspect_ratio*h), int(aspect_ratio*w)
        patch_topleft_x = random.randint(w - patch_w)
        patch_topleft_y = random.randint(h - patch_h)
        # shape = (1, 4)
        patch = np.array((patch_topleft_x, patch_topleft_y,
                          patch_topleft_x + patch_w, patch_topleft_y + patch_h))
        patch = np.expand_dims(patch, 0)

        # IoU
        overlaps = iou_numpy(bboxes, patch)
        if overlaps.min() < self.iou_min and overlaps.max() > self.iou_max:
            raise _SampledPatchOp.UnSatisfy
            #return None

        # cut patch
        ret_img = ret_img[patch_topleft_y:patch_topleft_y+patch_h, patch_topleft_x:patch_topleft_x+patch_w]

        # reconstruct box coordinates
        ret_bboxes[:, 0::2] *= float(w)
        ret_bboxes[:, 1::2] *= float(h)

        # convert minmax to centroids coordinates of bboxes
        # shape = (*, 4=(cx, cy, w, h))
        centroids_boxes = minmax2centroids_numpy(ret_bboxes)

        # check if centroids of boxes is in patch
        mask_box = (centroids_boxes[:, 0] > patch_topleft_x) * (centroids_boxes[:, 0] < patch_topleft_x+patch_w) *\
                   (centroids_boxes[:, 1] > patch_topleft_y) * (centroids_boxes[:, 1] < patch_topleft_y+patch_h)
        if not mask_box.any():
            raise _SampledPatchOp.UnSatisfy
            #return None

        # filtered out the boxes with unsatisfied above condition
        ret_bboxes = ret_bboxes[mask_box, :].copy()
        ret_labels = labels[mask_box]

        # adjust boxes within patch
        ret_bboxes[:, :2] = np.maximum(ret_bboxes[:, :2], patch[:, :2])
        ret_bboxes[:, 2:] = np.minimum(ret_bboxes[:, 2:], patch[:, 2:])

        # move new position
        ret_bboxes[:, :2] -= patch[:, :2]
        ret_bboxes[:, 2:] -= patch[:, :2]

        # to percent
        ret_bboxes[:, 0::2] /= float(patch_w)
        ret_bboxes[:, 1::2] /= float(patch_h)

        return ret_img, ret_bboxes, ret_labels, flags

class RandomSampledPatch(RandomIoUSampledPatch):
    def __init__(self):
        super().__init__(None, None)


class RandomSampled(object):
    def __init__(self, options=(
            EntireSample(),
            RandomIoUSampledPatch(0.1, None),
            RandomIoUSampledPatch(0.3, None),
            RandomIoUSampledPatch(0.5, None),
            RandomIoUSampledPatch(0.7, None),
            RandomIoUSampledPatch(0.9, None),
            RandomSampledPatch()
            ), max_iteration=20):

        # check argument is proper
        for op in options:
            if not isinstance(op, _SampledPatchOp):
                raise ValueError('All of option\'s element must be inherited to _SampledPatchOp')

        if not any([isinstance(op, EntireSample) for op in options]):
            logging.warning("Option does not contain entire sample. Could not return value in worst case")

        self.options = options
        self.max_iteration = max_iteration

    def __call__(self, img, bboxes, labels, flags):
        import time
        s = time.time()
        while True:
            # select option randomly
            op = random.choice(self.options)

            if isinstance(op, EntireSample):
                return op(img, bboxes, labels, flags)

            for _ in range(self.max_iteration):
                try:
                    return op(img, bboxes, labels, flags)
                except _SampledPatchOp.UnSatisfy:
                    continue
                """
                ret = op(img, bboxes, labels, flags)
                if ret:
                    print(time.time()-s)
                    return ret
                """

class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes, labels, flags):
        if decision(self.p):
            _, w_, _ = img.shape
            """
            copy because ->>>>
            ValueError: some of the strides of a given numpy array are negative.
             This is currently not supported, but will be added in future releases.
            """
            img = img[:, ::-1].copy()

            ret_bboxes = bboxes.copy()
            ret_bboxes[:, 0::2] = 1 - ret_bboxes[:, 2::-2]
            bboxes = ret_bboxes.clip(min=0, max=1)

        return img, bboxes, labels, flags


class GeometricDistortions(Compose):
    def __init__(self):
        gmdists = [
            RandomExpand(),
            RandomSampled(),
            RandomFlip()
        ]
        super().__init__(gmdists)