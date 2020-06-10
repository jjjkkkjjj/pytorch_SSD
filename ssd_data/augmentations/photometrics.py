from ._utils import decision
from numpy import random
import numpy as np
import cv2
from itertools import permutations

from .base import Compose

class RandomBrightness(object):
    def __init__(self, dmin=-32, dmax=32, p=0.5):
        self.delta_min = dmin
        self.delta_max = dmax
        self.p = p

        assert abs(self.delta_min) >= 0 and abs(self.delta_min) < 256, "must be range between -255 and 255"
        assert abs(self.delta_max) >= 0 and abs(self.delta_max) < 256, "must be range between -255 and 255"
        assert self.delta_max >= self.delta_min, "must be more than delta min"

    def __call__(self, img, bboxes, labels, flags, *args):
        if decision(self.p):
            # get delta randomly between delta_min and delta_max
            delta = random.uniform(self.delta_min, self.delta_max)
            img += delta
            img = np.clip(img, a_min=0, a_max=255)

        return img, bboxes, labels, flags, args

class RandomContrast(object):
    def __init__(self, fmin=0.5, fmax=1.5, p=0.5):
        self.factor_min = fmin
        self.factor_max = fmax
        self.p = p

        assert self.factor_min >= 0, "must be more than 0"
        assert self.factor_max >= self.factor_min, "must be more than factor min"

    def __call__(self, img, bboxes, labels, flags, *args):
        if decision(self.p):
            # get delta randomly between delta_min and delta_max
            factor = random.uniform(self.factor_min, self.factor_max)
            img *= factor
            img = np.clip(img, a_min=0, a_max=255)

        return img, bboxes, labels, flags, args

class RandomHue(object):
    def __init__(self, dmin=-18, dmax=18, p=0.5):
        self.delta_min = dmin
        self.delta_max = dmax
        self.p = p

        assert abs(self.delta_min) >= 0 and abs(self.delta_min) < 180, "must be range between -179 and 179"
        assert abs(self.delta_max) >= 0 and abs(self.delta_max) < 180, "must be range between -179 and 179"
        assert self.delta_max >= self.delta_min, "must be more than delta min"

    def __call__(self, img, bboxes, labels, flags, *args):
        if decision(self.p):
            # get delta randomly between delta_min and delta_max
            delta = random.uniform(self.delta_min, self.delta_max)
            img[:, :, 0] += delta

            # clip 0 to 180, note that opencv's hue range is [0, 180]
            over_mask = img[:, :, 0] >= 180
            img[over_mask, 0] -= 180

            under_mask = img[:, :, 0] < 0
            img[under_mask, 0] += 180

        return img, bboxes, labels, flags, args

class RandomSaturation(object):
    def __init__(self, fmin=0.5, fmax=1.5, p=0.5):
        self.factor_min = fmin
        self.factor_max = fmax
        self.p = p

        assert self.factor_min >= 0, "must be more than 0"
        assert self.factor_max >= self.factor_min, "must be more than factor min"

    def __call__(self, img, bboxes, labels, flags, *args):
        if decision(self.p):
            # get delta randomly between delta_min and delta_max
            factor = random.uniform(self.factor_min, self.factor_max)
            img[:, :, 1] *= factor
            img = np.clip(img, a_min=0, a_max=255)

        return img, bboxes, labels, flags, args

class RandomLightingNoise(object):
    def __init__(self, perms=None, p=0.5):
        self.p = p
        if perms:
            self.permutations = perms
        else:
            self.permutations = tuple(permutations([0, 1, 2]))

    def __call__(self, img, bboxes, labels, flags, *args):
        if decision(self.p):
            # get transposed indices randomly
            index = random.randint(0, len(self.permutations))
            t = SwapChannels(self.permutations[index])
            img, bboxes, labels, flags, args = t(img, bboxes, labels, flags, *args)

        return img, bboxes, labels, flags, args

class SwapChannels(object):
    def __init__(self, trans_indices):
        self.trans_indices = trans_indices

    def __call__(self, img, bboxes, labels, flags, *args):
        return img[:, :, self.trans_indices], bboxes, labels, flags, args


class ConvertImgOrder(object):
    def __init__(self, src='rgb', dst='hsv'):
        self.src_order = src.upper()
        self.dst_order = dst.upper()

    def __call__(self, img, bboxes, labels, flags, *args):
        try:
            img = cv2.cvtColor(img, eval('cv2.COLOR_{}2{}'.format(self.src_order, self.dst_order)))
        except:
            raise ValueError('Invalid src:{} or dst:{}'.format(self.src_order, self.dst_order))

        return img, bboxes, labels, flags, args


class PhotometricDistortions(Compose):
    def __init__(self, p=0.5):
        self.p = p

        self.brigtness = RandomBrightness()
        self.cotrast = RandomContrast()
        self.lightingnoise = RandomLightingNoise()

        pmdists = [
            ConvertImgOrder(src='rgb', dst='hsv'),
            RandomSaturation(),
            RandomHue(),
            ConvertImgOrder(src='hsv', dst='rgb')
        ]
        super().__init__(pmdists)

    def __call__(self, img, bboxes, labels, flags, *args):
        img, bboxes, labels, flags, args = self.brigtness(img, bboxes, labels, flags, *args)

        if decision(self.p): # random contrast first
            img, bboxes, labels, flags, args = self.cotrast(img, bboxes, labels, flags, *args)
            img, bboxes, labels, flags, args = super().__call__(img, bboxes, labels, flags, *args)

        else: # random contrast last
            img, bboxes, labels, flags, args = super().__call__(img, bboxes, labels, flags, *args)
            img, bboxes, labels, flags, args = self.cotrast(img, bboxes, labels, flags, *args)

        img, bboxes, labels, flags, args = self.lightingnoise(img, bboxes, labels, flags, *args)

        return img, bboxes, labels, flags, args