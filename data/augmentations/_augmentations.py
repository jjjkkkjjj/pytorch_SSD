from .photometrics import *
from .geometrics import *
from .utils import decision
"""
IMPORTANT: augmentation will be ran before transform and target_transform

ref > http://www.telesens.co/2018/06/28/data-augmentation-in-ssd/
"""
class Compose(object):
    def __init__(self, augmentaions):
        self.augmentaions = augmentaions

    def __call__(self, img, bboxes, labels, flags):
        for t in self.augmentaions:
            img, bboxes, labels, flags = t(img, bboxes, labels, flags)
        return img, bboxes, labels, flags

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.augmentaions:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class AugmentationOriginal(Compose):
    def __init__(self):
        augmentations = [
            PhotometricDistortions(),
            GeometricDistortions()
        ]
        super().__init__(augmentations)



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

    def __call__(self, img, bboxes, labels, flags):
        img, bboxes, labels, flags = self.brigtness(img, bboxes, labels, flags)

        if decision(self.p): # random contrast first
            img, bboxes, labels, flags = self.cotrast(img, bboxes, labels, flags)
            img, bboxes, labels, flags = super().__call__(img, bboxes, labels, flags)

        else: # random contrast last
            img, bboxes, labels, flags = super().__call__(img, bboxes, labels, flags)
            img, bboxes, labels, flags = self.cotrast(img, bboxes, labels, flags)

        img, bboxes, labels, flags = self.lightingnoise(img, bboxes, labels, flags)

        return img, bboxes, labels, flags

class GeometricDistortions(Compose):
    def __init__(self):
        gmdists = [
            RandomExpand(),
            RandomSampled(),
            RandomFlip()
        ]
        super().__init__(gmdists)
