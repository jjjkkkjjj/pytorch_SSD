
"""
IMPORTANT: augmentation will be ran before transform and target_transform

ref > http://www.telesens.co/2018/06/28/data-augmentation-in-ssd/
"""
class Compose(object):
    def __init__(self, augmentaions):
        self.augmentaions = augmentaions

    def __call__(self, img, bboxes, labels, flags, *args):
        for t in self.augmentaions:
            img, bboxes, labels, flags, args = t(img, bboxes, labels, flags, *args)
        return img, bboxes, labels, flags, args

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.augmentaions:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class AugmentationOriginal(Compose):
    def __init__(self):
        from .photometrics import PhotometricDistortions
        from .geometrics import GeometricDistortions
        augmentations = [
            PhotometricDistortions(),
            GeometricDistortions()
        ]
        super().__init__(augmentations)



