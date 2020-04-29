

"""
IMPORTANT: augmentation will be ran before transform and target_transform
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

class RandomPatch(object):
    def __init__(self):
        pass

    def __call__(self, img, bboxes, labels, flags):

        return img, bboxes, labels, flags

class JaccordPatch(object):
    def __init__(self):
        pass

    def __call__(self, img, bboxes, labels, flags):

        return img, bboxes, labels, flags

class Flip(object):
    def __init__(self):
        pass

    def __call__(self, img, bboxes, labels, flags):

        return img, bboxes, labels, flags