class RandomPatch(object):
    def __init__(self):
        pass

    def __call__(self, img, bboxes, labels, flags):

        return img, bboxes, labels, flags

class RandomJaccordPatch(object):
    def __init__(self):
        pass

    def __call__(self, img, bboxes, labels, flags):

        return img, bboxes, labels, flags

class RandomFlip(object):
    def __init__(self):
        pass

    def __call__(self, img, bboxes, labels, flags):

        return img, bboxes, labels, flags