from datasets.voc2007 import VOC2007Dataset, VOCDatasetTransform

if __name__ == '__main__':
    dataset = VOC2007Dataset(transform=VOCDatasetTransform())
    dataset[0]