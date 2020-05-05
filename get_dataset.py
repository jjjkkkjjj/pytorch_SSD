from data.downloader import *
from data.downloader import choices

import argparse

parser = argparse.ArgumentParser(description='Download datasets')
parser.add_argument('--datasets', help='select datasets from {}'.format(choices),
                    choices=choices, nargs='*', required=True)
args = parser.parse_args()

if __name__ == '__main__':
    if 'voc2007_trainval' in args.datasets:
        voc2007_trainval()
    if 'voc2007_test' in args.datasets:
        voc2007_test()
    if 'voc2012_trainval' in args.datasets:
        voc2012_trainval()
    if 'voc2012_test' in args.datasets:
        voc2012_test()
    if 'coco2014_trainval' in args.datasets:
        coco2014_trainval()
    if 'coco2014_test' in args.datasets:
        coco2014_test()
