choices = ['voc2007_trainval', 'voc2007_test', 'voc2012_trainval', 'voc2012_test', 'coco2014_trainval', 'coco2014_test']
__all__ = choices
import os
import pycurl
import tarfile, zipfile
import glob
import logging

class _Downloader:
    def __init__(self, url, compress_ext='tar'):
        self.url = url

        _compress_exts = ['tar', 'zip']
        if not compress_ext in _compress_exts:
            raise ValueError("Invalid argument, select proper extension from {}, but got {}".format(_compress_exts, compress_ext))
        self.compress_ext = compress_ext

    def run(self, out_base_dir, dirname, remove_comp_file=True):
        out_dir = os.path.join(out_base_dir, dirname)

        if len(glob.glob(os.path.join(out_dir, '*'))) > 0:
            logging.warning('dataset may be already downloaded. If you haven\'t done yet, remove \"{}\" directory'.format(out_base_dir))
            return

        curl = pycurl.Curl()
        curl.setopt(pycurl.URL, self.url)
        # allow redirect
        curl.setopt(pycurl.FOLLOWLOCATION, True)
        # show progress
        curl.setopt(pycurl.NOPROGRESS, False)

        os.makedirs(out_dir, exist_ok=True)

        dstpath = os.path.join(out_base_dir, 'tmp.{}'.format(self.compress_ext))

        with open(dstpath, 'wb') as f:
            curl.setopt(pycurl.WRITEFUNCTION, f.write)
            curl.perform()

        curl.close()


        # extract
        if self.compress_ext == 'tar':
            with tarfile.open(dstpath) as tar_f:
                tar_f.extractall(out_dir)
        elif self.compress_ext == 'zip':
            with zipfile.ZipFile(dstpath, 'r') as zip_f:
                zip_f.extractall(out_dir)
        else:
            assert False, "Bug occurred"

        if remove_comp_file:
            # remove tmp.*
            os.remove(dstpath)


from .utils import DATA_ROOT

def voc2007_trainval():
    trainval_downloader = _Downloader('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar')
    trainval_downloader.run(DATA_ROOT + '/voc/voc2007', 'trainval')

def voc2007_test():
    test_downloader = _Downloader('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar')
    test_downloader.run(DATA_ROOT + '/voc/voc2007', 'test')

def voc2012_trainval():
    trainval_downloader = _Downloader('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar')
    trainval_downloader.run(DATA_ROOT + '/voc/voc2012', 'trainval')

def voc2012_test():
    test_downloader = _Downloader('http://pjreddie.com/media/files/VOC2012test.tar')
    test_downloader.run(DATA_ROOT + '/voc/voc2012', 'test')

def coco2014_trainval():
    # get images
    train_downloader = _Downloader('http://images.cocodataset.org/zips/train2014.zip', 'zip')
    train_downloader.run(DATA_ROOT + '/coco/coco2014/images', 'train', remove_comp_file=False)

    val_downloader = _Downloader('http://images.cocodataset.org/zips/val2014.zip', 'zip')
    val_downloader.run(DATA_ROOT + '/coco/coco2014/images', 'val', remove_comp_file=False)

    # annotations
    trainval_downloader = _Downloader('http://images.cocodataset.org/annotations/annotations_trainval2014.zip', 'zip')
    trainval_downloader.run(DATA_ROOT + '/coco/coco2014/annotations', 'train', remove_comp_file=False)

def coco2014_test():
    # get images
    test_downloader = _Downloader('http://images.cocodataset.org/zips/test2014.zip', 'zip')
    test_downloader.run(DATA_ROOT + '/coco/coco2014/images', 'test', remove_comp_file=False)

    # annotations
    test_downloader = _Downloader('http://images.cocodataset.org/annotations/annotations_trainval2014.zip', 'zip')
    test_downloader.run(DATA_ROOT + '/coco/coco2014/annotations', 'test', remove_comp_file=False)
