__all__ = ['voc2007_trainval', 'voc2007_test', 'voc2012_trainval', 'voc2012_test']

import os
import pycurl
import tarfile
import glob
import logging

class _Downloader:
    def __init__(self, url):
        self.url = url

    def run(self, out_base_dir, dirname):
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

        tarpath = os.path.join(out_base_dir, 'tmp.tar')

        with open(tarpath, 'wb') as f:
            curl.setopt(pycurl.WRITEFUNCTION, f.write)
            curl.perform()

        curl.close()


        # extract tar
        with tarfile.open(tarpath) as tar:
            tar.extractall(out_dir)

        # remove tmp.tar
        os.remove(tarpath)


from .utils import _thisdir

def voc2007_trainval():
    trainval_downloader = _Downloader('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar')
    trainval_downloader.run(_thisdir + '/voc/voc2007', 'trainval')

def voc2007_test():
    test_downloader = _Downloader('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar')
    test_downloader.run(_thisdir + '/voc/voc2007', 'test')

def voc2012_trainval():
    trainval_downloader = _Downloader('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar')
    trainval_downloader.run(_thisdir + '/voc/voc2012', 'trainval')

def voc2012_test():
    test_downloader = _Downloader('http://pjreddie.com/media/files/VOC2012test.tar')
    test_downloader.run(_thisdir + '/voc/voc2012', 'test')

