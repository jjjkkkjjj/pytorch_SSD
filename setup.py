from distutils.core import setup

setup(
    name='pytorch_SSD',
    version='0.0.2',
    packages=['ssd', 'ssd.core', 'ssd.core.boxes', 'ssd.train', 'ssd.models', 'ssd_data', 'ssd_data.datasets',
              'ssd_data.augmentations'],
    url='https://github.com/jjjkkkjjj/pytorch_SSD',
    license='MIT',
    author='jjjkkkjjj',
    author_email='',
    description='Single Shot Multibox Detector Implementation with PyTorch.'
)
