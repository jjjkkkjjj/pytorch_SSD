import argparse
import os

from data.datasets.voc import VOC2007_ROOT, VOC_class_labels
from data.datasets.coco import COCO2014_ROOT, COCO_class_labels

voc_rootdir_default = [VOC2007_ROOT]
coco_rootdir_default = [os.path.join(COCO2014_ROOT, 'trainval')]

voc_focus_default = ['trainval']
coco_focus_default = ['train2014']

parser = argparse.ArgumentParser(description='Easy training script for VOC or COCO style dataset')

# dataset type
# required
parser.add_argument('dataset_type', choices=['VOC', 'COCO'],
                    type=str, help='Dataset type')
# root directory
parser.add_argument('-r', '--dataset_rootdir', default=None, nargs='+',
                    type=str, help='Dataset root directory path.\n'
                                   'If dataset type is \'VOC\', Default is;\n\'{}\'\n\n'
                                   'If dataset type is \'COCO\', Default is;\n\'{}\''.format(voc_rootdir_default, coco_rootdir_default))
# focus
parser.add_argument('--focus', default=None, nargs='+',
                    type=str, help='Image set name.\n'
                                   'If dataset type is \'VOC\', Default is;\n\'{}\'\n\n'
                                   'if dataset type is \'COCO\', Default is;\n\'{}\''.format(voc_focus_default, coco_focus_default))
# class labels
parser.add_argument('-l', '--labels', default=None, nargs='+',
                    type=str, help='Dataset class labels.\n'
                                   'If dataset type is \'VOC\', Default is;\n\'{}\'\n\n'
                                   'If dataset type is \'COCO\', Default is;\n\'{}\''.format(VOC_class_labels, COCO_class_labels)
                    )
# ignore difficult
parser.add_argument('-ig', '--ignore', choices=['difficult', 'truncated', 'occluded', 'iscrowd'], nargs='*',
                    type=str, help='Whether to ignore object')
# model
parser.add_argument('-m', '--model', default='SSD300', choices=['SSD300', 'SSD512'],
                    help='Trained model')
# model name
parser.add_argument('-n', '--model_name', default='SSD300', type=str,
                    help='Model name, which will be used as save name')
# batch normalization
parser.add_argument('-bn', '--batch_norm', action='store_true',
                    help='Whether to construct model with batch normalization')
# pretrained weight
parser.add_argument('-w', '--weights_path', type=str,
                    help='Pre-trained weights path. Default is pytorch\'s pre-trained one for vgg')
# batch size
parser.add_argument('-bs', '--batch_size', default=32, type=int,
                    help='Batch size')
# num_workers in DataLoader
parser.add_argument('-nw', '--num_workers', default=4, type=int,
                    help='Number of workers used in DataLoader')
# device
parser.add_argument('-d', '--device', default='cuda', choices=['cpu', 'cuda'], type=str,
                    help='Device for Tensor')
#parser.add_argument('--resume', default=None, type=str,
#                    help='Checkpoint state_dict file to resume training from')
# start iteration
parser.add_argument('-si', '--start_iteration', default=0, type=int,
                    help='Resume training at this iteration')
# augmentation
parser.add_argument('-na', '--no_augmentation', action='store_false', default=False,
                    help='Whether to do augmentation to your dataset')
# optimizer
parser.add_argument('-optimizer', default='SGD', choices=['SGD', 'Adam'],
                    type=str, help='Optimizer for training')
# learning rate
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                    help='Initial learning rate')
# momentum
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for Optimizer')
# weight decay
parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
# MultiStepLR gamma
parser.add_argument('--steplr_gamma', default=0.1, type=float,
                    help='Gamma for stepLR')
# MultiStepLR milestones
parser.add_argument('--steplr_milestones', default=[40000, 50000], type=int, nargs='+',
                    help='Milestones for stepLR')
# attr = list
# final iteration
parser.add_argument('-mi', '--max_iteration', default=60000, type=int,
                    help='')
# Checkpoints interval
parser.add_argument('-ci', '--checkpoints_interval', default=5000, type=int,
                    help='Checkpoints interval')
# loss alpha
parser.add_argument('--loss_alpha', default=1.0, type=float,
                    help='Loss\'s alpha')
args = parser.parse_args()

import torch
import logging
logging.basicConfig(level=logging.INFO)
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

from data import transforms, target_transforms, augmentations, utils, datasets
from ssd.models.ssd300 import SSD300
from ssd.models.ssd512 import SSD512
from ssd.train import *


rootdir = args.dataset_rootdir
if rootdir is None:
    if args.dataset_type == 'VOC':
        rootdir = voc_rootdir_default
    else:
        rootdir = coco_rootdir_default

class_labels = args.labels
if class_labels is None:
    if args.dataset_type == 'VOC':
        class_labels = VOC_class_labels
    else:
        class_labels = COCO_class_labels

focus = args.focus
if focus is None:
    if args.dataset_type == 'VOC':
        focus = voc_focus_default
    else:
        focus = coco_focus_default

if torch.cuda.is_available():
    if args.device != 'cuda':
        logging.warning('You can use CUDA device but you didn\'t set CUDA device.'
                        ' Run with \'-d cuda\' or \'--device cuda\'')
device = torch.device(args.device)

#### dataset ####
augmentation = None if args.no_augmentation else augmentations.AugmentationOriginal()

if args.model == 'SSD300':
    size = (300, 300)
elif args.model == 'SSD512': # SSD512
    size = (512, 512)
else:
    assert False, "Invalid model name"

transform = transforms.Compose(
    [transforms.Resize(size),
     transforms.ToTensor(),
     transforms.Normalize(rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))]
)
target_transform = target_transforms.Compose(
    [target_transforms.ToCentroids(),
     target_transforms.OneHot(class_nums=len(class_labels), add_background=True),
     target_transforms.ToTensor()]
)

if args.ignore:
    kwargs = {key: True for key in args.ignore}
    ignore = target_transforms.Ignore(**kwargs)
else:
    ignore = None

if args.dataset_type == 'VOC':
    train_dataset = datasets.VOCMultiDatasetBase(voc_dir=rootdir, focus=focus, ignore=ignore,
                                                 transform=transform, target_transform=target_transform, augmentation=augmentation,
                                                 class_labels=class_labels)
else:
    train_dataset = datasets.COCOMultiDatasetBase(coco_dir=rootdir, focus=focus, ignore=ignore,
                                                  transform=transform, target_transform=target_transform, augmentation=augmentation,
                                                  class_labels=class_labels)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          collate_fn=utils.batch_ind_fn, num_workers=args.num_workers, pin_memory=True)

logging.info('Dataset info:'
             '\nroot dir: {},'
             '\nfocus: {},'
             '\nlabels:{}'
             '\nignore object: {}'
             '\naugmentation: {}'
             '\nbatch size: {}'
             '\nnum_workers: {}\n'.format(rootdir, focus, class_labels,
                                          args.ignore, not args.no_augmentation,
                                          args.batch_size, args.num_workers))


#### model ####
if args.model == 'SSD300':
    model = SSD300(class_labels=class_labels, batch_norm=args.batch_norm).to(device)
elif args.model == 'SSD512': # SSD512
    model = SSD512(class_labels=class_labels, batch_norm=args.batch_norm).to(device)
else:
    assert False, "Invalid model name"

if args.weights_path is None:
    model.load_vgg_weights()
else:
    model.load_weights(args.weights_path)

logging.info(model)

### train info ###
if args.optimizer == 'SGD':
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    logging.info('Optimizer Info:'
                 '\nOptimizer: {}'
                 '\nlearning rate: {}, Momentum: {}, Weight decay: {}\n'.format(args.optimizer, args.learning_rate, args.momentum, args.weight_decay))
elif args.optimizer == 'Adam':
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    logging.info('Optimizer Info:'
                 '\nOptimizer: {}'
                 '\nlearning rate: {}, Weight decay: {}\n'.format(args.optimizer, args.learning_rate, args.weight_decay))
else:
    assert False, "Invalid optimizer"

iter_scheduler = SSDIterMultiStepLR(optimizer, milestones=args.steplr_milestones, gamma=args.steplr_gamma)
logging.info('Multi Step Info:'
             '\nmilestones: {}'
             '\ngamma: {}\n'.format(args.steplr_milestones, args.steplr_gamma))

save_manager = SaveManager(modelname=args.model_name, interval=args.checkpoints_interval, max_checkpoints=15)
log_manager = LogManager(interval=10, save_manager=save_manager, loss_interval=10, live_graph=None)
trainer = TrainLogger(model, loss_func=SSDLoss(alpha=args.loss_alpha), optimizer=optimizer, scheduler=iter_scheduler,
                      log_manager=log_manager)

logging.info('Save Info:'
             '\nfilename: {}'
             '\ncheckpoints interval: {}\n'.format(args.model_name, args.checkpoints_interval))

logging.info('Start Training\n\n')

trainer.train(args.max_iteration, train_loader, start_iteration=args.start_iteration)