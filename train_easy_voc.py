import argparse



from data.datasets.voc import VOC2007_ROOT, VOC_class_labels

parser = argparse.ArgumentParser(description='Easy training script for VOC style dataset')


# root directory
parser.add_argument('-r', '--dataset_rootdir', default=VOC2007_ROOT,
                    type=str, help='Dataset root directory path')
# focus
parser.add_argument('--focus', default='trainval',
                    type=str, help='Image set name')
# class labels
parser.add_argument('-l', '--labels', default=VOC_class_labels, nargs='+',
                    type=str, help='Dataset class labels')
# ignore difficult
parser.add_argument('-igd', '--ignore_difficult', action='store_true',
                    help='Whether to ignore difficult object')
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
#parser.add_argument('--start_iter', default=0, type=int,
#                    help='Resume training at this iter')
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
from ssd.train import *

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
     target_transforms.OneHot(class_nums=len(args.labels), add_background=True),
     target_transforms.ToTensor()]
)


train_dataset = datasets.VOCDatasetBase(voc_dir=args.dataset_rootdir, focus=args.focus,
                                        ignore=target_transforms.Ignore(difficult=args.ignore_difficult),
                                        transform=transform, target_transform=target_transform, augmentation=augmentation,
                                        class_labels=args.labels)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          collate_fn=utils.batch_ind_fn, num_workers=args.num_workers, pin_memory=True)

logging.info('Dataset info:'
             '\nroot dir: {},'
             '\nfocus: {},'
             '\nlabels:{}'
             '\nignore difficult object: {}'
             '\naugmentation: {}'
             '\nbatch size: {}'
             '\nnum_workers: {}\n'.format(args.dataset_rootdir, args.focus, args.labels,
                                         args.ignore_difficult, not args.no_augmentation,
                                        args.batch_size, args.num_workers))


#### model ####
if args.model == 'SSD300':
    model = SSD300(class_labels=args.labels, batch_norm=args.batch_norm)
elif args.model == 'SSD512': # SSD512
    raise NotImplementedError('Unsupported yet!')
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

trainer.train(args.max_iteration, train_loader)