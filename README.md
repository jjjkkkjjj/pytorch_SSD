# PyTorch SSD
The implementation of SSD (Single shot detector) in PyTorch.

![result img](assets/coco_testimg-result.jpg?raw=true "result img")

![result2 img](assets/coco_testimg-result2.jpg?raw=true "result2 img")

# TODO

- [x] Implement SSD300
- [x] Implement SSD300 with batch normalization
- [x] Implement SSD512
- [x] Implement SSD512 with batch normalization
- [x] Visualize inference result
- [x] Arg parse (easy training)
- [ ] Share pre-trained weights
  →SSD300's model has shared partially!
- [x] Well-introduction?
- [x] Support COCO Dataset
- [x] Support Custom Dataset
- [x] Speed up
- [x] mAP (I have no confidence...)

# Requirements and Settings

- Anaconda

  ```bash
  conda install -c anaconda pycurl
  conda install -c pytorch pytorch
  conda install -c conda-forge numpy opencv ffmpeg scipy jupyter_contrib_nbextensions jupyter_nbextensions_configurator pycocotools
  ```

- pip (optional)

  ```bash
  pip install git+https://github.com/jjjkkkjjj/pytorch_SSD.git
  ```

  

- Jupyter

  ```bash
  jupyter notebook
  ```

  ![nbextensions](https://user-images.githubusercontent.com/16914891/80898698-67145d80-8d41-11ea-92c3-76c3791bdb9f.png)

  

# How to start

## Get VOC and COCO Dataset

- You can download VOC2007-trainval, VOC2007-test, VOC2012-trainval, VOC2012-test, COCO2014-trainval and COCO2014-test dataset following command;

  ```bash
  python get_dataset.py --datasets [{dataset name} {dataset name}...]
  ```

  `{dataset name}` is;

  - `voc2007_trainval`
  - `voc2007_test`
  - `voc2012_trainval`
  - `voc2012_test`
  - `coco2014_trainval` 
  - `coco2017_trainval`

## Easy training

You can train (**your**) voc or coco style dataset easily when you use `easy_train.py`!

Example;

```bash
python easy_train.py VOC -r {your-voc-style-dataset-path} --focus trainval -l ball person -lr 0.003
```

or

```bash
python easy_train.py COCO -r {your-coco-style-dataset-path} --focus train2012 -l ball person -lr 0.003
```

```bash
usage: easy_train.py [-h] [-r DATASET_ROOTDIR [DATASET_ROOTDIR ...]]
                     [--focus FOCUS [FOCUS ...]] [-l LABELS [LABELS ...]]
                     [-ig [{difficult,truncated,occluded,iscrowd} [{difficult,truncated,occluded,iscrowd} ...]]]
                     [-m {SSD300,SSD512}] [-n MODEL_NAME] [-bn]
                     [-w WEIGHTS_PATH] [-bs BATCH_SIZE] [-nw NUM_WORKERS]
                     [-d {cpu,cuda}] [-si START_ITERATION] [-na]
                     [-optimizer {SGD,Adam}] [-lr LEARNING_RATE]
                     [--momentum MOMENTUM] [-wd WEIGHT_DECAY]
                     [--steplr_gamma STEPLR_GAMMA]
                     [--steplr_milestones STEPLR_MILESTONES [STEPLR_MILESTONES ...]]
                     [-mi MAX_ITERATION] [-ci CHECKPOINTS_INTERVAL]
                     [--loss_alpha LOSS_ALPHA]
                     {VOC,COCO}

Easy training script for VOC or COCO style dataset

positional arguments:
  {VOC,COCO}            Dataset type

optional arguments:
  -h, --help            show this help message and exit
  -r DATASET_ROOTDIR [DATASET_ROOTDIR ...], --dataset_rootdir DATASET_ROOTDIR [DATASET_ROOTDIR ...]
                        Dataset root directory path. If dataset type is 'VOC',
                        Default is; '['/home/kado/data/voc/voc2007/trainval/VO
                        Cdevkit/VOC2007']' If dataset type is 'COCO', Default
                        is; '['/home/kado/data/coco/coco2014/trainval']'
  --focus FOCUS [FOCUS ...]
                        Image set name. If dataset type is 'VOC', Default is;
                        '['trainval']' if dataset type is 'COCO', Default is;
                        '['train2014']'
  -l LABELS [LABELS ...], --labels LABELS [LABELS ...]
                        Dataset class labels. If dataset type is 'VOC',
                        Default is; '['aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']'
                        If dataset type is 'COCO', Default is; '['person',
                        'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                        'train', 'truck', 'boat', 'traffic light', 'fire
                        hydrant', 'stop sign', 'parking meter', 'bench',
                        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball
                        bat', 'baseball glove', 'skateboard', 'surfboard',
                        'tennis racket', 'bottle', 'wine glass', 'cup',
                        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                        'pizza', 'donut', 'cake', 'chair', 'couch', 'potted
                        plant', 'bed', 'dining table', 'toilet', 'tv',
                        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink',
                        'refrigerator', 'book', 'clock', 'vase', 'scissors',
                        'teddy bear', 'hair drier', 'toothbrush']'
  -ig [{difficult,truncated,occluded,iscrowd} [{difficult,truncated,occluded,iscrowd} ...]], --ignore [{difficult,truncated,occluded,iscrowd} [{difficult,truncated,occluded,iscrowd} ...]]
                        Whether to ignore object
  -m {SSD300,SSD512}, --model {SSD300,SSD512}
                        Trained model
  -n MODEL_NAME, --model_name MODEL_NAME
                        Model name, which will be used as save name
  -bn, --batch_norm     Whether to construct model with batch normalization
  -w WEIGHTS_PATH, --weights_path WEIGHTS_PATH
                        Pre-trained weights path. Default is pytorch's pre-
                        trained one for vgg
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -nw NUM_WORKERS, --num_workers NUM_WORKERS
                        Number of workers used in DataLoader
  -d {cpu,cuda}, --device {cpu,cuda}
                        Device for Tensor
  -si START_ITERATION, --start_iteration START_ITERATION
                        Resume training at this iteration
  -na, --no_augmentation
                        Whether to do augmentation to your dataset
  -optimizer {SGD,Adam}
                        Optimizer for training
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Initial learning rate
  --momentum MOMENTUM   Momentum value for Optimizer
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        Weight decay for SGD
  --steplr_gamma STEPLR_GAMMA
                        Gamma for stepLR
  --steplr_milestones STEPLR_MILESTONES [STEPLR_MILESTONES ...]
                        Milestones for stepLR
  -mi MAX_ITERATION, --max_iteration MAX_ITERATION
  -ci CHECKPOINTS_INTERVAL, --checkpoints_interval CHECKPOINTS_INTERVAL
                        Checkpoints interval
  --loss_alpha LOSS_ALPHA
                        Loss's alpha
```

- Caution!!

  When your terminal window is small, print training summary for each iteration

  ![smallconsole.png](assets/smallconsole.png?raw=true "smallconsole")

  To avoid this, please expand your terminal window.

  ![bigconsole.png](assets/bigconsole.png?raw=true "bigconsole")

# Script Example
## Training

See also [training-voc2007+2012.ipynb](https://github.com/jjjkkkjjj/pytorch_SSD/blob/master/demo/training-voc2007%2B2012.ipynb) or [training-voc2007.ipynb](https://github.com/jjjkkkjjj/pytorch_SSD/blob/master/demo/training-voc2007.ipynb).

- First, create `augmentation`, `transform`, `target_transform` instance using `augmentations`, `transforms` and `target_transforms` module in `data`

  Example;

  ```python
  from ssd_data import transforms, target_transforms, augmentations
  
  ignore = target_transforms.Ignore(difficult=True)
  augmentation = augmentations.AugmentationOriginal()
  
  transform = transforms.Compose(
      [transforms.Resize((300, 300)),
       transforms.ToTensor(),
       transforms.Normalize(rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))]
  )
  target_transform = target_transforms.Compose(
      [target_transforms.ToCentroids(),
       target_transforms.OneHot(class_nums=datasets.VOC_class_nums, add_background=True),
       target_transforms.ToTensor()]
  )
  ```
  

Note that `None` is available to set these instances

- Second, load dataset from `datasets` module in `data`.

  Example;

  ```python
  from ssd_data import datasets
  from ssd_data import _utils
  
  train_dataset = datasets.VOC2007Dataset(ignore=ignore, transform=transform, target_transform=target_transform, augmentation=augmentation)
  
  train_loader = DataLoader(train_dataset,
                            batch_size=32,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=_utils.batch_ind_fn)
  ```

  You can use `datasets.Compose` to combine plural datasets.

- Third, create model. You can create model with specified device by `.to(device)`

  Example;

  ```python
  from ssd.models.ssd300 import SSD300
  
  model = SSD300(class_labels=train_dataset.class_labels, batch_norm=False).cuda()
  model.load_vgg_weights()
  ```

  You can load your trained weights by using `model.load_weights(path)` too.

- Last, create `Optimizer`, `SaveManager`, `LogManager` and `TrainLogger` to train.

  Example;

  ```python
  from torch.utils.data import DataLoader
  from torch.optim.sgd import SGD
  from torch.optim.adam import Adam
  from ssd.train import *
  
  optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4) # slower
  #optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=5e-4) # faster
  iter_sheduler = SSDIterMultiStepLR(optimizer, milestones=(40000, 50000), gamma=0.1, verbose=True)
  
  save_manager = SaveManager(modelname='ssd300-voc2007', interval=5000, max_checkpoints=15, plot_yrange=(0, 8))
  log_manager = LogManager(interval=10, save_manager=save_manager, loss_interval=10, live_graph=LiveGraph((0, 8)))
  trainer = TrainLogger(model, loss_func=SSDLoss(), optimizer=optimizer, scheduler=iter_sheduler, log_manager=log_manager)
  
  trainer.train(60000, train_loader)
  ```

- Result
  Learning curve example(voc2007-trainval and voc2007-test)![learning curve07](assets/ssd300-voc2007_learning-curve_i-60000.png?raw=true "learning curve")

  
  Learning curve example(voc2007-trainval and voc2012-trainval)
  ![learning curve07+12](assets/ssd300-voc2007+2012_learning-curve_i-80000.png?raw=true "learning curve")

## Testing

- First, create model. You can create model with specified device by `.to(device)`

  Example;

  ```python
  from ssd.models.ssd300 import SSD300
  from ssd_data import datasets
  
  model = SSD300(class_labels=datasets.VOC_class_labels, batch_norm=False).cuda()
  model.load_weights('./weights/ssd300-voc2007/ssd300-voc2007_i-60000.pth')
  model.eval() ## Required!!!
  ```

- Pass image and show.

  Example;

  ```python
  # must be passed RGB order
  image = cv2.cvtColor(cv2.imread('assets/coco_testimg.jpg'), cv2.COLOR_BGR2RGB)
  # imgs is list of ndarray(img)
  infers, imgs = model.infer(cv2.resize(image, (300, 300)), visualize=True, toNorm=True)
  for img in imgs: 
      # returned img order is BGR
      cv2.imshow('result', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
      cv2.waitKey()
  ```
  
  Result;
  
  ![result img](assets/coco_testimg-result.jpg?raw=true "result img")

# Pre-trained Weights

※mAP for voc2007test

|                   | SSD300 (no batchnormed)                                      | SSD512 (no batchnormed) |
| ----------------- | ------------------------------------------------------------ | ----------------------- |
| VOC2007           | [mAP: 0.7572](https://drive.google.com/file/d/1N37Rn2pr_VPov6-Z5OXLkOWAxPgpzLSu/view?usp=sharing) | mAP:                    |
| VOC2007++         | [mAP: N/A](https://drive.google.com/file/d/17ehKZwH4C0fYM0xMYB79Pwa8UD0ZgLaD/view?usp=sharing) | mAP:                    |
| VOC2007+2012      | [mAP: 0.7636](https://drive.google.com/file/d/19qEEozVLj33OXNV5zUsEoqkBkojziODw/view?usp=sharing) | mAP:                    |
| VOC2007+2012+COCO | [mAP: 0.7682](https://drive.google.com/file/d/1Ly7LeheHDToY4s8u_Ja0ltLCptsdUJ-x/view?usp=sharing) | mAP:                    |

# About SSD

- Default Box in SSD300 has been implemented in [dbox.py](https://github.com/jjjkkkjjj/pytorch_SSD/blob/master/ssd/core/boxes/dbox.py)

  ![scale](https://user-images.githubusercontent.com/16914891/80902072-f5daa780-8d4c-11ea-86ea-0f837aed2a7a.png)
  
  where ![s_range](https://user-images.githubusercontent.com/16914891/80902408-0ee35880-8d4d-11ea-9e02-32f9b76ae4ad.png).

<!--
$$
  s_k = \begin{cases}
          0.1 &k=0 \\
          s_{min} + \frac{s_{max}-s_{min}}{m-2}(k-1) &k = 1 \ldots m-1
          \end{cases},
$$

  where $s_{min}=0.2, s_{max}=0.9$.
-->

- Encode in [codec.py](https://github.com/jjjkkkjjj/pytorch_SSD/blob/master/ssd/core/boxes/codec.py)

  ![encode](https://user-images.githubusercontent.com/16914891/80902084-f6733e00-8d4c-11ea-822c-b4fbf1f7410d.png)

  where ![norm](https://user-images.githubusercontent.com/16914891/80902089-f70bd480-8d4c-11ea-93d3-73392193a07b.png).

<!--
$$
  \begin{align*}
      (\hat{g}_{j}^{cx},\hat{g}_{j}^{cy})&=\left( \frac{\frac{g_{j}^{cx}-d_{i}^{cx}}{d_{i}^{w}}-\mu^{cx}}{\sigma^{cx}}, \frac{\frac{g_{j}^{cy}-d_{i}^{cy}}{d_{i}^{h}}-\mu^{cy}}{\sigma^{cy}} \right) \\
      (\hat{g}_{j}^{w}, \hat{g}_{j}^{h})&=\left( \frac{\log{\frac{g_j^{w}}{d_{i}^{w}}}-\mu^{w}}{\sigma^{w}}, \frac{\log{\frac{g_j^{h}}{d_{i}^{h}}}-\mu^{h}}{\sigma^{h}} \right)
  \end{align*},
$$

  where $\bf{\mu}=(\mu^{cx},\mu^{cy},\mu^{w},\mu^{h})=(0,0,0,0),\bf{\sigma}=(\sigma^{cx},\sigma^{cy},\sigma^{w},\sigma^{h})=(0.1,0.1,0.2,0.2)$.
-->

- Decode in [codec.py](https://github.com/jjjkkkjjj/pytorch_SSD/blob/master/ssd/core/boxes/codec.py)

  ![decode](https://user-images.githubusercontent.com/16914891/80902094-f70bd480-8d4c-11ea-940e-ab70248c3d5e.png)

  where ![norm](https://user-images.githubusercontent.com/16914891/80902089-f70bd480-8d4c-11ea-93d3-73392193a07b.png).

<!--
$$
  \begin{align*}
      (\hat{p}_{j}^{cx}, \hat{p}_{j}^{cy})&=(d_{i}^{w}(p_{j}^{cx}\sigma^{cx}+\mu^{cx})+d_{i}^{cx}, d_{i}^{h}(p_{j}^{cy}\sigma^{cy}+\mu^{cy})+d_{i}^{cy}) \\
      (\hat{p}_{j}^{w}, \hat{p}_{j}^{h})&=(d_{i}^{w}\exp(p_{j}^{w}\sigma^{w}+\mu^{w}), d_{i}^{h}\exp(p_{j}^{h}\sigma^{h}+\mu^{h}))
  \end{align*},
$$

  where $\bf{\mu}=(\mu^{cx},\mu^{cy},\mu^{w},\mu^{h})=(0,0,0,0),\bf{\sigma}=(\sigma^{cx},\sigma^{cy},\sigma^{w},\sigma^{h})=(0.1,0.1,0.2,0.2)$.
-->

- 
