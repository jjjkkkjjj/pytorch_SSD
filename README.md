# PyTorch SSD
The implementation of SSD (Single shot detector) in PyTorch.

# Requirements and Settings

- Anaconda

  ```bash
  conda install -c anaconda pycurl
  conda install -c pytorch pytorch
  conda install -c conda-forge numpy opencv ffmpeg scipy jupyter_contrib_nbextensions jupyter_nbextensions_configurator
  ```

- Jupyter

  ```bash
  jupyter notebook
  ```

  ![nbextensions](https://user-images.githubusercontent.com/16914891/80898698-67145d80-8d41-11ea-92c3-76c3791bdb9f.png)

  

# Get VOC and COCO Dataset

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
  - `coco2014_test`

# Training

See [training-voc2007+2012.ipynb](https://github.com/jjjkkkjjj/pytorch_SSD/blob/master/demo/training-voc2007%2B2012.ipynb) or [training-voc2007.ipynb](https://github.com/jjjkkkjjj/pytorch_SSD/blob/master/demo/training-voc2007.ipynb).

- First, create `augmentation`, `transform`, `target_transform` instance using `augmentations`, `transforms` and `target_transforms` module in `data`

  Example;

  ```python
  from data import transforms, target_transforms, augmentations
  
  augmentation = augmentations.AugmentationOriginal()
  
  transform = transforms.Compose(
      [transforms.Normalize(rgb_means=(103.939, 116.779, 123.68), rgb_stds=1),
       transforms.Resize((300, 300)),
       transforms.ToTensor()]
  )
  target_transform = target_transforms.Compose(
      [target_transforms.Ignore(difficult=True),
       target_transforms.ToCentroids(),
       target_transforms.OneHot(class_nums=datasets.VOC_class_nums),
       target_transforms.ToTensor()]
  )
  ```

  Note that `None` is available to set these instances

- Second, load dataset from `datasets` module in `data`.

  Example;

  ```python
  from data import datasets
  from data import utils
  
  train_dataset = datasets.VOC2007Dataset(transform=transform, target_transform=target_transform, augmentation = augmentation)
  
  train_loader = DataLoader(train_dataset,
                            batch_size=32,
                            shuffle=True,
                            collate_fn=utils.batch_ind_fn)
  ```

  You can use `datasets.Compose` to combine plural datasets.

- Third, create model.

  Example;

  ```python
  from ssd.models.ssd300 import SSD300
  
  model = SSD300(class_nums=train_dataset.class_nums, batch_norm=False)
  model.load_vgg_weights()
  ```

  You can load your trained weights by using `model.load_weights(path)`

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
  trainer = TrainLogger(model, loss_func=SSDLoss(), optimizer=optimizer, scheduler=iter_sheduler, log_manager=log_manager, gpu=True)
  
  trainer.train(60000, train_loader)
  ```

- Result
  Learning curve example(voc2007-trainval and voc2007-test)

  ![learning curve](assets/ssd300-voc2007_learning-curve_i-60000.png?raw=true "learning curve")

# Testing

- First, create model.

  Example;

  ```python
  from ssd.models.ssd300 import SSD300
  
  model = SSD300(class_nums=test_dataset.class_nums, batch_norm=False)
  model.load_weights('weights/ssd300-voc2007-augmentation/ssd300-voc2007_i-60000.pth')
  model.eval() ## Required!!!
  ```

- Pass image and show.

  Example;

  ```python
  image = cv2.imread('assets/coco_testimg.jpg')
  infers, imgs = model.infer(cv2.resize(image, (300, 300)), visualize=True, toNorm=True)
  for img in imgs:
      cv2.imshow('result', img)
      cv2.waitKey()
  ```

  Result;

  ![result img](assets/coco_testimg-result.jpg?raw=true "result img")

# Pre-trained Weights

Sorry! I haven't uploaded them yet!! Please train yourself... :(

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
