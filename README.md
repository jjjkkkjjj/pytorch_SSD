# PyTorch SSD
The implementation of SSD (Single shot detector) in PyTorch.

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

- PyCharm

  Preference > Build, Execution, Deployment > Python Debugger
  check "Collect run-time types information for code insight"

  ![pycharm](https://user-images.githubusercontent.com/16914891/73588552-f7a68c00-450d-11ea-95f9-1f7f9c5e0128.png)