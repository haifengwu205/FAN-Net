# FAN-Net

This is the reference PyTorch implementation for training and testing MFIF models using the method described in
> **FAN-Net: Frequency Adaptive Deep Neural Network for Multi-focus Image Fusion**


## ⚙️ Setup
Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
conda install pytorch=1.12.1 torchvision=0.13.1 -c pytorch
pip install tensorboardX==2.6.2
conda install opencv-python=4.7.0.68
conda install pandas=1.1.5
pip install Pillow=9.4.0
pip install pydensecrf
pip install scikit-image=0.19.3
pip install tqdm=4.64.1
pip install wandb=0.13.10
pip install PyWavelets=1.3.0
```

## ⏳ Training
```shell
sh train.sh
```

## 📊 Evaluation
```shell
sh test.sh
```

## 👩‍⚖️ License
Paper submission in progress.
All rights reserved.
