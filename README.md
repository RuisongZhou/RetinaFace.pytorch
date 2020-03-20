# RetinaFace in PyTorch

A [PyTorch](https://pytorch.org/) implementation of [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641). Model size only 1.7M, when Retinaface use mobilenet0.25 as backbone net. We also provide resnet50 as backbone net to get better result. The official code in Mxnet can be found [here](https://github.com/deepinsight/insightface/tree/master/RetinaFace).

## Mobile or Edge device deploy
We also provide a set of Face Detector for edge device in [here](https://github.com/biubug6/Face-Detector-1MB-with-landmark) from python training to C++ inference.


## WiderFace Val Performance in single scale When testing scale is oringinal scale

| Style | easy | medium | hard | pretrained | batch_size| train_size |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
| ResNet50 | 95.48% | 94.04% | 84.43% |true |24| 840| 
| Mobilenet0.25 (original image scale) | 90.70% | 88.16% | 73.82% | true | 32 | 640 |
| Mobilenet0.25(替换了fpn为dw) | 90.5% | 87.5% | 72.1% |   true | 32 |640 |
| Mobilenet0.25(替换了fpn为dw,替换ssh为dw) | 89.7% | 86.7% | 69.9% | true | 32 |640 |
| Mobilenet0.25(替换了fpn为dw,替换ssh为dw, outchannel=32)| 89.6% | 85.8% | 67.8% | true | 32 |640 |
| MobileNetV3 1.0 | 90.95% | 88.11% | 69.85% | true | 48 |640 |
| MobileNetV3 1.0 | 84.24% | 81.98% | 61.44% | False | 48 |640 |
| GhostNet 1.0 | 87.79% | 84.05% | 66.77% | False | 48 |640 |
| GhostNet 0.5 | 79.88% | 75.67% | 52.90% | False | 48 |640 |
## WiderFace Val Performance in single scale When testing scale is 640*480 (长边不大于640，保持长宽比)
**替换了fpn为dw,替换ssh为dw, outchannel=32**
| Style | easy | medium | hard | pretrained | batch_size |train_size |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
| Mobilenet0.25| 86.6% | 78.9% | 48.0% | true | 32 |640|
| MobileNetv3 1.0 | 88.25% | 81.38% | 49.48% | true | 48 |640 |
| MobileNetv3 1.0 | 84.27% | 75.80% | 43.32% | False | 48 |640 |
| GhostNet 1.0 | 84.98% | 77.62% | 47.24% | False | 48 |640 |
| GhostNet 0.5 | 76.29% | 66.38% | 35.02% | False | 48 |640 |

<p align="center"><img src="curve/Widerface.jpg" width="640"\></p>

## FDDB Performance.
| FDDB(pytorch) | performance | img_size| train_size | FLOPS｜
|:-|:-:|:-:|:-:|
| Mobilenet0.25 | 98.64% | origin| 640 | - |
| Resnet50 | 99.22% | origin |640 | - |
| MobileNetV3 1.0(no pretrained) | 75.36% | 128 |640 | 14.844M|
|GhostNet 1.0| 79.7% | 128 |640 | 10.296M|
|GhostNet 0.5| 70.3% | 128 |640 | 4.139M|
|GhostNet 0.5| 76.36% | 128 |320 | [[8, 16], [32, 64], [128, 256]]| 4.139M|
|GhostNet 0.5| 79.26% | 128 |320 [[10,20], [32, 64], [128, 256]] ,SSH为RFB| 4.99M|
<p align="center"><img src="curve/FDDB.png" width="640"\></p>

### Contents
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [References](#references)

## Installation
##### Clone and install
1. git clone https://github.com/biubug6/Pytorch_Retinaface.git

2. Pytorch version 1.1.0+ and torchvision 0.3.0+ are needed.

3. Codes are based on Python 3

##### Data
1. Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset.

2. Download annotations (face bounding boxes & five facial landmarks) from [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or [dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0)

3. Organise the dataset directory as follows:

```Shell
  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```
ps: wider_val.txt only include val file names but not label information.

##### Data1
We also provide the organized dataset we used as in the above directory structure.

Link: from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [baidu cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) Password: ruck

## Training
We provide restnet50 and mobilenet0.25 as backbone network to train model.
We trained Mobilenet0.25 on imagenet dataset and get 46.58%  in top 1. If you do not wish to train the model, we also provide trained model. Pretrain model  and trained model are put in [google cloud](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) and [baidu cloud](https://pan.baidu.com/s/12h97Fy1RYuqMMIV-RpzdPg) Password: fstq . The model could be put as follows:
```Shell
  ./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      Resnet50_Final.pth
```
1. Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in ``data/config.py and train.py``.

2. Train the model using WIDER FACE:
  ```Shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --network resnet50 or
  CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25
  ```


## Evaluation
### Evaluation widerface val
1. Generate txt file
```Shell
python test_widerface.py --trained_model weight_file --network mobile0.25 or resnet50
```
2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)
```Shell
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```
3. You can also use widerface official Matlab evaluate demo in [Here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)
### Evaluation FDDB

1. Download the images [FDDB](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH) to:
```Shell
./data/FDDB/images/
```

2. Evaluate the trained model using:
```Shell
python test_fddb.py --trained_model weight_file --network mobile0.25 or resnet50
```

3. Download [eval_tool](https://bitbucket.org/marcopede/face-eval) to evaluate the performance.

<p align="center"><img src="curve/1.jpg" width="640"\></p>

## References
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
