# 3D-Hand-Pose

## Learning to Estimate 3D Hand Pose From Single RGB Images 
- [Paper](https://lmb.informatik.uni-freiburg.de/projects/hand3d/)
- Original code: [tensorflow](https://github.com/lmb-freiburg/hand3d)(official code), [pytorch](https://github.com/ajdillhoff/colorhandpose3d-pytorch)
- [Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)  


## Getting Started
### Requirements
 * Python3.6
 * PyTorch (1.2.0)
 * [RoIAlign](https://github.com/longcw/RoIAlign.pytorch) -> tensorflow.image.crop_and_resize
 * [Dilation2d](https://github.com/ajdillhoff/colorhandpose3d-pytorch/tree/095eb9e76bedbabc7ec23ed2e3a77b9abd6a5760)-> tensorflow.nn.dilation2d
 
 
### To Do
 * RoIAlign (crop and resize)
 * dilation morphology 
### Train
  * Download [HandSegnet, PoseNet, PosePrior pretrained model ](https://drive.google.com/drive/folders/1mw0wLaxfN-L6hd1wopPl94ubFfahPNh1) and put your ./pre_model folder
  ```
  python train.py 
  ```
### Test
```
python test.py
```


