# 3D-Hand-Pose

## Learning to Estimate 3D Hand Pose From Single RGB Images 
- Paper : (https://lmb.informatik.uni-freiburg.de/projects/hand3d/)
- Original code : (https://github.com/lmb-freiburg/hand3d), https://github.com/ajdillhoff/colorhandpose3d-pytorch
- Dataset : (https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)  


## Getting Started
### Requirements
 * Python3.6
 * PyTorch (1.2.0)
 * [RoIAlign](https://github.com/longcw/RoIAlign.pytorch)
 * [Dilation2d](https://github.com/ajdillhoff/colorhandpose3d-pytorch/tree/095eb9e76bedbabc7ec23ed2e3a77b9abd6a5760)
 
### Train
  * Download[HandSegnet and PoseNet pretrained model](https://drive.google.com/drive/folders/1mw0wLaxfN-L6hd1wopPl94ubFfahPNh1)   
  >python train.py 
