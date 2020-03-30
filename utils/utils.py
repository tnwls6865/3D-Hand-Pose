import cv2
import math

import numpy as np
from scipy.ndimage.morphology import binary_dilation
import torch
import torch.nn as nn
import torch.nn.functional as F
from .morphology import Dilation2d 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def single_obj_scoremap(scoremap, filter_size=21):
    
    s = scoremap.size() # (b,2,256,256)
    #assert len(s) == 4, "score map must be 4D"

    scoremap_softmax = F.softmax(scoremap, dim=1)
    scoremap_softmax = scoremap_softmax[:, 1:, :, :] # hand score map
    scoremap_val, _ = scoremap_softmax.max(dim=1, keepdim=False) # argmax (b, 256, 256)
    detmap_fg = torch.round(scoremap_val) # (b,256,256)
    

    max_loc = find_max_location(scoremap_val).to(torch.float32) # (b, 256, 256)

    # max location을 사용하여 dilation
    objectmap_list = []
    kernel_dil = torch.ones(filter_size, filter_size, device=scoremap.device) / float(filter_size * filter_size)

    for i in range(s[0]): # 각 배치마다 
        objectmap = max_loc[i].clone()
        
        num_passes = max(s[2], s[3]) // (filter_size//2)
        
        for j in range(num_passes): # max값 
            objectmap = torch.reshape(objectmap, [1, 1, s[2], s[3]]) # (1, 256, 256)
            objectmap_num = objectmap.cpu().detach().numpy()

            ####### TODO: dilation morphology #######
            objectmap_dil = binary_dilation(objectmap_num).astype(np.float32)
            objectmap_dil = torch.tensor(objectmap_dil).to(device)
            objectmap_dil = torch.reshape(objectmap_dil, [s[2],s[3]])
            objectmap = torch.round(detmap_fg[i]*objectmap_dil)
        
        objectmap = torch.reshape(objectmap, [1, s[2],s[3]])
        objectmap_list.append(objectmap)
    
    objectmap_list = torch.stack(objectmap_list) # 1x1x256x256
    return objectmap_list

def find_max_location(scoremap):
    
    s = scoremap.size() # bx256x256
    
    
    output = torch.zeros_like(scoremap, dtype=torch.int32) # bx256x256
    coords = scoremap.view(s[0], -1) # coords: 1x65536

    _, max_coords = torch.max(coords, -1)  
    X = torch.remainder(max_coords[:], s[1]) # max index를 scoremap크기로 나눈 나머지
    Y = max_coords[:] / s[2] # 256x256 max index를 scoremap 크기로 나눈 몫
    for i in range(s[0]): # 각 batch score map에 대해 max 좌표에 1로 만들기 
        output[i, Y[i], X[i]] = 1
    
    return output
def boolean_mask(self, tensor, mask):
    mask = self.astensor(mask, dtype='bool')
    return torch.masked_select(tensor, mask)

def cal_center_bb(binary_class_mask):
    # centers: center of the hand (batch x 2)
    # bbs: bounding box of containing the hand (batch x 4)
    # crops: size of crop defined by the bounding box (batch x 2)

    binary_class_mask = binary_class_mask.to(torch.int32)
    binary_class_mask = torch.eq(binary_class_mask, 1) # 손 부분 
    if len(binary_class_mask.shape) == 4:
        binary_class_mask = binary_class_mask.squeeze(1)
    
    s = binary_class_mask.size()
    assert len(s) == 3, "binary_class_mask must be 3D"

    bbs = []
    centers = []
    crops = []

    for i in range(s[0]):
        if len(binary_class_mask[i].nonzero().shape) < 2:
            bb = torch.zeros(2,2,
                            dtype=torch.int32,
                            device=binary_class_mask.device)
            bbs.append(bb)
            centers.append(torch.tensor([160,160], dtype=torch.int32, device=binary_class_mask.device))
            crops.append(torch.tensor(100, dtype=torch.int32, device=binary_class_mask.device))
            continue

        else:
            x_min = binary_class_mask[i].nonzero()[:, 0].min().to(torch.int32)
            y_min = binary_class_mask[i].nonzero()[:, 1].min().to(torch.int32)
            x_max = binary_class_mask[i].nonzero()[:, 0].max().to(torch.int32)
            y_max = binary_class_mask[i].nonzero()[:, 1].max().to(torch.int32)

        start = torch.stack([y_min, x_min])
        end = torch.stack([y_max, x_max])
        bb = torch.stack([start, end], 1)
        bbs.append(bb)

        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2
        center = torch.stack([center_y, center_x])
        centers.append(center)

        crop_size_x = x_max - x_min
        crop_size_y = y_max - y_min
        crop_size = max(crop_size_y, crop_size_x)
        crops.append(crop_size)

    bbs = torch.stack(bbs)
    centers = torch.stack(centers)
    crops = torch.stack(crops)

    return centers, bbs, crops

def crop_image_from_xy(image, crop_location, crop_size, scale=1.0):
    """
    image: b x C x H x W 
    crop_location: B x 2 - Height and width locations to crop
    crop_size: int - Size of the crop
    scale: float - Scale factor

    image_crop: b x c x crop_size x crop_size - cropped images
    """
    size = image.size()
    crop_location = crop_location.to(torch.float32)
    crop_size_scaled = float(crop_size) / scale

    y1 = crop_location[:, 0] - crop_size_scaled // 2
    y2 = y1 + crop_size_scaled
    x1 = crop_location[:, 1] - crop_size_scaled // 2
    x2 = x1 + crop_size_scaled
    x1 /= image.size()[3]
    x2 /= image.size()[3]
    y1 /= image.size()[2]
    y1 /= image.size()[2]
    boxes = torch.stack([y1, x1, y2, x2], -1).to(torch.float32).cuda()
    box_ind = torch.arange(0, size[0], dtype=torch.int32).cuda()

    ####### TODO: crop and resize  #######
    #image_crops = CropAndResizeFunction(crop_size, crop_size, 0)(image, boxes, box_ind)
    
    return image

    





