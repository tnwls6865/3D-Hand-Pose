
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import single_obj_scoremap, cal_center_bb, crop_image_from_xy

class HandSegNet(nn.Module):

    def __init__(self, crop_size=256):
        super(HandSegNet, self).__init__()
        
        self.crop_size = crop_size

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 128, 3, padding=1)

        self.conv6_1 = nn.Conv2d(128, 512, 1)
        self.conv6_2 = nn.Conv2d(512, 2, 1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

    def model_conv(self, x):
        
        s = x.size()                       # size of the s: bx3x256x256
        out = self.relu(self.conv1_1(x))   # 1 size of the out: (b,64,256,256)
        out = self.relu(self.conv1_2(out)) # 2 size of the out: (b,64,256,256)
        out = self.pool(out)            # 3 size of the out:(b,64,128,128)
        out = self.relu(self.conv2_1(out)) # 4 size of the out: (b,128,128,128)
        out = self.relu(self.conv2_2(out)) # 5 size of the out: (b,128,128,128)
        out = self.pool(out)            # 6 size of the out: (b,128,64,64)
        out = self.relu(self.conv3_1(out)) # 7 size of the out: (b,256,64,64)
        out = self.relu(self.conv3_2(out)) # 8 size of the out: (b,256,64,64)
        out = self.relu(self.conv3_3(out)) # 9 size of the out: (b,256,64,64)
        out = self.relu(self.conv3_4(out)) # 10 size of the out:(b,256,64,64)
        out = self.pool(out)            # 11 size of the out: (b,256,32,32)
        out = self.relu(self.conv4_1(out)) # 12 size of the out: (b,512,32,32)
        out = self.relu(self.conv4_2(out)) # 13 size of the out: (b,512,32,32)
        out = self.relu(self.conv4_3(out)) # 14 size of the out: (b,512,32,32)
        out = self.relu(self.conv4_4(out)) # 15 size of the out: (b,512,32,32)
        out = self.relu(self.conv5_1(out)) 
        out = self.relu(self.conv5_2(out)) 
        out = self.relu(self.conv6_1(out))
        out = self.conv6_2(out) 
       
        out = F.interpolate(out, s[2], mode='bilinear', align_corners=False) 

        return out


    def forward(self, x):
        
        # score for background and hand class 
        hand_seg = self.model_conv(x) # (b, 2, 256, 266)
        
        # calculate single highest scoring object
        hand_mask = single_obj_scoremap(hand_seg) # (b, 1, 256, 256)

        # crop and resize
        center, _, crop = cal_center_bb(hand_mask)
        crop = crop.to(torch.float32)

        crop *= 1.25

        scale_crop = torch.min(
                torch.max(self.crop_size / crop, torch.tensor(0.25, device=x.device)),torch.tensor(5.0, device=x.device))
        image_crop = crop_image_from_xy(x, center, self.crop_size, scale_crop)


        return image_crop, scale_crop, center, hand_mask, hand_seg
