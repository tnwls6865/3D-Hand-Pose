import os

import torch
import torch.nn as nn
import torch.optim as optim

from models.PosePrior import PosePrior
from models.Viewpoint import Viewpoint

from utils.transforms import get_rotation_matrix, flip_right_hand

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HandPose(nn.Module):
    def __init__(self, crop_size=256, num_keypoints=21, model_path=None):
        super(HandPose, self).__init__()

        self.crop_size = crop_size
        self.num_keypoints = num_keypoints

        self.poseprior = PosePrior()
        self.viewpoint = Viewpoint()


    def forward(self, keypoint_scoremap, hand_sides):
        
        self.poseprior.to(device)
        self.viewpoint.to(device)

        # estimate 3d pose
        coord_can = self.poseprior(keypoint_scoremap, hand_sides) # (b, 21, 3)
        rot_params = self.viewpoint(keypoint_scoremap, hand_sides)
        rot_matrix = get_rotation_matrix(rot_params) # (b, 3, 3)

        # flip hand according to hand side
        cond_right = torch.eq(torch.argmax(hand_sides, 1), 1) # (b)
        cond_rihgt_all = torch.reshape(cond_right, [-1, 1, 1]).repeat(1, self.num_keypoints, 3) # (b, 21, 3)
        coords_xyz_can_flip = flip_right_hand(coord_can, cond_rihgt_all) # (b, 21, 3)

        keypoint_coord3d = torch.matmul(coords_xyz_can_flip, rot_matrix)


        return keypoint_coord3d, rot_matrix

    
