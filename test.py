
import argparse
import os
import cv2
import numpy as numpy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from models.HandSegNet import HandSegNet
from models.PoseNet import PoseNet
from models.HandPose import HandPose

from Dataloader import test_get_loader
from torchvision import transforms 

from utils.utils import single_obj_scoremap, cal_center_bb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):

    # data load
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_loader = test_get_loader(args.image_path,
                              transform,
                              batch_size=args.batch_size,
                              num_workers=args.num_works,
                              shuffle=False)

    # model load
    handseg = HandSegNet()
    posenet = PoseNet()
    hand3d = HandPose()
    
    handseg.to(device)
    posenet.to(device)
    hand3d.to(device)

    
    print("==== model load====")
    handseg.load_state_dict(
        torch.load(os.path.join(args.model_path, 'HandSegnet.pth.tar')))
    posenet.load_state_dict(
        torch.load(os.path.join(args.model_path, 'PoseNet.pth.tar')))
    ####### TODO
    #hand3d.load_state_dict(
    #    torch.load(os.path.join(model_path, 'poseprior.pth.tar')))

    for i, image in enumerate(test_loader):

        image = image.to(device)
        hand_side = torch.tensor([[1.0, 0.0]]).to(device)

        # hand segment
        image_crop, scale_crop, centers, _, _ = handseg(image) 

        # detect keypoints in 2D
        keypoint_scoremap = posenet(image_crop)

        # estimate 3d pose
        keypoint_coord3d, rot_matrix, keypoint_scoremap = hand3d(keypoint_scoremap, hand_sides) 

        if device == 'cuda':
            keypoint_coord3d = keypoint_coord3d.cpu()
            keypoint_scoremap = keypoint_scoremap.cpu()
            image_crop = image_crop.cpu()
            centers = centers.cpu()
            scale_crop = scale_crop.cpu()

        keypoint_coords3d = keypoint_coord3d.detach().numpy()
        keypoint_coords3d = keypoint_coords3d.squeeze()

        keypoint_coords_crop = detect_keypoints(keypoint_scoremap[0].detach().numpy())
        keypoint_coords = transform_cropped_coords(keypoint_coords_crop, centers, scale_crop, 256)
        
        fig = plt.figure(1, figsize=(16, 16))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224, projection='3d')
        ax1.imshow(image)
        plot_hand(keypoint_coords, ax1)
        ax2.imshow(transform0(image_crop[0] + 0.5))
        plot_hand(keypoint_coords_crop, ax2)
        ax3.imshow(np.argmax(keypoint_scoremap[0].detach().numpy(), 0))
        plot_hand_3d(keypoint_coords3d, ax4)
        ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
        ax4.set_xlim([-3, 3])
        ax4.set_ylim([-3, 1])
        ax4.set_zlim([-3, 3])
        plt.show()
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Learning to estimate 3D hand pose from single rgb images')
    parser.add_argument('--image_path', type=str, help='image path')
    parser.add_argument('--model_path', default='./pre_model/', type=str, help='pretrained model path')
    parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size (default: 8)')
    parser.add_argument('--num_works', default=0, type=int, help='number of data loading workers (default: 0)')
    
    args = parser.parse_args()

    main(args)