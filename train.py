
import argparse
import os
import cv2
import numpy as numpy

import torch
import torch.nn as nn
import torch.optim as optim

from models.HandSegNet import HandSegNet
from models.PoseNet import PoseNet
from models.HandPose import HandPose

from Dataloader import get_loader
from torchvision import transforms 

from utils.utils import single_obj_scoremap, cal_center_bb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # data load
    transform = transforms.Compose([
        transforms.RandomResizedCrop((args.crop_size, args.crop_size)),
        transforms.ColorJitter(hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    mask_transform = transforms.Compose([transforms.ToTensor()])

    train_loader = get_loader(args.image_path,
                              args.mask_path,
                              args.anno_path,
                              transform,
                              mask_transform,
                              batch_size=args.batch_size,
                              num_workers=args.num_works,
                              shuffle=True)
    # model load
    handseg = HandSegNet()
    posenet = PoseNet()
    hand3d = HandPose()
    
    handseg.to(device)
    posenet.to(device)
    hand3d.to(device)

    if args.pretrained :
        print("====HandsegNet, PoseNet model load====")
        handseg.load_state_dict(
            torch.load(os.path.join(args.model_path, 'HandSegnet.pth.tar')))
        posenet.load_state_dict(
            torch.load(os.path.join(args.model_path, 'PoseNet.pth.tar')))

    if args.resume :
        print("====3D Hand Pose model load====")
        hand3d.load_state_dict(
            torch.load(os.path.join(model_path, '3DhandposeNet.pth.tar')))
    
    optimizer = optim.Adam(hand3d.parameters(), 0.0001)

    loss = nn.MSELoss().to(device)
    for epoch in range(args.epochs):
        
        for i, (image, hand_sides, keypoint_gt, rot_mat_gt) in enumerate(train_loader):

            image = image.to(device)
            hand_sides = hand_sides.to(device)
            keypoint_gt = keypoint_gt.to(device)
            rot_mat_gt = rot_mat_gt.to(device)

            # hand segment
            image_crop, scale_crop, center, hand_mask, hand_seg = handseg(image) #hand_seg output

            # detect keypoints in 2D
            keypoint_scoremap = posenet(image_crop)

            # estimate 3d pose
            keypoint_coord3d, rot_matrix, _ = hand3d(keypoint_scoremap, hand_sides) # (b, 21, 3)
            
            total_loss = loss(keypoint_coord3d, keypoint_gt) + loss(rot_matrix, rot_mat_gt)
            
            optimizer.step()
            optimizer.zero_grad()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss:.4f} \t'.format(epoch, i, len(train_loader),loss=total_loss.item()))
            

if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(description='Learning to estimate 3D hand pose from single rgb images')
    parser.add_argument('--image_path', default='../RHD_v1-1/RHD_published_v2/training/color/', type=str,
                        help='image path')
    parser.add_argument('--mask_path', default='../RHD_v1-1/RHD_published_v2/training/color/', type=str,
                        help='image path')
    parser.add_argument('--anno_path', default='../RHD_v1-1/RHD_published_v2/training/anno_training.pickle', type=str,
                        help='annotation path')
    parser.add_argument('--model_path', default='./pre_model/', type=str, help='pretrained model path')
    parser.add_argument('--print_freq', default=100, type=int, help='print frequency (default: 100)')
    parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size (default: 8)')
    parser.add_argument('--num_works', default=0, type=int, help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--resume', default=False, type=bool, help='retrain')
    parser.add_argument('--pretrained', default=True, type=bool, help='HandSegNet and PoseNet pretrained')
    parser.add_argument('--crop_size', default=256, type=int, help='image crop size')'''
    

    args = parser.parse_args()

    main(args)