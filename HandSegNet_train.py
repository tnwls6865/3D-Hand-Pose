
import os
import cv2
import numpy as numpy

import torch
import torch.nn as nn
import torch.optim as optim

from models.HandSegNet import HandSegNet

from Dataloader import get_loader
from torchvision import transforms 

from utils.utils import single_obj_scoremap, cal_center_bb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    # path_setting 
    image_path = '../RHD_v1-1/RHD_published_v2/training/color/'
    mask_path = '../RHD_v1-1/RHD_published_v2/training/mask/'
    anno_path = '../RHD_v1-1/RHD_published_v2/training/anno_training.pickle'
    model_path = None
    # parameter 
    print_freq = 100
    batch_size = 1
    num_workers = 0
    epoch = 100


    # data load
    transform = transforms.Compose([
        transforms.RandomResizedCrop((256,256)),
        transforms.ColorJitter(hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    mask_transform = transforms.Compose([transforms.ToTensor()])

    train_loader = get_loader(image_path,
                              mask_path,
                              anno_path,
                              transform,
                              mask_transform,
                              batch_size=batch_size,
                              num_workers=num_workers)
    # model load
    handseg = HandSegNet()

    handseg.to(device)
    
    
    optimizer = optim.Adam(handseg.parameters(), 0.00001)

    loss = nn.CrossEntropyLoss.to(device)

    for i, (image, mask) in enumerate(train_loader):

        image = image.to(device)
        mask = mask.to(device)

        # hand segment
        _, _, _, _, hand_seg = handseg(image) #hand_seg output

        total_loss = loss(hand_seg, mask) 
        optimizer.step()
        optimizer.zero_grad()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f} \t'.format(epoch, i, len(train_loader),loss=total_loss.item()))
            

if __name__ == "__main__":
    main()