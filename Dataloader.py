import cv2
import glob
import numpy as np
import os
from scipy.stats import truncnorm
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.canonical_trafo import canonical_trafo, flip_right_hand

class Train_Dataset(Dataset):
    def __init__(self, image_path, mask_path, anno_path, transform=None, mask_transform=None, sigma=25.0,
                 use_wrist_coord=True, hand_crop=False, coord_uv_noise=False, crop_center_noise=False,
                 crop_scale_noise=False, crop_offset_noise=False, scoremap_dropout=False, scale_to_size=False):
        self.image_path = image_path
        self.mask_path = mask_path
        self.transform = transform
        self.mask_transform = mask_transform

        with open(anno_path, 'rb') as f:
            self.data = pickle.load(f)
            
            self.num_samples = len(self.data)

        self.hand_crop = hand_crop
        self.coord_uv_noise = coord_uv_noise
        if crop_scale_noise:
            # std dev in px of noise on the uv coordinates
            self.coord_uv_noise_sigma = 2.5
        
        if crop_center_noise:
            # std dev in px: this moves what is in the "center", but the crop always contains all keypoints
            self.crop_center_noise_sigma = 20.0
        
        self.crop_scale_noise = crop_scale_noise
        if crop_offset_noise:
            # translates the crop after size calculation (this can move keypoints outside)
            self.crop_offset_noise_sigma = 10.0
        
        self.scoremap_dropout = scoremap_dropout
        if scoremap_dropout:
            self.scoremap_dropout_prob = 0.8
        
        self.scale_to_size= scale_to_size
        if scale_to_size:
            self.scale_target_size = (240, 320)

        self.image_size = (320, 320)
        self.crop_size = 256
        self.num_kp = 42
        self.sigma = sigma # float, size of the ground truth scoremaps

        self.use_wrist_coord = use_wrist_coord
        
        

    def __getitem__(self, index):
        data = self.data
        image = Image.open(os.path.join(self.image_path, '%.5d.png' % index)).convert('RGB')
        #image = cv2.imread(os.path.join(self.image_path, '%.5d.png' % index), 0)
        if self.transform is not None:
            image = self.transform(image)

        mask = Image.open(os.path.join(self.mask_path, '%.5d.png' % index))
        #mask = Image.open(os.path.join(self.mask_path, '%.5d.png' % index), 0)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        
        keypoint_xyz = data[index]['xyz']
        if not self.use_wrist_coord:
            palm_coord_l = torch.squeeze(0.5*(keypoint_xyz[0, :] + keypoint_xyz[12, :]), 0)
            palm_coord_r = torch.squeeze(0.5*(keypoint_xyz[21, :] + keypoint_xyz[33, :]), 0)
            keypoint_xyz = torch.cat([palm_coord_l, keypoint_xyz[1:21, :], palm_coord_r, keypoint_xyz[-20:, :]], 0)

        keypoint_uv = data[index]['uv_vis']
        u = torch.tensor(data[index]['uv_vis'][:,0])
        v = torch.tensor(data[index]['uv_vis'][:,1])
        keypoint_uv = torch.stack((u,v), dim=1)
        if not self.use_wrist_coord:
            palm_coord_uv_l = torch.squeeze(0.5*(keypoint_uv[0, :] + keypoint_uv[12, :]), 0)
            palm_coord_uv_r = torch.squeeze(0.5*(keypoint_uv[21, :] + keypoint_uv[33, :]), 0)
            keypoint_uv = tf.concat([palm_coord_uv_l, keypoint_uv[1:21, :], palm_coord_uv_r, keypoint_uv[-20:, :]], 0)
        if self.coord_uv_noise :
            #shape, mean, stddev
            ##################### truncated normal distribution 추가 #####################
            noise = tf.truncated_normal([42, 2], mean=0.0, stddev=self.coord_uv_noise_sigma)
            keypoint_uv += noise
        
        hand_parts = mask.to(torch.int32)
        hand_mask = (mask > 1)
        bg_mask = ~hand_mask
        hand_mask = torch.stack([bg_mask, hand_mask], 1)

        keypoint_vis = data[index]['uv_vis'][:,2].astype(bool)
        if not self.use_wrist_coord:
            palm_vis_l = torch.squeeze(np.logical_or(keypoint_vis[0], keypoint_vis[12]), 0)
            palm_vis_r = torch.squeeze(np.logical_or(keypoint_vis[21], keypoint_vis[33]), 0)
            keypoint_vis = torch.concat([palm_vis_l, keypoint_vis[1:21], palm_vis_r, keypoint_vis[-20:]], 0)

        one_map, zero_map = torch.ones_like(mask), torch.zeros_like(mask)
        one_map =  one_map.to(torch.int32)
        zero_map =  zero_map.to(torch.int32)
        cond_l = np.logical_and((hand_parts > one_map), (hand_parts < one_map*18))
        cond_r = (hand_parts > one_map*17)
        hand_map_l = torch.where(cond_l, one_map, zero_map)
        hand_map_r = torch.where(cond_r, one_map, zero_map)
        num_px_left_hand = hand_map_l.sum()
        num_px_right_hand = hand_map_r.sum()

        # produce the 21 subset using the segmentation masks
        kp_coord_xyz_left = keypoint_xyz[:21, :]
        kp_coord_xyz_right = keypoint_xyz[-21:, :]

        cond_left = np.logical_and(torch.ones_like(torch.tensor(kp_coord_xyz_left)).to(torch.bool), (num_px_left_hand > num_px_right_hand))
        kp_coord_xyz21 = torch.where(cond_left, torch.tensor(kp_coord_xyz_left), torch.tensor(kp_coord_xyz_right))

        hand_side = torch.where((num_px_left_hand > num_px_right_hand),torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32))
        temp = torch.zeros(1, 2)
        temp[range(temp.shape[0]), hand_side]=1
        hand_side = temp.reshape(temp.size()[-1])

        keypoint_xyz21 = kp_coord_xyz21

        # make coords relative to root joint
        kp_coord_xyz_root = kp_coord_xyz21[0, :] # palm coord
        kp_coord_xyz21_rel = kp_coord_xyz21 - kp_coord_xyz_root  # relative coords in metric coords
        index_root_bone_length = torch.sqrt(((kp_coord_xyz21_rel[12, :] - kp_coord_xyz21_rel[11, :])**2).sum())
        keypoint_scale = index_root_bone_length
        keypoint_xyz21_normed = kp_coord_xyz21_rel / index_root_bone_length # normalized

        # calculate viewpoint and coords in canonical coordinates
        kp_coord_xyz21_rel_can, rot_mat = canonical_trafo(keypoint_xyz21_normed)
        kp_coord_xyz21_rel_can, rot_mat = torch.squeeze(kp_coord_xyz21_rel_can), torch.squeeze(rot_mat)
        kp_coord_xyz21_rel_can = flip_right_hand(kp_coord_xyz21_rel_can, np.logical_not(cond_left))
        keypoint_xyz21_can = kp_coord_xyz21_rel_can
        rot_mat = torch.inverse(rot_mat)

        # set 21 of for visibility
        keypoint_vis_left = keypoint_vis[:21]
        keypoint_vis_right = keypoint_vis[-21:]
        keypoint_vis_left = keypoint_vis_left.astype(np.int32)
        keypoint_vis21 = torch.where(cond_left[:, 0], torch.tensor(keypoint_vis_left.astype(np.int32)), torch.tensor(keypoint_vis_right.astype(np.int32)))
        
        # set of 21 for UV coordinates
        keypoint_uv_left = keypoint_uv[:21, :]
        keypoint_uv_right = keypoint_uv[-21:, :]
        keypoint_uv21 = torch.where(cond_left[:, :2], keypoint_uv_left, keypoint_uv_right)

        # create scoremaps from the subset of 2D annotation
        keypoint_hw21 = torch.stack([keypoint_uv21[:, 1], keypoint_uv21[:, 0]], -1)
        
        scoremap_size = self.image_size
        if self.hand_crop:
            scoremap_size = (self.crop_size, self.crop_size)
        
        score_map = self.create_multiple_gaussian_map(keypoint_hw21,
                                                      scoremap_size,
                                                      self.sigma,
                                                      valid_vec=keypoint_vis21)
        if self.scoremap_dropout:
            drop = torch.nn.Dropout(self.scoremap_dropout_prob)
            score_map = drop(score_map)

        if self.scale_to_size:
            s = image.size()
            image = image.resize(self.scale_target_size)
            scale = (self.scale_target_size[0]/float(s[0]), self.scale_target_size[1]/float(s[1]))
            keypoint_uv21 = torch.stack([keypoint_uv21[:, 0] * scale[1],
                                         keypoint_uv21[:, 1] * scale[0]], 1)


        return image, hand_side, keypoint_xyz21, rot_mat
        #return image, hand_side, hand_parts, keypoint_xyz21, keypoint_vis21, keypoint_scale


        
    def __len__(self):
        return self.num_samples

    
    @staticmethod
    def create_multiple_gaussian_map(coords_uv, output_size, sigma, valid_vec=None):
        """ Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
            with variance sigma for multiple coordinates."""
        
        assert len(output_size) == 2
        s = coords_uv.size()
        coords_uv = coords_uv.to(torch.int32)
        if valid_vec is not None:
            valid_vec = valid_vec.to(torch.float32)
            valid_vec = torch.squeeze(valid_vec)
            cond_val = (valid_vec > 0.5)
        else:
            cond_val = torch.ones_like(coords_uv[:, 0], dtype=torch.float32)
            cond_val = (cond_val > 0.5)

        cond_1_in = np.logical_and((coords_uv[:, 0] < output_size[0]-1), (coords_uv[:, 0] > 0))
        cond_2_in = np.logical_and((coords_uv[:, 1] < output_size[1]-1), (coords_uv[:, 1] > 0))
        cond_in = np.logical_and(cond_1_in, cond_2_in)
        cond = np.logical_and(cond_val, cond_in)

        coords_uv = coords_uv.to(torch.float32)

        # create meshgrid
        x_range = torch.unsqueeze(torch.arange(output_size[0]), 1)
        y_range = torch.unsqueeze(torch.arange(output_size[1]), 0)

        X = (x_range).repeat([1, output_size[1]]).to(torch.float32)
        Y = (y_range).repeat([output_size[0], 1]).to(torch.float32)

        X = X.view((output_size[0], output_size[1]))
        Y = Y.view((output_size[0], output_size[1]))

        X = torch.unsqueeze(X, -1)
        Y = torch.unsqueeze(Y, -1)

        X_b = X.repeat([1, 1, s[0]])
        Y_b = Y.repeat([1, 1, s[0]])

        X_b -= coords_uv[:, 0]
        Y_b -= coords_uv[:, 1]

        dist = (X_b)**2 + (Y_b)**2

        scoremap = torch.exp(-dist / (sigma**2)) * cond.to(torch.float32)

        return scoremap


class Test_Dataset(Dataset):
    def __init__(self, image_path, transform=None):
        
        self.image_path = image_path
        self.transform = transform
        self.image_list = glob.glob(image_path+'*.*')

        self.num_samples = len(self.image_list)

    def __getitem__(self, index):

        image = Image.open(os.path.join(self.image_path, '%.5d.png' % index)).convert('RGB')
        #image = cv2.imread(os.path.join(self.image_path, '%.5d.png' % index), 0)
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return self.num_samples


def collate_fn(data):
    
    data = [b for b in data if b is not None]
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, hand_sides, keyponit_xyz21, rot_mat = zip(*data)

    images = torch.stack(images, 0)
    hand_sides = torch.stack(hand_sides, 0)
    keyponit_xyz21 = torch.stack(keyponit_xyz21, 0)
    rot_mat = torch.stack(rot_mat, 0)


    return images, hand_sides, keyponit_xyz21, rot_mat
    
def get_loader(image_path, mask_path, anno_path, transform, mask_transform, batch_size, num_workers, shuffle=True):
    dataset = Train_Dataset(image_path, mask_path, anno_path, transform, mask_transform)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn = collate_fn)
    return data_loader

def test_get_loader(image_path, transform, batch_size, num_workers, shuffle=False):
    dataset = Test_Dataset(image_path, transform)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn = collate_fn)
    return data_loader
    