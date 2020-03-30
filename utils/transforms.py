import math

import numpy as np
import torch
import torch.nn.functional as F


def get_rotation_matrix(rot_params):
    # rot_params = ux, uy, uz
    
    theta = rot_params.norm(dim=1)

    st = torch.sin(theta)
    ct = torch.cos(theta)
    one_ct = 1.0 - torch.cos(theta)

    norm_fac = 1.0 / theta
    ux = rot_params[:, 0] * norm_fac
    uy = rot_params[:, 1] * norm_fac
    uz = rot_params[:, 2] * norm_fac

    top = torch.stack((ct + ux * ux * one_ct, ux * uy * one_ct - uz * st, ux * uz * one_ct + uy * st), dim=1)
    mid = torch.stack((uy * ux * one_ct + uz * st, ct + uy * uy * one_ct, uy * uz * one_ct - ux * st), dim=1)
    bot = torch.stack((uz * ux * one_ct - uy * st, uz * uy * one_ct + ux * st, ct + uz * uz * one_ct), dim=1)

    rot_matix = torch.stack((top, mid, bot), dim=1)

    return rot_matix

def flip_right_hand(coords_xyz_canonical, cond_right):

    s = coords_xyz_canonical.shape

    #coords_xyz_canonical_mirrored = coords_xyz_canonical.clone()
    coords_xyz_canonical_mirrored = torch.stack([coords_xyz_canonical[:, :, 0], coords_xyz_canonical[:, :, 1], -coords_xyz_canonical[:,:,2]], -1)
    coords_xyz_canonical_left = torch.where(cond_right, coords_xyz_canonical_mirrored, coords_xyz_canonical)

    return coords_xyz_canonical_left
