import torch


def atan2(y, x):
    """ My implementation of atan2 in tensorflow.  Returns in -pi .. pi."""

    tan = torch.atan(y / (x + 1e-8))  # this returns in -pi/2 .. pi/2

    one_map = torch.ones_like(tan)

    # correct quadrant error
    correction = torch.where((x + 1e-8 < 0.0), 3.141592653589793*one_map, 0.0*one_map)
    tan_c = tan + correction  # this returns in -pi/2 .. 3pi/2

    # bring to positive values
    correction = torch.where((tan_c < 0.0), 2*3.141592653589793*one_map, 0.0*one_map)
    tan_zero_2pi = tan_c + correction  # this returns in 0 .. 2pi

    # make symmetric
    correction = torch.where((tan_zero_2pi > 3.141592653589793), -2*3.141592653589793*one_map, 0.0*one_map)
    tan_final = tan_zero_2pi + correction  # this returns in -pi .. pi
    return tan_final

def dynamic_stitch(indices, data):
    n = sum(torch.tensor(idx).numel() for idx in indices)
    res  = [None] * n
    for i, data_ in enumerate(data):
        idx = torch.tensor(indices[i]).view(-1)
        d = data_.view(idx.numel(), -1)
        k = 0
        for idx_ in idx: res[idx_] = d[k]; k += 1
    return res

def _stitch_mat_from_vecs(vector_list):
    """ Stitches a given list of vectors into a 3x3 matrix.

        Input:
            vector_list: list of 9 tensors, which will be stitched into a matrix. list contains matrix elements
                in a row-first fashion (m11, m12, m13, m21, m22, m23, m31, m32, m33). Length of the vectors has
                to be the same, because it is interpreted as batch dimension.
    """

    assert len(vector_list) == 9, "There have to be exactly 9 tensors in vector_list."
    batch_size = vector_list[0].size()[0]
    vector_list = [torch.reshape(x, [1, batch_size]) for x in vector_list]

    trafo_matrix = dynamic_stitch([[0], [1], [2],
                                      [3], [4], [5],
                                      [6], [7], [8]], vector_list)

    trafo_matrix = torch.reshape(torch.tensor(trafo_matrix), [3, 3, batch_size])
    trafo_matrix = torch.tensor(trafo_matrix).permute(2,0,1)
    #trafo_matrix = torch.permute(torch.tensor(trafo_matrix), [2, 0, 1])

    return trafo_matrix

def _get_rot_mat_x(angle):
    """ Returns a 3D rotation matrix. """
    one_vec = torch.ones_like(angle)
    zero_vec = one_vec*0.0
    trafo_matrix = _stitch_mat_from_vecs([one_vec, zero_vec, zero_vec,
                                          zero_vec, torch.cos(angle), torch.sin(angle),
                                          zero_vec, -torch.sin(angle), torch.cos(angle)])
    return trafo_matrix


def _get_rot_mat_y(angle):
    """ Returns a 3D rotation matrix. """
    one_vec = torch.ones_like(angle)
    zero_vec = one_vec*0.0
    trafo_matrix = _stitch_mat_from_vecs([torch.cos(angle), zero_vec, -torch.sin(angle),
                                          zero_vec, one_vec, zero_vec,
                                          torch.sin(angle), zero_vec, torch.cos(angle)])
    return trafo_matrix

def _get_rot_mat_z(angle):
    """ Returns a 3D rotation matrix. """
    one_vec = torch.ones_like(angle)
    zero_vec = one_vec*0.0
    trafo_matrix = _stitch_mat_from_vecs([torch.cos(angle), torch.sin(angle), zero_vec,
                                          -torch.sin(angle), torch.cos(angle), zero_vec,
                                          zero_vec, zero_vec, one_vec])
    return trafo_matrix


def canonical_trafo(coords_xyz):
    """ Transforms the given real xyz coordinates into some canonical frame.
        Within that frame the hands of all frames are nicely aligned, which
        should help the network to learn reasonable shape priors.

        Inputs:
            coords_xyz: BxNx3 matrix, containing the coordinates for each of the N keypoints
    """

    coords_xyz = torch.reshape(coords_xyz, [-1, 21, 3])

    ROOT_NODE_ID = 0  # Node that will be at 0/0/0: 0=palm keypoint (root)
    ALIGN_NODE_ID = 12  # Node that will be at 0/-D/0: 12=beginning of middle finger
    ROT_NODE_ID = 20  # Node that will be at z=0, x>0; 20: Beginning of pinky

    # 1. Translate the whole set s.t. the root kp is located in the origin
    trans = torch.unsqueeze(coords_xyz[:, ROOT_NODE_ID, :], 1)
    coords_xyz_t = coords_xyz - trans

    # 2. Rotate and scale keypoints such that the root bone is of unit length and aligned with the y axis
    p = coords_xyz_t[:, ALIGN_NODE_ID, :]  # thats the point we want to put on (0/1/0)

    # Rotate point into the yz-plane
    alpha = atan2(p[:, 0], p[:, 1])
    rot_mat = _get_rot_mat_z(alpha)
    coords_xyz_t_r1 = torch.matmul(coords_xyz_t, rot_mat)
    total_rot_mat = rot_mat

    # Rotate point within the yz-plane onto the xy-plane
    p = coords_xyz_t_r1[:, ALIGN_NODE_ID, :]
    beta = -atan2(p[:, 2], p[:, 1])
    rot_mat = _get_rot_mat_x(beta + 3.141592653589793)
    coords_xyz_t_r2 = torch.matmul(coords_xyz_t_r1, rot_mat)
    total_rot_mat = torch.matmul(total_rot_mat, rot_mat)

    # 3. Rotate keypoints such that rotation along the y-axis is defined
    p = coords_xyz_t_r2[:, ROT_NODE_ID, :]
    gamma = atan2(p[:, 2], p[:, 0])
    rot_mat = _get_rot_mat_y(gamma)
    coords_xyz_normed = torch.matmul(coords_xyz_t_r2, rot_mat)
    total_rot_mat = torch.matmul(total_rot_mat, rot_mat)

    return coords_xyz_normed, total_rot_mat
    
def flip_right_hand(coords_xyz_canonical, cond_right):
    """ Flips the given canonical coordinates, when cond_right is true. Returns coords unchanged otherwise.
        The returned coordinates represent those of a left hand.

        Inputs:
            coords_xyz_canonical: Nx3 matrix, containing the coordinates for each of the N keypoints
    """
    expanded = False
    s = coords_xyz_canonical.size()
    if len(s) == 2:
        coords_xyz_canonical = torch.unsqueeze(coords_xyz_canonical, 0)
        cond_right = torch.unsqueeze(cond_right, 0)
        expanded = True

        # mirror along y axis
        coords_xyz_canonical_mirrored = torch.stack([coords_xyz_canonical[:, :, 0], coords_xyz_canonical[:, :, 1], -coords_xyz_canonical[:, :, 2]], -1)

        # select mirrored in case it was a right hand
        coords_xyz_canonical_left = torch.where(cond_right, coords_xyz_canonical_mirrored, coords_xyz_canonical)

        if expanded:
            coords_xyz_canonical_left = torch.squeeze(torch.tensor(coords_xyz_canonical_left), 0)

        return coords_xyz_canonical_left