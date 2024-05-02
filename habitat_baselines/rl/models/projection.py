from einops import rearrange
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_grid(pose, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)
    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))

    return rot_grid, trans_grid


class ComputeSpatialLocs():
    def __init__(self, egocentric_map_size, global_map_size, 
        device, coordinate_min, coordinate_max
    ):
        self.device = device
        self.cx, self.cy = 256./2., 256./2.     # Hard coded camera parameters
        self.fx = self.fy =  (256. / 2.) / np.tan(np.deg2rad(79 / 2.))
        self.egocentric_map_size = egocentric_map_size
        self.local_scale = float(coordinate_max - coordinate_min)/float(global_map_size)
        
    def forward(self, depth):
        depth = depth.permute(0, 3, 1, 2)
        _, _, imh, imw = depth.shape   # batchsize, 1, imh, imw
        x    = rearrange(torch.arange(0, imw), 'w -> () () () w').to(self.device)
        y    = rearrange(torch.arange(imh, 0, step=-1), 'h -> () () h ()').to(self.device)
        xx   = (x - self.cx) / self.fx
        yy   = (y - self.cy) / self.fy
        
        # 3D real-world coordinates (in meters)
        Z = depth
        X = xx * Z
        Y = yy * Z

        # Valid inputs
        valid_inputs = (depth != 0)  & ((Y > -0.5) & (Y < 1))

        # X ground projection and Y ground projection
        x_gp = ( (X / self.local_scale) + (self.egocentric_map_size-1)/2).round().long() # (bs, imh, imw, 1)
        y_gp = (-(Z / self.local_scale) + (self.egocentric_map_size-1)/2).round().long() # (bs, imh, imw, 1)

        return torch.cat([x_gp, y_gp], dim=1), valid_inputs


class RotateTensor:
    def __init__(self, device):
        self.device = device

    def forward(self, x_gp, heading):
        sin_t = torch.sin(heading.squeeze(1))
        cos_t = torch.cos(heading.squeeze(1))
        A = torch.zeros(x_gp.size(0), 2, 3).to(self.device)
        A[:, 0, 0] = cos_t
        A[:, 0, 1] = sin_t
        A[:, 1, 0] = -sin_t
        A[:, 1, 1] = cos_t

        grid = F.affine_grid(A, x_gp.size())
        rotated_x_gp = F.grid_sample(x_gp, grid)
        return rotated_x_gp


class Projection:
    def __init__(self, egocentric_map_size, global_map_size, device, coordinate_min, coordinate_max):
        self.egocentric_map_size = egocentric_map_size
        self.global_map_size = global_map_size
        self.compute_spatial_locs = ComputeSpatialLocs(egocentric_map_size, global_map_size, 
            device, coordinate_min, coordinate_max
        )
        self.project_to_ground_plane = ProjectToGroundPlane(egocentric_map_size, device)
        self.rotate_tensor = RotateTensor(device)

    def forward(self, conv, depth, heading):
        spatial_locs, valid_inputs = self.compute_spatial_locs.forward(depth)
        x_gp = self.project_to_ground_plane.forward(conv, spatial_locs, valid_inputs)
        rotated_x_gp = self.rotate_tensor.forward(x_gp, heading)
        return rotated_x_gp

