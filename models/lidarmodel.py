"""
This file contains code from https://github.com/nv-tlabs/lift-splat-shoot
License available in https://github.com/nv-tlabs/lift-splat-shoot/blob/master/LICENSE
"""

from matplotlib import use
import torch
import numpy as np
from torch import nn
import time
#import ipdb as pdb
from tools import cumsum_trick, QuickCumsum
from dataloaders.data_tools.tools_map import gen_dx_bx



class lidar_projection(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, use_fpn=False):
        super(lidar_projection, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.use_fpn = use_fpn

        dx, bx, nx = gen_dx_bx(**self.grid_conf)
        
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum_dict = {}

        #if use_fpn:
        self.frustum_dict[self.downsample] = self.create_frustum()
        
        self.downsample = 8
        self.frustum_dict[self.downsample] = self.create_frustum()

        self.downsample = 16

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

        self.lidar_resizer = {16: torch.nn.MaxPool2d(
                                    self.downsample), 
                              8: torch.nn.MaxPool2d(self.downsample//2)}
        self.time_series = []

    def _apply(self, fn):
        super(lidar_projection, self)._apply(fn)
        for k in self.frustum_dict.keys():
            self.frustum_dict[k] = fn(self.frustum_dict[k])
        return self

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        xs = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).view(1, fW).expand(fH, fW)
        ys = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).view(fH, 1).expand(fH, fW)
        ds = torch.ones_like(xs)
        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, depth_img, rot_aug=0):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape
        # Intrinsics
        # B x N x D x H x W x 3
        B, N, _ = trans.shape
        points = self.frustum - post_trans.view(B, N, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # Extrinsics
        # cam_to_ego
        # Multiplying by z
        B, N, H, W, _, _ = points.shape

        if depth_img.shape[-1] > W:
            depth_img = self.lidar_resizer[self.downsample](depth_img.float()).view(B,N,H,W,1,1).expand(B,N,H,W,3,1).clone()
            depth_img[torch.where(depth_img==0)] = 200.
        else:
            depth_img = depth_img.view(B,N,H,W,1,1).expand(B,N,H,W,3,1).clone()

        #h = time.time()
        points = points * depth_img

        #h = time.time()
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 3)

        if type(rot_aug) != int:
            #breakpoint()
            rot_aug = rot_aug.view(B,1,3,3).expand(B,N,3,3)
            points = rot_aug.view(B, N, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
            #breakpoint()

        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, H, W, C = x.shape
        Nprime = B*N*H*W
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        # breakpoint()

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros(
            (B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2],
              geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final


    def forward(self, x, lidar_img, rots, trans, intrins, post_rots, post_trans, shape_info, rot_aug=0):

        B, N, imH, imW = shape_info

        #breakpoint()
        #if not self.use_fpn:
        #    x = [x]

        if type(lidar_img) == list:
            depth_list = lidar_img.copy()

        geom = [] # x1->1/16 | x2->1/8  Downsampling rates
        for i in range(len(x)):

            self.downsample = 2**(4-i)
            self.frustum = self.frustum_dict[self.downsample]
            
            #breakpoint()
            
            if 'depth_list' in locals():
                lidar_img = depth_list[i]

            geom.append(self.get_geometry(rots, trans, intrins, post_rots, post_trans, lidar_img, rot_aug))

            x[i] = x[i].view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
            x[i] = x[i].permute(0, 1, 3, 4, 2)

        self.downsample = 16

        for i in range(len(x)):
            x[i] = self.voxel_pooling(geom[i], x[i])

        if self.use_fpn:
            x = x[0]+x[1]
        else:
            x = x[0]

        return x
 
