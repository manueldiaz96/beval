"""
Modified from https://github.com/nv-tlabs/lift-splat-shoot
License available in https://github.com/nv-tlabs/lift-splat-shoot/blob/master/LICENSE
"""

from matplotlib import use
import torch
import numpy as np
from torch import nn
from models.camencode import CamEncode
from models.bevencode import BevEncode
from models.lidarmodel import lidar_projection
import time

class LAPTNet(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC, use_fpn=False):
        super(LAPTNet, self).__init__()
        
        self.camC = 64
        self.use_fpn = use_fpn

        self.camencode = CamEncode(self.camC, use_fpn=self.use_fpn)

        self.LAPT = lidar_projection(grid_conf, data_aug_conf, self.use_fpn)

        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        self.timeseries = []

    def save_lidar_maps(self, x, name=None):

        with torch.no_grad():
            proj_fts = x.clone()
            proj_fts[proj_fts != 0] = 1
            proj_fts = proj_fts.sum(1).unsqueeze(1)
            proj_fts[proj_fts != 0] = 1

        if name != None:
            torch.save(proj_fts, name)

        return proj_fts

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, aug_rot=0):
        
        # t0 = time.time()

        x, lidar_img = x

        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)

        x = self.camencode(x)
        
        x = self.LAPT(x, lidar_img, rots, trans, intrins, post_rots, post_trans, [B,N,imH,imW], rot_aug=aug_rot)

        x = self.bevencode(x)

        # tf = time.time()-t0

        # if tf < 5:
        #     if len(self.timeseries) > 100:
        #         self.timeseries.pop(0)
        #     self.timeseries.append(tf)
        #     print(1/np.mean(self.timeseries))
                
        return x


def compile_model(grid_conf, data_aug_conf, outC, use_fpn=False):
    return LAPTNet(grid_conf, data_aug_conf, outC, use_fpn=use_fpn)
