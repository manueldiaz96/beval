"""
This file contains code from https://github.com/nv-tlabs/lift-splat-shoot
License available in https://github.com/nv-tlabs/lift-splat-shoot/blob/master/LICENSE
"""

from matplotlib import use
import torch
from torch import nn
from models.camencode import CamEncode
from models.bevencode import BevEncode
from models.lidarmodel import lidar_projection
from models.pointpillars import PillarFeatures

import time
import numpy as np

from tools import fuse_features

class LAPTNet_PP(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC, use_fpn=False, cfg_pp=None, fusion_method='add'):
        super(LAPTNet_PP, self).__init__()


        assert cfg_pp != None, 'PointPillars configuration is required'

        self.camC = 64
        self.use_fpn = use_fpn

        self.camencode = CamEncode(self.camC, use_fpn=self.use_fpn)

        self.LAPT = lidar_projection(grid_conf, data_aug_conf, self.use_fpn)

        assert fusion_method in ['add', 'avg', 'concat', 'avg_pool', 'max_pool'], 'Fusion method not supported'
        self.fusion_method = fusion_method

        self.cfg_pp = cfg_pp
        self.pointpillars = PillarFeatures(cfg_pp)

        self.bevencode = BevEncode(inC=self.camC*(1+int(self.fusion_method=='concat')), outC=outC)

        self.timeseries = []


    def save_lidar_maps(self, x, name=None):

        with torch.no_grad():
            proj_fts = x.clone()
            proj_fts[proj_fts != 0] = 1
            proj_fts = proj_fts.sum(1).unsqueeze(1)
            proj_fts[proj_fts != 0] = 1

        if name != None:
            torch.save(proj_fts.cpu(), name)

        return proj_fts

    def forward(self, x, pp_data, rots, trans, intrins, post_rots, post_trans, aug_rot=0):
        
        # t0 = time.time()
        
        x, lidar_img = x
        voxels, coors, num_points = pp_data

        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)

        x = self.camencode(x)
        
        x = self.LAPT(x, lidar_img, rots, trans, intrins, post_rots, post_trans, [B,N,imH,imW], aug_rot)

        pp_feats = self.pointpillars(voxels, coors, num_points, B).flip(-1)
        #self.save_lidar_maps(pp_feats, 'dev_notebooks/debugging_lapt_pp/pp_feats.pt')

        x = fuse_features((x,pp_feats), self.fusion_method)

        x = self.bevencode(x)

        # tf = time.time()-t0

        # if tf < 5:
        #     if len(self.timeseries) > 100:
        #         self.timeseries.pop(0)
        #     self.timeseries.append(tf)
        #     print(1/np.mean(self.timeseries))

        return x

def compile_model(grid_conf, data_aug_conf, outC, use_fpn=False, cfg_pp=None, fusion_method='add'):
    return LAPTNet_PP(grid_conf, data_aug_conf, outC, use_fpn=use_fpn, cfg_pp=cfg_pp, fusion_method=fusion_method)
