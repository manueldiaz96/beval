"""
This file contains code from https://github.com/nv-tlabs/lift-splat-shoot
License available in https://github.com/nv-tlabs/lift-splat-shoot/blob/master/LICENSE
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import math

from tools import Up


class CamEncode(nn.Module):
    def __init__(self, C, use_fpn=False):
        super(CamEncode, self).__init__()
        self.C = C
        self.use_fpn = use_fpn

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        if self.use_fpn:
            self.up2 = Up(112+40, 512)
        self.depthnet = nn.Conv2d(512, self.C, kernel_size=1, padding=0)

    def get_depth_feat(self, x):

        if self.use_fpn:
            x1, x2 = self.get_eff_depth(x)
            # Depth
            feats = []

            for x in [x1, x2]:
                feats.append(self.depthnet(x))
        else:
            x = self.get_eff_depth(x)
            feats = self.depthnet(x)
            feats = [feats]
            
        return feats

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()
        sz = x.size()
        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                # scale drop connect_rate
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                #print(sz, x.size())
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x1 = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        if self.use_fpn:
            # print('FPN')
            x2 = self.up2(endpoints['reduction_4'], endpoints['reduction_3'])
            return x1, x2
        else:
            return x1

    def forward(self, x):

        # breakpoint()
        x = self.get_depth_feat(x)

        return x

class DepthEncode(nn.Module):
    """
    Depth prediction head based on QuadroNet:
    https://openaccess.thecvf.com/content/WACV2021/papers/Goel_QuadroNet_Multi-Task_Learning_for_Real-Time_Semantic_Depth_Aware_Instance_Segmentation_WACV_2021_paper.pdf
    """

    def __init__(self, 
                 inC, 
                 max_depth, 
                 min_depth,
                 num_bins,) -> None:
        super(DepthEncode, self).__init__()

        self.inC = inC
        self.alpha = min_depth
        self.beta = max_depth
        self.num_bins = num_bins

        bins_edges = [math.exp( math.log(self.alpha) + ( (math.log(self.beta/self.alpha)*i) / self.num_bins ) ) for i in range(self.num_bins+1)]
        midpoints_log = [ ( math.log(bins_edges[i+1]) + math.log(bins_edges[i]) ) / 2 for i in range(self.num_bins)]
        
        bins_edges = torch.tensor(bins_edges)
        midpoints_log = torch.tensor(midpoints_log)

        self.bins_edges = nn.Parameter(bins_edges, requires_grad=False)
        self.midpoints_log = nn.Parameter(midpoints_log, requires_grad=False)
        
        self.conv_bins = nn.Conv2d(self.inC, num_bins, kernel_size=1, padding=0)
        
        self.conv_residual = nn.Conv2d(self.inC, num_bins, kernel_size=1, padding=0)

        self.bin_criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.residual_criterion = nn.SmoothL1Loss()

    # def _apply(self, fn):
    #     super(DepthEncode, self)._apply(fn)
    #     self.bins_edges = fn(self.bins_edges)
    #     self.midpoints_log = fn(self.midpoints_log)
    #     return self

    def logits_to_depth(self, bin_logits_idx, bin_logits, residuals, single=True):

        bin_logits = bin_logits.sigmoid()

        if single:

            indexes = torch.clamp(bin_logits_idx, min=0, max=self.num_bins-1)

            midpoints = self.midpoints_log[indexes.flatten()].view_as(indexes)
            left_edges = self.bins_edges[indexes.flatten()].view_as(indexes)
            right_edges = self.bins_edges[(indexes+1).flatten()].view_as(indexes)

            residuals_ = torch.gather(residuals, 1, indexes.unsqueeze(1)).squeeze(1)

            depth = torch.exp(midpoints+(residuals_*(torch.log(right_edges)-torch.log(left_edges))))

        else:

            for i in [-1,0,1]:

                indexes = torch.clamp(bin_logits_idx+i, min=0, max=self.num_bins-1)

                midpoints = self.midpoints_log[indexes.flatten()].view_as(indexes)
                left_edges = self.bins_edges[indexes.flatten()].view_as(indexes)
                right_edges = self.bins_edges[(indexes+1).flatten()].view_as(indexes)

                residuals_ = torch.gather(residuals, 1, indexes.unsqueeze(1)).squeeze(1)

                if i==-1:
                    depth_map = torch.exp(midpoints+(residuals_*(torch.log(right_edges)-torch.log(left_edges)))).unsqueeze(1)
                    bin_prob = torch.gather(bin_logits, 1, indexes.unsqueeze(1))

                else:
                    depth_map_ = torch.exp(midpoints+(residuals_*(torch.log(right_edges)-torch.log(left_edges)))).unsqueeze(1)
                    depth_map = torch.cat((depth_map,depth_map_),dim=1)

                    bin_prob_ = torch.gather(bin_logits, 1, indexes.unsqueeze(1))
                    bin_prob = torch.cat((bin_prob, bin_prob_),dim=1)

            bin_prob = bin_prob / torch.exp(bin_prob).sum(1).unsqueeze(1)

            depth = (depth_map*bin_prob).sum(1)

        return depth
    
    def gt2labels(self, lidar_imgs, kernel_size):
        label = nn.functional.max_pool2d(lidar_imgs, kernel_size)

        label[label==0] = self.beta * 1.5
        
        #label[label!=0] = torch.clamp(label[label!=0], min=self.alpha, max=self.beta)
        label = torch.clamp(label, min=self.alpha, max=self.beta)
        
        label_bins = torch.bucketize(label, self.bins_edges[:-2]).long()

        midpoints = self.midpoints_log[label_bins.flatten()].view_as(label_bins)
        left_edges = self.bins_edges[label_bins.flatten()].view_as(label_bins)
        right_edges = self.bins_edges[(label_bins+1).flatten()].view_as(label_bins)

        #label_bins[label==0] = 255
        label_residuals = (torch.log(label)-midpoints)/(torch.log(right_edges)-torch.log(left_edges))
        #label_residuals[label==0] = 0

        return label_bins, label_residuals.float()


    def forward(self, x):
        bin_logits = self.conv_bins(x)
        residuals = self.conv_residual(x)
        
        #bin_logits = torch.nn.functional.interpolate(bin_logits, scale_factor=2, mode='nearest')
        #residuals = torch.nn.functional.interpolate(residuals, scale_factor=2, mode='bilinear', align_corners=True)
        
        bin_logits_idx = bin_logits.argmax(1)
        
            
        depth = self.logits_to_depth(bin_logits_idx, bin_logits, residuals)
        
        indexes = torch.clamp(bin_logits_idx, min=0, max=self.num_bins)
        residuals = torch.gather(residuals, 1, indexes.unsqueeze(1)).squeeze(1)         
            
        return depth, bin_logits, residuals

    def loss_depth(self, bin_logits, residuals, lidar_imgs):
        #breakpoint()
        kernel_size = lidar_imgs.shape[-1]//bin_logits.shape[-1]
        
        label_bins, label_residuals = self.gt2labels(lidar_imgs, kernel_size)

        bin_logits = bin_logits.permute(0, 2, 3, 1).reshape(-1, self.num_bins)
        label_bins = label_bins.reshape(bin_logits.shape[0])
        
        bin_loss = self.bin_criterion(bin_logits, label_bins) #* self.bin_loss_weight
        residual_loss = self.residual_criterion(residuals, label_residuals) #* self.res_loss_weight
        
        return bin_loss, residual_loss


