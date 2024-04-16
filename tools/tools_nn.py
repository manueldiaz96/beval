
"""
conv_block taken from SOLOv2 repository: https://github.com/WXinlong/SOLO/
License available at: https://github.com/WXinlong/SOLO/blob/master/LICENSE

"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class MaxChannelPool_NN(torch.nn.MaxPool1d):
    # Taken from https://stackoverflow.com/questions/46562612/pytorch-maxpooling-over-channels-dimension

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = F.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)

class AvgChannelPool_NN(torch.nn.AvgPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = F.avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode
        )
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)

def MaxChannelPool(input, kernel_size, stride, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    # Taken from https://stackoverflow.com/questions/46562612/pytorch-maxpooling-over-channels-dimension

    n, c, w, h = input.size()
    input = input.view(n, c, w * h).permute(0, 2, 1)
    pooled = F.max_pool1d(
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        return_indices,
    )
    _, _, c = pooled.size()
    pooled = pooled.permute(0, 2, 1)
    return pooled.view(n, c, w, h)

def AvgChannelPool(input, kernel_size, stride, padding=0, ceil_mode=False):

    n, c, w, h = input.size()
    input = input.view(n, c, w * h).permute(0, 2, 1)
    pooled = F.avg_pool1d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode
    )
    _, _, c = pooled.size()
    pooled = pooled.permute(0, 2, 1)
    return pooled.view(n, c, w, h)

def fuse_features(features, fuse_option='add', mixing='top_bottom'):

    """
    Method to fuse features using Pytorch functions.

    Args:
        features (Tuple): Tuple containing the feature maps of the same shape to fuse, e.g. (cam_fts, lidar_fts, radar_fts)
        
        fuse_option (str): Fusion method to apply, can be: 
            'add' - addition

            'avg' - average

            'concat' - concatenation

            'avg_pool' - average pooling per channel of each feature, i.e. avg(cam_fts[n], lidar_fts[n], radar_fts[n])

            'max_pool' - max pooling per channel of each feature, i.e. max(cam_fts[n], lidar_fts[n], radar_fts[n])


        mixing (str): Mixing method for concatenation or pooling fusion, can be:

            'top_bottom' - features are stacked on top of each other, e.g. (a,a,a,b,b,b,c,c,c)

            'alternate' - features are alternated channel-wise, e.g. (a,b,c,a,b,c,a,b,c)

    Returns:
        Tensor: fused_feats, tensor of shape features[0].shape 
    """
    assert fuse_option in ['add', 'avg', 'concat', 'avg_pool', 'max_pool'], 'Fusion method not supported'
    assert type(features) is tuple, 'Features must be given as a tuple for fusion'

    num_feats = len(features)
    assert num_feats > 1, 'Need features from at least two sources to fuse'

    if fuse_option in ['concat', 'avg_pool', 'max_pool']:
        assert mixing in ['top_bottom', 'alternate'], 'Mixing method not supported for {}'.format(fuse_option)

    fused_feats = features[0]

    for i in range(1, num_feats):
        #breakpoint()
        assert (fused_feats.shape[0] == features[i].shape[0]), 'Features must have same batch size'
        assert (fused_feats.shape[-1] == features[i].shape[-1]) and (fused_feats.shape[-2] == features[i].shape[-2]), 'Features must have same spatial size'

        if fuse_option == 'add' or fuse_option == 'avg':
            fused_feats += features[i]

    if fuse_option == 'avg':
        fused_feats = fused_feats/num_feats
            
    if fuse_option in ['concat', 'avg_pool', 'max_pool']:
        if mixing == 'top_bottom':
            fused_feats = torch.cat(features, dim=1)
        if mixing == 'alternate':
            B, C, H, W = features[0].shape
            fused_feats = torch.stack(features, dim=2).view(B, C*num_feats, H, W)

        if fuse_option == 'avg_pool':
            fused_feats = AvgChannelPool(fused_feats, num_feats, num_feats)
        
        elif fuse_option == 'max_pool':
            fused_feats = MaxChannelPool(fused_feats, num_feats, num_feats)


    return fused_feats