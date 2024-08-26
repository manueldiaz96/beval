"""
DIfferent validation functions used for the tested tasks 

This file contains code from https://github.com/nv-tlabs/lift-splat-shoot
License available in https://github.com/nv-tlabs/lift-splat-shoot/blob/master/LICENSE
"""

import torch
import numpy as np
import cv2
import os
import copy

from tqdm import tqdm
from dataloaders.data_tools.tools_pc import points_to_voxel_loop

import warnings
warnings.filterwarnings("ignore")


def get_batch_iou(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        pred = (preds > 0)
        tgt = binimgs.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if (union > 0) else 1.0

def get_batch_dice(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        pred = (preds > 0)
        tgt = binimgs.bool()
        num = 2 * (pred & tgt).sum().float().item()
        den = pred.sum().float().item() + tgt.sum().float().item()
    return num, den, num / den if (den > 0) else 1.0


def get_batch_iou_multi_class(preds, binimgs, num_classes=1):
    """Assumes preds has NOT been sigmoided yet
    """
    iou = torch.zeros((num_classes), dtype=float)
    intersect = torch.zeros((num_classes), dtype=float)
    union = torch.zeros((num_classes), dtype=float)

    for c in range(num_classes):
        with torch.no_grad():
            pred = (preds[:, c] > 0)
            tgt = binimgs == c
            intersect[c] = (pred & tgt).sum().float().item()
            union[c] = (pred | tgt).sum().float().item()
            iou[c] = intersect[c] / union[c] if (union[c] > 0) else 1.0

    return intersect, union, iou

def get_batch_dice_multi_class(preds, binimgs, num_classes=1):
    """Assumes preds has NOT been sigmoided yet
    """
    dice = torch.zeros((num_classes), dtype=float, device=preds.device)
    num = torch.zeros((num_classes), dtype=float, device=preds.device)
    den = torch.zeros((num_classes), dtype=float, device=preds.device)

    for c in range(num_classes):
        with torch.no_grad():
            pred = (preds[:, c] > 0)
            tgt = binimgs == c

            num[c] = 2 * (pred & tgt).sum().float().item()
            den[c] = pred.sum().float().item() + tgt.sum().float().item()
            
            dice[c] = num[c] / den[c] if (den[c] > 0) else 1.0

    return num, den, dice


def get_val_info(model, valloader, loss_fn, device, use_tqdm=False, num_classes=1, dIoU_conf=None, cfg_pp=None):
    model.eval()
    total_loss = 0.0
    is_multidataset = valloader.dataset.is_multidataset

    if num_classes == 1:
        total_intersect = 0.0
        total_union = 0.0
        iou = 0.0
    else:
        total_intersect = torch.zeros((num_classes), dtype=float)
        total_union = torch.zeros((num_classes), dtype=float)
        iou = torch.zeros((num_classes), dtype=float)

    if is_multidataset:
        dict_multidataset = {
                    'nuscenes':{'intersect':copy.deepcopy(total_intersect),
                                'union':copy.deepcopy(total_intersect)},
                    
                    'lyft':{'intersect':copy.deepcopy(total_intersect),
                                'union':copy.deepcopy(total_intersect)},}
        
        current_dataset = dict_multidataset['nuscenes']
        change_index = valloader.dataset.lyft_idx
        bsz = valloader.batch_size

    print('running eval...')
    loader = tqdm(valloader) if use_tqdm else valloader

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            allimgs, rots, trans, intrins, post_rots, post_trans, binimgs, rot_deg = batch

            if type(allimgs) == list:
                if len(allimgs) == 2:

                    allimgs, lidar_imgs = allimgs

                    preds = model([allimgs.to(device), lidar_imgs.to(device)], rots.to(device),
                              trans.to(device), intrins.to(
                                  device), post_rots.to(device),
                              post_trans.to(device), rot_deg.to(device))

                elif len(allimgs) == 3:

                    allimgs, lidar_imgs, points = allimgs

                    voxels, coors, num_points = points_to_voxel_loop(
                        points, cfg_pp)
                    voxels = voxels.to(device)
                    coors = coors.to(device)
                    num_points = num_points.to(device)

                    pp_data = [voxels, coors, num_points]
                    preds = model([allimgs.to(device), lidar_imgs.to(device)], pp_data, rots.to(device),
                              trans.to(device), intrins.to(
                                  device), post_rots.to(device),
                              post_trans.to(device), rot_deg.to(device))
                
            else:
                preds = model(allimgs.to(device), rots.to(device),
                              trans.to(device), intrins.to(
                                  device), post_rots.to(device),
                              post_trans.to(device))

            binimgs = binimgs.to(device)
            # breakpoint()
            # loss
            total_loss += loss_fn(preds, binimgs).item() * preds.shape[0]

            # iou
            if num_classes == 1:
                intersect, union, _ = get_batch_iou(preds, binimgs)
                total_intersect += intersect
                total_union += union
            else:
                intersect, union, _ = get_batch_iou_multi_class(
                    preds, binimgs, num_classes=num_classes)
                total_intersect += intersect
                total_union += union

            if is_multidataset:
                if (idx * bsz) >= change_index:
                    current_dataset = dict_multidataset['lyft']

                current_dataset['intersect'] += intersect
                current_dataset['union'] += union

    model.train()
    result_dict = {
        'loss': total_loss / len(valloader.dataset),
        'iou': total_intersect / total_union,
        'intersection': total_intersect,
        'union': total_union
    }

    if is_multidataset:
        dataset_iou = {}
        for k, dataset in dict_multidataset.items():
            dataset_iou[k] = dataset['intersect'] / dataset['union']
        
        result_dict['dataset_iou'] = dataset_iou

    return result_dict


def update_symlink(source_path, symlink_path):
    try:
        os.symlink(source_path, symlink_path)
    except FileExistsError:
        os.unlink(symlink_path)
        os.symlink(source_path, symlink_path)

def update_model(model_weights, logdir, model_name):

    path = os.path.join(logdir, model_name)
    try:
        torch.save(model_weights, path)
    except FileExistsError:
        os.remove(path)
        torch.save(model_weights, path)
