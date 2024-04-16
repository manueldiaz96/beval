"""
This file contains code from https://github.com/nv-tlabs/lift-splat-shoot
License available in https://github.com/nv-tlabs/lift-splat-shoot/blob/master/LICENSE
"""


import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os
import logging

from models.LAPTNet_PP import compile_model
from dataloaders.data import compile_data
from tools import (compile_loss, get_batch_iou, get_batch_iou_multi_class, 
                    get_val_info, points_to_voxel_loop, update_model)

from tools import get_cfgs, save_conf


def train(cfg=None):
    
    train_config, data_config = get_cfgs(cfg)
    
    trainloader, valloader = compile_data(data_config)

    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device(f'cuda:{train_config.gpuid}')

    num_classes = len(data_config.train_label) 

    if data_config.add_map:
        num_classes += 2 #Drivable area + Unknown

    data_aug_conf = data_config.data_aug_conf.to_dict()
    grid_conf = data_config.grid_conf.to_dict()

    cfg_pp = train_config.cfg_pp.to_dict()

    model = compile_model(grid_conf, 
                          data_aug_conf, 
                          fusion_method=train_config.fusion_method,
                          outC=num_classes, 
                          use_fpn=train_config.use_fpn, 
                          cfg_pp=cfg_pp)

    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=train_config.lr,
                           weight_decay=train_config.weight_decay)


    if num_classes == 1 and not data_config.add_map:
        loss_fn = compile_loss(num_classes,  data_config.add_map, train_config.gpuid, task='semseg')

    else:
        loss_fn, text_labels = compile_loss(num_classes,  data_config.add_map, train_config.gpuid, task='semseg', train_label= data_config.train_label[0])

    writer = SummaryWriter(logdir=train_config.logdir)
    save_conf(cfg, train_config.logdir)

    model.train()
    counter = 0
    best_iou = 0

    print('Dataloader samples:', len(trainloader))

    for epoch in range(train_config.epochs):
        np.random.seed()
        for batchi, batch in enumerate(trainloader):
            
            data, rots, trans, intrins, post_rots, post_trans, binimgs, rot_deg = batch

            opt.zero_grad()
            
            t0 = time()

            imgs, lidar_imgs, points = data

            voxels, coors, num_points = points_to_voxel_loop(points, cfg_pp)

            voxels = voxels.to(device)
            coors = coors.to(device)
            num_points = num_points.to(device)

            pp_data = [voxels, coors, num_points]

            imgs = imgs.to(device)
            lidar_imgs = lidar_imgs.to(device)
            rots = rots.to(device)
            trans = trans.to(device)
            intrins = intrins.to(device)
            post_rots = post_rots.to(device)
            post_trans = post_trans.to(device)
            rot_deg = rot_deg.to(device)

            preds = model([imgs, lidar_imgs], pp_data, rots, trans,
                          intrins, post_rots, post_trans, rot_deg)
            binimgs = binimgs.to(device)

            loss = 0
            loss = loss_fn(preds, binimgs)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            opt.step()

            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                if num_classes == 1:
                    _, _, iou = get_batch_iou(preds, binimgs)
                    writer.add_scalar('train/iou', iou, counter)

                else:
                    _, _, iou = get_batch_iou_multi_class(
                        preds, binimgs, num_classes=num_classes)
                    writer.add_scalar('train/iou', iou.mean(), counter)
                    for c in range(num_classes):
                        if iou[c] != 1:
                            # breakpoint()
                            writer.add_scalar(
                                'train/iou_cls_{}'.format(text_labels[c]), iou[c], counter)

                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)
                
        model.eval()

        print('Validation:')
        val_info = get_val_info(model, valloader, loss_fn, device, 
                                num_classes=num_classes, use_tqdm=True, cfg_pp=cfg_pp)
        print('VAL', val_info)
        writer.add_scalar('val/loss', val_info['loss'], counter)

        if num_classes == 1:
            writer.add_scalar('val/iou', val_info['iou'], counter)

            if 'dataset_iou' in val_info.keys():
                for d in val_info['dataset_iou']:
                    writer.add_scalar('multidataset/{}_iou'.format(d), 
                                      val_info['dataset_iou'][d], counter)

        else:
            writer.add_scalar('val/iou', val_info['iou'][1:].mean(), counter)

            if 'dataset_iou' in val_info.keys():
                for d in val_info['dataset_iou']:
                        writer.add_scalar('multidataset/{}_iou'.format(d), 
                                        val_info['dataset_iou'][d][1:].mean(), 
                                        counter)

            for c in range(num_classes):
                writer.add_scalar(
                    'val/iou_cls_{}'.format(text_labels[c]),  val_info['iou'][c], counter)

                if 'dataset_iou' in val_info.keys():
                    for d in val_info['dataset_iou']:
                        writer.add_scalar('multidataset/{}_iou_cls_{}'.format(d,text_labels[c]), 
                                        val_info['dataset_iou'][d][c], 
                                        counter)

        mname = os.path.join(train_config.logdir, "model{}.pt".format(counter))
        optim_path =  os.path.join(train_config.logdir, 'optim-latest.pt')

        print('saving', mname)

        torch.save(model.state_dict(), mname)
        torch.save(opt.state_dict(), optim_path)

        update_model(model.state_dict(), train_config.logdir, "model_latest.pt")

        if num_classes==1 and val_info['iou'] > best_iou:
            best_iou = val_info['iou']
            update_model(model.state_dict(), train_config.logdir, "model_best.pt")

        elif num_classes>1 and val_info['iou'][1:].mean() > best_iou:
            best_iou = val_info['iou'][1:].mean()
            update_model(model.state_dict(), train_config.logdir, "model_best.pt")
            

        model.train()


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='Train the LAPTNet model.')
  parser.add_argument('--cfg', type=str, required=True,
                      help='Path to the config file.')
  args = parser.parse_args()

  train(cfg=args.cfg)

