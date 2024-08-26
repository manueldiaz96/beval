"""
This file contains code from https://github.com/nv-tlabs/lift-splat-shoot
License available in https://github.com/nv-tlabs/lift-splat-shoot/blob/master/LICENSE
"""


import torch

from models.LAPTNet_PP import compile_model
from dataloaders.data import compile_data
from tools import (compile_loss, get_val_info)

from tools import get_cfgs

def test(cfg=None, weight_path=None):
    
    train_config, data_config = get_cfgs(cfg)
    
    _, valloader = compile_data(data_config)

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
    
    print("Loading weights under:", weight_path)

    weight_dict = torch.load(weight_path)
    model.load_state_dict(weight_dict)

    model.to(device)

    if num_classes == 1 and not data_config.add_map:
        loss_fn = compile_loss(num_classes, data_config.add_map, train_config.gpuid, task='semseg')

    else:
        loss_fn, text_labels = compile_loss(num_classes, data_config.add_map, train_config.gpuid, task='semseg', train_label=data_config.train_label[0])


    print('Dataloader samples:', len(valloader))

    model.eval()
    print('Validation:')
    val_info = get_val_info(model, valloader, loss_fn, device, 
                            num_classes=num_classes, use_tqdm=True, cfg_pp=cfg_pp)
    
    if num_classes == 1:
        print('{} IoU: {:.3f}'.format(data_config.train_label[0], val_info['iou'][0]*100))
        print('{} Loss: {:.3f}'.format(data_config.train_label[0], val_info['loss']))

    else:
        for i in range(len(text_labels)):
            print('{} IoU: {:.3f}'.format(text_labels[i], val_info['iou'][i]*100))
        print('Loss: {:.3f}'.format(val_info['loss']))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='Evaluate the LAPTNet model.')
  parser.add_argument('--cfg', type=str, required=True,
                      help='Path to the config file.')
  parser.add_argument('--weights', type=str, required=True,
                      help='Path to weights of the trained model.')
  args = parser.parse_args()

  test(cfg=args.cfg, weight_path=args.weights)

