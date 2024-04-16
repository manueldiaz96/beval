"""
Modified from https://github.com/nv-tlabs/lift-splat-shoot
License available in https://github.com/nv-tlabs/lift-splat-shoot/blob/master/LICENSE
"""

import torch
import os
import numpy as np

from dataloaders.data_tools.tools_map import get_nusc_maps
from dataloaders.multi_dataset import MultiDataset

SUPPORTED_DATASETS = ['nuscenes', 'lyft']    

   
class LSS_dataloader(MultiDataset):
    def __init__(self, *args, **kwargs):
        super(LSS_dataloader, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        dataset, n_index = self.get_dataset(index)
        rec = dataset.ixes[n_index]

        cams = dataset.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = dataset.get_image_data(rec, cams)

        rot_quat = dataset.get_rot_quat()

        binimg = dataset.get_binimg(rec, rot_quat)

        rot_deg = torch.Tensor(rot_quat.rotation_matrix)

        return imgs, rots, trans, intrins, post_rots, post_trans, binimg, rot_deg
        
    
class LAPT_dataloader(MultiDataset):
    def __init__(self, *args, **kwargs):
        super(LAPT_dataloader, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        dataset, n_index = self.get_dataset(index)
        rec = dataset.ixes[n_index]

        cams = dataset.choose_cams()
        imgs, lidar_imgs, rots, trans, intrins, post_rots, post_trans = dataset.get_image_data(rec, cams, use_lidar_img=True)

        rot_quat = dataset.get_rot_quat()

        binimg = dataset.get_binimg(rec, rot_quat)

        rot_deg = torch.Tensor(rot_quat.rotation_matrix)

        return [imgs, lidar_imgs], rots, trans, intrins, post_rots, post_trans, binimg, rot_deg
    

class LAPT_PP_dataloader(MultiDataset):
    def __init__(self, *args, **kwargs):
        super(LAPT_PP_dataloader, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        dataset, n_index = self.get_dataset(index)
        rec = dataset.ixes[n_index]

        cams = dataset.choose_cams()
        imgs, lidar_imgs, rots, trans, intrins, post_rots, post_trans = dataset.get_image_data(rec, cams, use_lidar_img=True)

        rot_quat = dataset.get_rot_quat()

        points = dataset.get_point_cloud(rec, rot_aug=rot_quat)

        binimg = dataset.get_binimg(rec, rot_quat)

        rot_deg = torch.Tensor(rot_quat.rotation_matrix)

        return [imgs, lidar_imgs, points], rots, trans, intrins, post_rots, post_trans, binimg, rot_deg
    

def worker_rnd_init(x):
    np.random.seed(42 + x)

def check_supported_datasets(dataset_list):

    for d in dataset_list:
        assert d in SUPPORTED_DATASETS, "{} dataset not supported. Only 'nuscenes' and 'lyft' are currently supported".format(d)
        

def compile_data(data_conf):

    check_supported_datasets(data_conf.datasets)

    dataset_dict = {}
    
    if 'nuscenes' in data_conf.datasets:

        print("Loading nuScenes dataset...", end='\r')
        
        from nuscenes.nuscenes import NuScenes

        dataroot_nusc = os.environ['NUSCENES']

        nusc = NuScenes(version='v1.0-{}'.format(data_conf.version),
                        dataroot=dataroot_nusc,
                        verbose=False)
        
        dataset_dict['nuscenes'] = {'nusc': nusc,
                                    'nusc_map': get_nusc_maps(dataroot_nusc)}
        
        print('   Using nuScenes dataset    ')
        
    if 'lyft' in data_conf.datasets:

        print("Loading Lyft dataset...", end='\r')
        
        from lyft_dataset_sdk.lyftdataset import LyftDataset

        if data_conf.version == "mini":
            dataroot_lyft = os.environ['LYFT_MINI']
        else:
            dataroot_lyft = os.environ['LYFT']

        lyft = LyftDataset( data_path=dataroot_lyft, 
                            json_path=os.path.join(dataroot_lyft,'train_data'), 
                            verbose=False)
        
        dataset_dict['lyft'] = lyft

        print('   Using Lyft dataset   ')    

    parser_choice = {'lss-semseg': LSS_dataloader,
                     'lapt-semseg': LAPT_dataloader,
                     'lapt-pp-semseg': LAPT_PP_dataloader,
                    }   

    parser = parser_choice[data_conf.parser]                  

    traindata = parser(datasets=dataset_dict, dataset_config=data_conf, is_train=True)

    valdata = parser(datasets=dataset_dict, dataset_config=data_conf, is_train=False)

   
    # else: 
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=data_conf.batch_size,
                                                shuffle=data_conf.shuffle_train,
                                                num_workers=data_conf.n_workers,
                                                drop_last=True,
                                                worker_init_fn=worker_rnd_init)
    
    if not data_conf.shuffle_train:
        print("\n\n\n"+"ATTENTION!\n"+"TRAINLOADER IS NOT BEING SHUFFLED\n"+"\n\n")

    valloader = torch.utils.data.DataLoader(valdata, batch_size=data_conf.batch_size,
                                            shuffle=False,
                                            num_workers=data_conf.n_workers)

    return trainloader, valloader

