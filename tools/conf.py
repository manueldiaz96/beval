from yacs.config import CfgNode as _CfgNode 
from os import path
import shutil

'''
   Base config for the dataloader and training

    Params for dataloader:

        version (string): which split to load from the dataset -> ['mini', 'trainval']

        datasets (List): which datasets to load -> check SUPPORTED_DATASETS

        data_aug_conf (Dict): dictionary containing the bounds for the data augmentation 
                       operations to be done. 
                    
                        - resize_lim (tuple): limits to choose a resize factor from.
                        - final_dim (tuple): final image dimensions in (height, width)
                        - bot_crop_lim (tuple): limits to include from the bottom of 
                                                the image when doing the crop
                        - rot_lim (tuple): rotation limits in deg for the image
                        - rand_flip (bool): parameter to decide if rhe images are 
                                            randomly flipped along the horizontal axis
                        - cams (list): list of cams to choose from if n_cams < 6 
                        - n_cams (int) : number of camera images to use during training
                        - pc_rot (tuple): rotation limits in deg for the point cloud
                        

        grid_conf (Dict): BEV space parameters such as extents and resolution

                        - xbound (list): list containing the bounds in meters for the 
                                         vertical axis (top to bottom) in the BEV (x axis 
                                         for the ego vehicle) as well as the resolution.
                        - ybound (list): list containing the bounds in meters for the 
                                         horizontal (left to right) axis in the BEV 
                                         (y axis for the ego vehicle) as well as the 
                                         resolution.
                        - zbound (list): list containing the bounds in meters for the 
                                         limits on the Z axis for point cloud voxelization
                        - dbound (list): Legacy parameter used only for LSS-based networks

        bsz (int): batch size to be used per dataloader

        n_workers (int): number of workers to be used per dataloader

        parser_name (string): type of datloader to return, will depend on the NN being 
                              trained

        dist (bool): Flag indicating if dataloaders for distributed training should 
                     be returned 

        num_classes (int): number of classes that the dataloader should return. 
                           this helps generate the proper GT depending if we are 
                           training on only one class or multiple.

        train_label: If binary segmentation is being trained

        add_map (bool): Tells us to add the drivable area information in the GT when 

        cfg_pp (Dict) : Dictionary with the information to format the point clouds
                        if needed for the model

        nr_conditions (string): [NuScenes Only] string to filter the scenes with
                                night or rain conditions

        vis_level 0: 0-40% | 1: 40-60% | 2: 60-80% | 3: 80-100%  --> visibility on cameras 

    '''


def convert_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary."""
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, _CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                'Key {} with value {} is not a valid type; valid types: {}'.format(
                    '.'.join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


class CfgNode(_CfgNode):
    
    def to_dict(self):
        return convert_to_dict(self)
    
config = CfgNode()

#----------------------- Dataset------------------------

config.data_config = CfgNode()

config.data_config.parser = 'debug'

config.data_config.version = 'mini'

config.data_config.n_workers = 0
config.data_config.shuffle_train = False
config.data_config.batch_size = 1

config.data_config.use_visibility_map = False
config.data_config.datasets = ['nuscenes', 'lyft']
config.data_config.train_label = ['vehicle']
config.data_config.add_map = True
config.data_config.in_cs_frame = False
config.data_config.vis_level = 0
config.data_config.nr_conditions = ''


# Data Augmentation Params
config.data_config.data_aug_conf = CfgNode()

config.data_config.data_aug_conf.resize_lim_nusc = (0.225, 0.225)
config.data_config.data_aug_conf.bot_crop_lim_nusc = (0., 0.22) 

config.data_config.data_aug_conf.resize_lim_lyft = (0.35, 0.35)
config.data_config.data_aug_conf.bot_crop_lim_lyft = (0.20, 0.22) 

config.data_config.data_aug_conf.final_dim = (128, 352) 

config.data_config.data_aug_conf.rot_lim = (0, 0)  
config.data_config.data_aug_conf.rand_flip = False
config.data_config.data_aug_conf.cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
config.data_config.data_aug_conf.ncams = 6
config.data_config.data_aug_conf.pc_rot = (0,0)

# BEV Params
config.data_config.grid_conf = CfgNode()

config.data_config.grid_conf.xbound = [-50., 50., 0.5] # Top - bottom
config.data_config.grid_conf.ybound = [-50., 50., 0.5] # Left - Right
config.data_config.grid_conf.zbound = [-10.0, 10.0, 20.0]
config.data_config.grid_conf.dbound = [4.0, 45.0, 1.0]

# Point cloud / Point Pillars Params
config.data_config.cfg_pp = CfgNode()

config.data_config.cfg_pp.num_points = 100000
config.data_config.cfg_pp.n_points = 34720


#----------------------- Train Params ------------------------

config.train_config = CfgNode()

config.train_config.logdir = '/tmp/PLACEHOLDER'

config.train_config.epochs = 400
config.train_config.gpuid = 0

config.train_config.max_grad_norm = 5.0
config.train_config.loss_weight = 2.13
config.train_config.lr = 1e-3
config.train_config.weight_decay = 1e-7

config.train_config.use_fpn = False
config.train_config.fusion_method = 'add'

# CFG Point Pillars

config.train_config.cfg_pp = CfgNode()

config.train_config.cfg_pp.num_points = config.data_config.cfg_pp.num_points
config.train_config.cfg_pp.n_points = config.data_config.cfg_pp.n_points

config.train_config.cfg_pp.max_points_voxel = 100
config.train_config.cfg_pp.max_voxels = 10000
config.train_config.cfg_pp.input_features = 4
config.train_config.cfg_pp.use_norm = True
config.train_config.cfg_pp.vfe_filters = [64]
config.train_config.cfg_pp.with_distance = False

# Cfg for InferenceEngines

config.train_config.model_path = None



def save_conf(conf_pth, logdir):
    shutil.copy(conf_pth, path.join(logdir,'config.yaml'))

def get_cfgs(cfg_path=None):
    """ First get default config. Then merge cfg_dict. Then merge according to args. """

    cfg = config.clone()

    if cfg_path is not None and path.exists(cfg_path):
        cfg.merge_from_file(cfg_path)

    cfg = update_cfg_pp(cfg)

    train_config = cfg.train_config.clone()
    data_config = cfg.data_config.clone()
    
    return train_config, data_config


def update_cfg_pp(cfg):
    
    pc_range = [cfg.data_config.grid_conf.ybound[0], 
                cfg.data_config.grid_conf.xbound[0], 
                -4, 
                cfg.data_config.grid_conf.ybound[1], 
                cfg.data_config.grid_conf.xbound[1], 
                4 ]
    
    voxel_size = [cfg.data_config.grid_conf.xbound[2], 
                  cfg.data_config.grid_conf.ybound[2], 
                  8]
    
    cfg.train_config.cfg_pp.pc_range = pc_range
    cfg.train_config.cfg_pp.voxel_size = voxel_size
    cfg.train_config.cfg_pp.batch_size = cfg.data_config.batch_size

    cfg.data_config.cfg_pp.pc_range = pc_range

    return cfg


