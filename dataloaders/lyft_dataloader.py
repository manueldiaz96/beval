from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import Box # NOQA

import torch
import numpy as np
import cv2
import os
import random

from pyquaternion import Quaternion
from PIL import Image
from lyft_dataset_sdk.utils.data_classes import Box


from dataloaders.data_tools.tools_pc import get_lidar_data, read_point_cloud_subsampled, lidar_visibility_map,\
                                random_sample_points, extract_pc_in_box2d

from dataloaders.data_tools.tools_img import img_transform, normalize_img, get_lidar_img

from dataloaders.data_tools.tools_map import parse_label_lyft, gen_dx_bx, check_out_of_bounds, get_ego_pose, MapManager

from dataloaders.data_tools.lyft_splits import create_splits_scenes



class LyftData(torch.utils.data.Dataset):
    def __init__(self, lyft, is_train=True, dataset_conf=None):

        self.lyft = lyft
        
        self.is_train = is_train
        self.data_aug_conf = dataset_conf.data_aug_conf.to_dict()
        self.grid_conf = dataset_conf.grid_conf.to_dict()
        self.num_classes = len(dataset_conf.train_label)
        self.add_map = dataset_conf.add_map      

        if not self.is_train:
            self.data_aug_conf['Ncams'] = 6
        else:
            self.data_aug_conf['Ncams'] = len(self.data_aug_conf['cams'])
        
        if self.num_classes == 1:
            self.train_label = dataset_conf.train_label[0]
        else:
            self.train_label = dataset_conf.train_label

        if self.add_map or self.train_label == 'drivable_area':
           self.lyft_map = MapManager(lyft, grid_config= self.grid_conf)

        dx, bx, nx = gen_dx_bx(**self.grid_conf)

        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.use_visibility_map = dataset_conf.use_visibility_map

        self.visibility_map = np.ones((nx[1],nx[0]))

        self.cfg_pp = dataset_conf.cfg_pp.to_dict()

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

    
    def get_scenes(self):
        # filter by scene split
        if self.is_train:
            split = 'train'
        else:
            split = 'val'
        scene_names = create_splits_scenes()[split]

        return scene_names
    
    def prepro(self):
        samples = [samp for samp in self.lyft.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.lyft.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams
    
    def sample_augmentation(self, img_shape):
        W, H = img_shape
        fH, fW = self.data_aug_conf['final_dim']

        if W == 1920:
            resize_factor = 0.8
        else:
            resize_factor = 1

        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim_lyft'])*resize_factor
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_crop_lim_lyft']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_crop_lim_lyft']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate
    
    def get_rot_quat(self):
        
        if self.is_train:
            rot_deg = torch.FloatTensor(1).uniform_(self.data_aug_conf['pc_rot'][0], self.data_aug_conf['pc_rot'][1])
            rot_quat = Quaternion(axis=[0,0,1], degrees=rot_deg)

        else:
            rot_quat = Quaternion(axis=[0,0,1], degrees=0)

        return rot_quat

    def get_image_data(self, rec, cams, use_lidar_img=False):

        imgs = []
        lidar_imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        if use_lidar_img:
            lidar_top_data = self.lyft.get('sample_data', rec['data']['LIDAR_TOP'])

        for cam in cams:
            samp = self.lyft.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.lyft.data_path, samp['filename'])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.lyft.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(img.size)
            
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )

            
            
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            if use_lidar_img:
                lidar_img = get_lidar_img(self.lyft, 
                                          samp['token'], 
                                          lidar_top_data['token'], 
                                          tok_lyft=rec['token'],
                                          resize=resize,
                                          crop=crop)
                
                lidar_img = torch.from_numpy(lidar_img)
                
                lidar_imgs.append(lidar_img)

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

            

        if use_lidar_img:
            return (torch.stack(imgs), torch.stack(lidar_imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

        else:
            return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.lyft, rec, nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_point_cloud(self, rec, rot_aug=0):

        pts = read_point_cloud_subsampled(self.lyft, rec)
        pts_rot = Quaternion(axis=[0,0,1], degrees=-90)

        pts = extract_pc_in_box2d(pts, self.cfg_pp['pc_range'])
        pts = random_sample_points(pts, self.cfg_pp['n_points'])

        pts_rot_matrix = torch.from_numpy(pts_rot.rotation_matrix).float()
        pts_xyz, pts_refl = torch.split(pts, [3,1], dim=1)
        pts_rot = torch.matmul(pts_rot_matrix, pts_xyz.t())
        pts = torch.cat((pts_rot.t(), pts_refl), dim=1)


        if rot_aug != 0 : 
            pts_xyz, pts_refl = torch.split(pts, [3,1], dim=1)

            rot_matrix = torch.from_numpy(rot_aug.rotation_matrix).float()

            pts_xyz_rot = torch.matmul(rot_matrix, pts_xyz.t())
            pts = torch.cat((pts_xyz_rot.t(), pts_refl), dim=1)

        return pts
    
    def get_binimg(self, rec, rot_aug):

        egopose = get_ego_pose(self.lyft, rec)

        img = np.zeros((self.nx[0], self.nx[1]))

        if self.add_map or (self.train_label=='drivable_area'):
            rot_m = rot_aug.rotation_matrix
            aug_angle = (np.arctan2(rot_m[1,0],rot_m[0,0])*180.0/np.pi)

            map_mask = self.lyft_map.get_map(self.lyft, rec, aug_angle)                

            if self.train_label=='drivable_area':
                img[map_mask.astype(bool)] = 1
            else:
                img[map_mask.astype(bool)] = self.num_classes + 1

        for tok in rec['anns']:

            inst = self.lyft.get('sample_annotation', tok)

            label, idx = parse_label_lyft(inst['category_name'])

            if self.num_classes == 1 :
                #print(self.train_label)
                if label != self.train_label:
                    continue
                box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
                box.translate(egopose['trans'])
                box.rotate(egopose['rot'])
                box.rotate(rot_aug)    

                out_of_x, out_of_z = check_out_of_bounds(box.center, self.grid_conf)
                                 
                # check if instance is out of bounds
                if out_of_x or out_of_z:
                    #print('Outside bounds', box.center)
                    continue        
                
                #print('Within bounds', box.center)
                pts = box.bottom_corners()[:2].T
                pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(img, [pts], 1.)
        
            else:

                if label == 'other':
                    continue
                if label not in self.train_label:
                    continue

                # add category for lyft
                box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
                box.translate(egopose['trans'])
                box.rotate(egopose['rot'])
                box.rotate(rot_aug)

                out_of_x, out_of_z = check_out_of_bounds(box.center, self.grid_conf)
                                 
                # check if instance is out of bounds
                if out_of_x or out_of_z:
                    continue    
                
                pts = box.bottom_corners()[:2].T
                pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(img, [pts], idx)


        if self.use_visibility_map:
            self.update_visibility_map(rec, rot_aug)
            img = img * self.visibility_map

        if self.add_map:
            return torch.Tensor(img).long()
        else:  
            return torch.Tensor(img).unsqueeze(0)
           
    def update_visibility_map(self, rec, rot_quat):

        points = self.get_point_cloud(rec, self.cfg_pp['pc_range'], self.cfg_pp['n_points'], rot_quat)

        angles = np.arctan2(points[:,0],points[:,1]).numpy()
        ranges = np.sqrt(points[:,0]**2 + points[:,1]**2).numpy()

        grid_size = (int(self.nx[0]), int(self.nx[1])) # No idea why numba doesn't accept numpy.int64 to create zeros matrices, so we cast to 'int'
        resolution = float(self.dx[0])
        
        visibility_map = lidar_visibility_map(angles=angles, ranges=ranges, grid_size=grid_size, resolution=resolution)

        # Align visibility map to GT
        visibility_map = np.flip(visibility_map, -1)
        visibility_map = np.rot90(visibility_map, k=-1)
        
        self.visibility_map = visibility_map
        
    def get_sample(self, tok):
        return self.lyft.get('sample', tok)
    
    def __str__(self):
        return f"""lyftData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)
