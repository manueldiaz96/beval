import torch
import os
import random
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenesExplorer
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

from dataloaders.data_tools.tools_pc import get_lidar_data, read_point_cloud, lidar_visibility_map,\
                                random_sample_points, extract_pc_in_box2d

from dataloaders.data_tools.tools_img import img_transform, get_lidar_img, normalize_img

from dataloaders.data_tools.tools_map import get_gt_map_mask, gen_dx_bx, parse_label_nusc, \
                                             check_out_of_bounds, get_ego_pose


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, nusc_map=None, is_train=True, dataset_conf=None):

        assert type(dataset_conf) != type(None)
                    
        self.nusc = nusc
        self.helper = NuScenesExplorer(self.nusc)

        self.in_cs_frame = dataset_conf.in_cs_frame

        if self.in_cs_frame:
            print('Annotations loaded in the LiDAR Calibrated Sensor TF')
        else:
            print('Annotations loaded in the Ego Pose TF')

        self.is_train = is_train
        self.data_aug_conf = dataset_conf.data_aug_conf.to_dict()
        self.grid_conf = dataset_conf.grid_conf.to_dict()

        if not self.is_train:
            self.data_aug_conf['Ncams'] = 6
        else:
            self.data_aug_conf['Ncams'] = len(self.data_aug_conf['cams'])

        self.num_classes = len(dataset_conf.train_label)
        self.add_map = dataset_conf.add_map
        self.nusc_map = nusc_map

        self.use_visibility_map = dataset_conf.use_visibility_map

        self.vis_level = dataset_conf.vis_level # 1: 0-40% | 2: 40-60% | 3: 60-80% | 4: 80-100%  --> visibility on cameras 

        if self.use_visibility_map and self.vis_level == 0:
            self.vis_level = 1  # We adjust the visibility for ground truth since we're using visibility map 

        dx, bx, nx = gen_dx_bx(**self.grid_conf)
        
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.visibility_map = np.ones((nx[0],nx[1]))

        print("Minimum visibility Level:", {0: '0%', 1: '40%', 2: '60%', 3: '80%'}[self.vis_level])

        if not self.is_train:
            self.data_aug_conf['Ncams'] = 6

        assert dataset_conf.nr_conditions in ['','night','rain'], "Invalid weather condition"
        self.nr_conditions = dataset_conf.nr_conditions

        if self.num_classes == 1:
            self.train_label = dataset_conf.train_label[0]
        else:
            self.train_label = dataset_conf.train_label

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        self.cfg_pp = dataset_conf.cfg_pp.to_dict()

        self.fix_nuscenes_formatting()

        #print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        if self.nr_conditions != '':
            print('Getting samples ', self.nr_conditions)
            samples = [samp for samp in samples if       
                      self.nr_conditions in self.nusc.get('scene', samp['scene_token'])['description'].lower()]                   
        else:
            print('Getting all samples')

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
    def sample_augmentation(self, img_shape):
        W, H = img_shape
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim_nusc'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_crop_lim_nusc']))*newH) - fH
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
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_crop_lim_nusc']))*newH) - fH
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

    def get_image_data(self, rec, cams, use_lidar_img=None):
        imgs = []
        lidar_imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        if self.in_cs_frame:
            cs = self.nusc.get('calibrated_sensor',self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['calibrated_sensor_token'])
            cs_tran = -torch.Tensor(cs['translation'])
            cs_rot = Quaternion(cs['rotation']).inverse
            cs_rot = torch.Tensor(cs_rot.rotation_matrix)

        if use_lidar_img:
            lidar_top_data = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])


        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            if self.in_cs_frame:
                rot = cs_rot.matmul(rot)
                tran = cs_rot.matmul((tran + cs_tran).view(3,1)).view(3)

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, self.rotate = self.sample_augmentation(img.size)
            
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=self.rotate,
                                                     )

            if use_lidar_img:
                lidar_img = get_lidar_img(self.nusc, samp['token'], 
                                          lidar_top_data['token'], 
                                          resize, 
                                          crop)
                
                lidar_img = torch.from_numpy(lidar_img)
                
                lidar_imgs.append(lidar_img)

            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

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
        pts = get_lidar_data(self.nusc, rec, nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec, rot_aug):

        egopose = get_ego_pose(self.nusc, rec, self.in_cs_frame)

        img = np.zeros((self.nx[0], self.nx[1]))

        map_labels = ['drivable_area']#,'walkway']#,'lane_divider']

        rot_m = rot_aug.rotation_matrix
        aug_angle = (np.arctan2(rot_m[1,0],rot_m[0,0])*180.0/np.pi)
        if self.num_classes == 1 and self.train_label in map_labels :
            map_mask = get_gt_map_mask(self.nusc, self.nusc_map, rec, [self.train_label],
                                h = self.dx[1]*self.nx[1], w = self.dx[0]*self.nx[0],
                                canvas_size = (self.nx[0], self.nx[1]), aug_angle=aug_angle,
                                in_cs_frame=self.in_cs_frame, grid_config=self.grid_conf).squeeze(axis=0)
            
            img = map_mask.copy()

        elif self.add_map:
            for i in range(len(map_labels)):
                map_mask = get_gt_map_mask(self.nusc, self.nusc_map, rec, [map_labels[i]],
                                h = self.dx[1]*self.nx[1], w = self.dx[0]*self.nx[0],
                                canvas_size = (self.nx[0], self.nx[1]), aug_angle=aug_angle,
                                in_cs_frame=self.in_cs_frame, grid_config=self.grid_conf).squeeze(axis=0)
                
                img[map_mask.astype(bool)] = self.num_classes + i + 1

        
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)

            if self.train_label in map_labels:
                break

            if int(inst['visibility_token']) < self.vis_level:
                continue

            if self.num_classes == 1 :
                #print(self.train_label)
                if not inst['category_name'].split('.')[0] == self.train_label:
                    continue
                box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
                box.translate(egopose['trans'])
                box.rotate(egopose['rot'])

                if self.in_cs_frame:
                    box.translate(egopose['cs_trans'])
                    box.rotate(egopose['cs_rot'])

                out_of_x, out_of_z = check_out_of_bounds(box.center, self.grid_conf)
                                 
                # check if instance is out of bounds
                if out_of_x or out_of_z:
                    continue        

                box.rotate(rot_aug)

                pts = box.bottom_corners()[:2].T
                pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(img, [pts], 1.0)

            else:
                inst = self.nusc.get('sample_annotation', tok)
                label, idx = parse_label_nusc(inst['category_name'])

                if label in ['animal','static_object']:
                    continue

                if label not in self.train_label:
                    continue

                # add category for lyft
                box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
                box.translate(egopose['trans'])
                box.rotate(egopose['rot'])

                if self.in_cs_frame:
                    box.translate(egopose['cs_trans'])
                    box.rotate(egopose['cs_rot'])

                out_of_x, out_of_z = check_out_of_bounds(box.center, self.grid_conf)
                                 
                # check if instance is out of bounds
                if out_of_x or out_of_z:
                    continue        

                pts = box.bottom_corners()[:2].T
                pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(img, [pts], idx+1)

        if self.use_visibility_map:
            self.update_visibility_map(rec, rot_aug)
            img = img * self.visibility_map

        if self.add_map:
            #img += -1
            return torch.Tensor(img).long()
        else:  
            return torch.Tensor(img).unsqueeze(0)
                    

    def get_point_cloud(self, rec, rot_aug=0):
        
        if self.in_cs_frame:
            pts = get_lidar_data(self.nusc, rec, nsweeps=1, min_distance=2.2)[:4]
            pts = torch.from_numpy(np.swapaxes(pts, 0, 1))

        else:
            pts = read_point_cloud(self.nusc, rec)

        pts = extract_pc_in_box2d(pts, self.cfg_pp['pc_range'])
        pts = random_sample_points(pts, self.cfg_pp['n_points'])


        if rot_aug != 0 : 
            pts_xyz, pts_refl = torch.split(pts, [3,1], dim=1)

            # to_ego_frame = torch.from_numpy(Quaternion(axis=[0,0,1], degrees=-90).rotation_matrix).float()
            # pts_xyz = torch.matmul(to_ego_frame, pts_xyz.t())

            rot_matrix = torch.from_numpy(rot_aug.rotation_matrix).float()
            pts_xyz_rot = torch.matmul(rot_matrix, pts_xyz.t())

            pts = torch.cat((pts_xyz_rot.t(), pts_refl), dim=1)

        return pts
    
    def update_visibility_map(self, rec, rot_quat):
        points = self.get_point_cloud(rec, self.cfg_pp['pc_range'], self.cfg_pp['n_points'], rot_quat)

        # points_in = points[:,0] > 0

        # points = points[points_in]

        angles = np.arctan2(points[:,0],points[:,1]).numpy()
        ranges = np.sqrt(points[:,0]**2 + points[:,1]**2).numpy()

        grid_size = (int(self.nx[0]), int(self.nx[1])) # No idea why numba doesn't accept numpy.int64 to create zeros matrices, so we cast to 'int'
        resolution = float(self.dx[0])
    
        visibility_map = lidar_visibility_map(angles=angles, ranges=ranges, grid_size=grid_size, resolution=resolution)
        
        visibility_map = np.flip(visibility_map, -1)

        self.visibility_map = visibility_map.copy()

        
    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams
    
    def get_sample(self, tok):
        return self.nusc.get('sample', tok)

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)
