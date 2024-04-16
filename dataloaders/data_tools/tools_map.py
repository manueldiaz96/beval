import torch, math
from pyquaternion import Quaternion

import numpy as np
from PIL import Image
import cv2
import time

import matplotlib.pyplot as plt

### 

# Shared Map Tools


def check_out_of_bounds(center, grid_conf):

    out_of_bounds_x = (center[0] < grid_conf['xbound'][0]) or  (center[0] > grid_conf['xbound'][1])
    out_of_bounds_z = (center[1] < grid_conf['ybound'][0]) or  (center[1] > grid_conf['ybound'][1])

    return out_of_bounds_x, out_of_bounds_z

def gen_dx_bx(xbound, ybound, zbound, dbound=None):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])

    bx = torch.Tensor(
        [row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])

    nx = torch.Tensor([(row[1] - row[0]) / row[2]
                          for row in [xbound, ybound, zbound]]).long()

    return dx, bx, nx


###

# nuScenes Map Tools

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.eval.common.utils import quaternion_yaw

def get_gt_map_mask(nusc, nmap, rec, layer_name, h=100, w=100, canvas_size=(200, 200), 
                    plot_mask=False, plot_map=False, aug_angle=0, in_cs_frame=False,
                    grid_config=None):
    """
    returns ground truth the mask for a class given a sample token
    also plot mask or map if required.

    nusc: Nuscenes object
    nusc_map: NuScenes map object

    sample_token: token from nuscenes sample
    layer_name: Class name to retreive the mask as array
    h_w: height and width for the mask area
    canvas_size: canvas size to retreive the mask

    retrun: mask from layer name with canvas size
    """

    sample_token = rec['token']

    sample_rec = nusc.get('sample', sample_token)
    scene_rec = nusc.get('scene', sample_rec['scene_token'])

    map_name = nusc.get('log', scene_rec['log_token'])['location']
    nusc_map = nmap[map_name]

    sample_data_token = sample_rec['data']['LIDAR_TOP']
    ego_pose_token = nusc.get('sample_data', sample_data_token)[
        'ego_pose_token']
    ego_pose_data = nusc.get('ego_pose', ego_pose_token)
    x, y, _ = ego_pose_data['translation']
    q_orientation = ego_pose_data['rotation']
    rotation = Quaternion(q_orientation)

    if in_cs_frame:
        cs = nusc.get('calibrated_sensor', nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])['calibrated_sensor_token'])
        
        cs_trans = cs['translation']
        cs_rot = Quaternion(cs['rotation'])

        x += cs_trans[0]
        y += cs_trans[1]
        rotation *= cs_rot

    aug_angle = -aug_angle
    patch_angle = ((quaternion_yaw(rotation) / np.pi) * 180) - 90

    canvas_size_ext = (3*canvas_size[0], 3*canvas_size[1])

    w_ext = w * 3
    h_ext = h * 3

    patch_box = (x, y, w_ext, h_ext)

    map_mask = nusc_map.get_map_mask(
        patch_box, patch_angle + aug_angle, layer_name, canvas_size_ext)
    
    _, map_h, map_w = map_mask.shape

    top_m, bottom_m, left_m, right_m = calculate_map_margins(grid_config=grid_config,
                                                             map_h=map_h,
                                                             map_w=map_w,
                                                             canvas_size=canvas_size)
        
    map_mask = map_mask[:, top_m:bottom_m, left_m:right_m]
        
    if plot_mask == True:
        figsize = (24, 8)
        fig, ax = nusc_map.render_map_mask(
            patch_box, patch_angle, layer_name, canvas_size, figsize=figsize, n_row=1)

    if plot_map == True:
        bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')
        my_patch = (x-(h/2), y-(w/2), x+(h/2), y+(w/2))
        fig, ax = nusc_map.render_map_patch(
            my_patch, layer_name, figsize=(10, 10), bitmap=bitmap)
    
    return np.flip(map_mask, 2).copy()


def calculate_map_margins(grid_config, map_h, map_w, canvas_size):

    # Here we calculate the margins for cropping the map according to 
    # the wanted area of interest

    # Useful for when the car is not in the center of the map.

    top_margin = grid_config['xbound'][0]//grid_config['xbound'][2]
    top_margin = (map_h//2) + int(top_margin)
    top_margin = max(0, top_margin)

    bottom_margin = top_margin + canvas_size[0]

    # Since the NuscMap is flipped along the horizontal axis (ybound)
    # we flip the order of the bounds
    left_margin = -grid_config['ybound'][1]//grid_config['ybound'][2]
    left_margin = (map_w//2) + int(left_margin)
    left_margin = max(0, left_margin)

    right_margin = left_margin + canvas_size[1]

    return top_margin, bottom_margin, left_margin, right_margin


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                                       map_name=map_name) for map_name in [
        "singapore-hollandvillage",
        "singapore-queenstown",
        "boston-seaport",
        "singapore-onenorth",
    ]}
    return nusc_maps

def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx, angle_rot=0):
    egopose = nusc.get('ego_pose', nusc.get(
        'sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0],
                      egopose['translation'][1], np.cos(rot), np.sin(rot)])

    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
                         50.0, poly_names, line_names)
    for name in poly_names:
        for la in lmap[name]:
            pts = (la - bx) / dx
            plt.fill(pts[:, 1], pts[:, 0], c=(1.00, 0.50, 0.31), alpha=0.2)
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(0.0, 0.0, 1.0), alpha=0.5)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(159./255., 0.0, 1.0), alpha=0.5)


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
            )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys

def parse_label_nusc(category, num_classes=1):
        
        label = category.split('.')[0]

        idx = { 
            "human" : 2,
            "movable_object" : 1,
            "vehicle" : 0
            }

        if label in idx.keys():
            idx = idx[label]
        else:
            idx = -1

        return label, idx

def get_ego_pose(dataset, rec, in_cs_frame=False):

    egopose = dataset.get('ego_pose',
                            dataset.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    trans = -np.array(egopose['translation'])

    rot = Quaternion(egopose['rotation']).inverse

    egopose = {'trans': trans, 'rot': rot}

    if in_cs_frame:
        cs = dataset.get('calibrated_sensor', dataset.get('sample_data', rec['data']['LIDAR_TOP'])['calibrated_sensor_token'])
        cs_trans = -np.array(cs['translation'])
        cs_rot = Quaternion(cs['rotation']).inverse

        egopose['cs_rot'] = cs_rot
        egopose['cs_trans'] = cs_trans

    return egopose


#### 

# LYFT Map Tools

Image.MAX_IMAGE_PIXELS = 400000 * 400000


class MapManager(object):
    def __init__(self, lyft, grid_config):

        t = time.time()
        print('Loading map...')
        self.map_token = '53992ee3023e5494b90c316c183be829'

        self.map = lyft.get("map", self.map_token)['mask']

        self.map_raster = self.map.mask()
        self.map_raster_h, self.map_raster_w, _ = self.map_raster.shape

        self.map_ori_res = self.map.resolution

        self.grid_config = grid_config

        _, _, nx = gen_dx_bx(**self.grid_config)

        self.final_size = (int(nx[1]), int(nx[0]))

        self.map_limit_h = int(max(np.abs(grid_config['xbound'][0]), np.abs(grid_config['xbound'][1])) / self.map_ori_res)
        self.map_limit_w = int(max(np.abs(grid_config['ybound'][0]), np.abs(grid_config['ybound'][1])) / self.map_ori_res)
        
        print('Map loaded in {:.1f}s'.format(time.time()-t))
        

    def crop_image(self, image, x_px, y_px, margin=1):
        x_min = int(x_px - self.map_limit_w * margin)
        x_max = int(x_px + self.map_limit_w * margin)
        y_min = int(y_px - self.map_limit_h * margin)
        y_max = int(y_px + self.map_limit_h * margin)

        if x_max > self.map_raster_h:
            print('Bottom limit reached')
        if y_max > self.map_raster_w:
            print('Right limit reached')        
        if x_min < 0:
            print('Top limit reached')
        if y_min < 0:
            print('Left limit reached')

        cropped_image = image[y_min:y_max, x_min:x_max]

        return cropped_image
    
    def make_drivable_area(self, map):
        
        drivable_area = (map[:,:,  0]<251).astype(np.uint8)
        kernel = np.ones((5,5), dtype=np.uint8)

        drivable_area = cv2.morphologyEx(drivable_area, cv2.MORPH_CLOSE, kernel)

        return drivable_area
    
    def get_map(self, lyft, rec, aug_angle=0):
        
        
        pose = lyft.get('ego_pose',
                                lyft.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        
        pixel_coords = self.map.to_pixel_coords(pose["translation"][0], pose["translation"][1])
        
        cropped = self.crop_image(self.map_raster,
                                  pixel_coords[0], 
                                  pixel_coords[1], 
                                  margin=3)
        
        ypr_rad = Quaternion(pose["rotation"]).yaw_pitch_roll
        yaw_deg = -math.degrees(ypr_rad[0]) - 90
        yaw_deg += aug_angle

        #ego_car = cv2.RotatedRect(center=(cropped.shape[0]//2, cropped.shape[1]//2), size=(40,20), angle=-yaw_deg).points()
        #ego_car[:,0], ego_car[:,1] = ego_car[:,1].copy(), ego_car[:,0].copy()
        #cropped = cv2.fillPoly(cropped,[ego_car.astype(int)],(0,0,0))

        rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))

        ego_map = self.crop_image(
            rotated_cropped, rotated_cropped.shape[1] // 2, rotated_cropped.shape[0] // 2, margin=2
        )

        map_h, map_w, _ = ego_map.shape

        top_m, bottom_m, left_m, right_m = self.calculate_map_margins(map_h, map_w)

        ego_h, ego_w, _ = ego_map.shape

        #ego_map = self.add_borders(ego_map, top_m, bottom_m, left_m, right_m, map_w, map_h)

        ego_map = ego_map[top_m:bottom_m, left_m:right_m, :]

        ego_map = self.make_drivable_area(ego_map)

        final_map = cv2.resize(ego_map, self.final_size, interpolation = cv2.INTER_NEAREST)

        return final_map
    
    def calculate_map_margins(self, map_h, map_w):

        # Here we calculate the margins for cropping the map according to 
        # the wanted area of interest

        # Useful for when the car is not in the center of the map.
        # breakpoint()
        top_margin = int(self.grid_config['xbound'][0]/self.map_ori_res)
        top_margin = (map_h//2) + int(top_margin)
        # top_margin = max(0, top_margin)

        bottom_margin = int(self.grid_config['xbound'][1]/self.map_ori_res)
        bottom_margin = (map_h//2) + int(bottom_margin) 
        # bottom_margin = min(bottom_margin, map_h)

        left_margin = int(self.grid_config['ybound'][0]/self.map_ori_res) 
        left_margin = (map_w//2) + int(left_margin)
        # left_margin = max(0, left_margin)

        right_margin = int(self.grid_config['ybound'][1]/self.map_ori_res)
        right_margin = (map_w//2) + int(right_margin) 
        # right_margin = min(right_margin, map_w)

        return top_margin, bottom_margin, left_margin, right_margin
    
    def add_borders(self, map_raster, top_margin, bottom_margin, left_margin, right_margin, map_w, map_h):
        
        top_border, bottom_border, left_border, right_border = 0, 0, 0, 0

        if top_margin < 0:
            top_border = np.abs(top_margin)
        if bottom_margin > map_h:
            bottom_border = int(bottom_margin - map_h)
        if left_margin < 0:
            left_border = np.abs(left_margin)
        if right_margin > map_w:
            right_border = int(right_margin - map_w)

        if top_border!=0 or bottom_border!=0 or left_border !=0 or right_border!=0 :
            breakpoint()
        
        map_raster = cv2.copyMakeBorder(
                    map_raster,
                    top=top_border,
                    bottom=bottom_border,
                    left=left_border,
                    right=right_border,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )
        
        return map_raster

def parse_label_lyft(category):
    # All categories:
    # ['car', 'other_vehicle', 'bicycle', 
    #  'bus', 'pedestrian', 'truck',  
    #  'motorcycle', 'emergency_vehicle', 'animal']

    if category in ['car', 'other_vehicle', 'bicycle', 'bus', 'truck', 
                    'motorcycle', 'emergency_vehicle']:
        return 'vehicle', 1
    
    elif category in ['pedestrian']:
        return 'human', 2
    
    else:
        return 'other', 3  #animal