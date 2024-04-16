"""
File containing various functions for the manipulation of image and image-related samples
for the dataloaders

This file contains code from https://github.com/nv-tlabs/lift-splat-shoot
License available in https://github.com/nv-tlabs/lift-splat-shoot/blob/master/LICENSE

This file contains from MMCV v0.2.16, taken from: https://github.com/open-mmlab/mmcv/blob/v0.2.16/mmcv/image/transforms/resize.py
Under Apache License 2.0 available at https://github.com/open-mmlab/mmcv/blob/v0.2.16/LICENSE
"""

import torch, torchvision
import os
import cv2
import numpy as np

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

from PIL import Image
from pyquaternion import Quaternion

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


def get_lidar_img(nusc, cam_token, lidar_token, resize, crop, tok_lyft=None):

    cam = nusc.get('sample_data', cam_token)
    pointsensor = nusc.get('sample_data', lidar_token)

    if tok_lyft is None:
        pcl_path = os.path.join(nusc.dataroot, pointsensor['filename'])
        pc = LidarPointCloud.from_file(pcl_path)

    else:
        pc_path = tok_lyft+'.npy'
        pc_path = os.path.join(nusc.data_path, 'subsampled_lidar', pc_path)
        scan = np.load(pc_path)
        pc = LidarPointCloud(scan.T)
    #im = Image.open(os.path.join(nusc.dataroot, cam['filename']))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor',
                         pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    coloring = depths

    intrinsic = np.array(cs_record['camera_intrinsic'])

    # Apply transformations to intrinsic (view) matrix
    intrinsic[0, :] *= resize
    intrinsic[1, :] *= resize
    intrinsic[0, 2] = intrinsic[0, 2] - crop[0]
    intrinsic[1, 2] = intrinsic[1, 2] - crop[1]

    points = view_points(pc.points[:3, :], intrinsic, normalize=True)

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < (crop[2] - crop[0]))
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < (crop[3] - crop[1]))
    points = points[:, mask]
    coloring_2 = coloring[mask]

    pts_px = np.floor(points.T)
    pts_px[:, 2] = coloring_2

    img_pts = np.zeros((crop[3]-crop[1], crop[2]-crop[0])).astype(float)
    img_pts[pts_px[:, 1].astype(int), pts_px[:, 0].astype(int)] = coloring_2

    return img_pts

def get_lidar_img_aug(nusc, cam_token, lidar_token, resize, crop):

    cam = nusc.get('sample_data', cam_token)
    pointsensor = nusc.get('sample_data', lidar_token)
    pcl_path = os.path.join(nusc.dataroot, pointsensor['filename'])

    pc = LidarPointCloud.from_file(pcl_path)
    #im = Image.open(os.path.join(nusc.dataroot, cam['filename']))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor',
                         pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    coloring = depths

    intrinsic = np.array(cs_record['camera_intrinsic'])

    # Apply transformations to intrinsic (view) matrix
    intrinsic[0, :] *= resize
    intrinsic[1, :] *= resize
    intrinsic[0, 2] = intrinsic[0, 2] - crop[0]
    intrinsic[1, 2] = intrinsic[1, 2] - crop[1]

    points = view_points(pc.points[:3, :], intrinsic, normalize=True)

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < (crop[2] - crop[0]))
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < (crop[3] - crop[1]))
    points = points[:, mask]
    coloring_2 = coloring[mask]

    pts_px = np.floor(points.T)
    pts_px[:, 2] = coloring_2

    img_pts = np.zeros((crop[3]-crop[1], crop[2]-crop[0])).astype(float)
    img_pts[pts_px[:, 1].astype(int), pts_px[:, 0].astype(int)] = coloring_2

    return img_pts



class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denormalize_img = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage(),
))


normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))