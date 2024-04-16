"""
File containing various functions for the manipulation of point clouds and 
point-cloud-related samples for the dataloaders

This file contains code from https://github.com/nv-tlabs/lift-splat-shoot
License available in https://github.com/nv-tlabs/lift-splat-shoot/blob/master/LICENSE

"""

import torch
import numba
import numpy as np
import os

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from functools import reduce

from scipy.spatial import Delaunay
from pyquaternion import Quaternion


def random_sample_points(cloud, N):

    cloud = torch.from_numpy(np.asarray(cloud)).float()

    points_count = cloud.shape[0]

    if(points_count > 1):
        prob = torch.randperm(points_count)  # sampling without replacement
        if(points_count > N):
            idx = prob[:N]
            sampled_cloud = cloud[idx]

        else:
            r = int(N/points_count)
            cloud = cloud.repeat(r+1, 1)
            sampled_cloud = cloud[:N]

    else:
        sampled_cloud = torch.ones(N, 3)

    return sampled_cloud  # .cpu().numpy()

def in_hull(p, hull):

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0

def extract_pc_in_box2d(pc, pc_range):
    
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d = [pc_range[0],
                pc_range[1],
                pc_range[3],
                pc_range[4]]

    box2d_corners = np.zeros((4, 2))
    box2d_corners[0, :] = [box2d[0], box2d[1]]
    box2d_corners[1, :] = [box2d[2], box2d[1]]
    box2d_corners[2, :] = [box2d[2], box2d[3]]
    box2d_corners[3, :] = [box2d[0], box2d[3]]
    box2d_roi_inds = in_hull(pc[:, 0:2], box2d_corners)

    return pc[box2d_roi_inds, :]

"""
Taken from https://github.com/anshulpaigwar/GndNet/blob/master/misc/point_cloud_ops_test.py
Under MIT License terms available at: https://github.com/anshulpaigwar/GndNet/blob/master/LICENSE
"""

@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points=35,
                                    max_voxels=20000):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num

@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
                            voxel_size,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=35,
                            max_voxels=20000):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    lower_bound = coors_range[:3]
    upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def points_to_voxel(points,
                     voxel_size,
                     coors_range,
                     max_points=35,
                     reverse_index=True,
                     max_voxels=20000):
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud) 
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output 
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    else:
        voxel_num = _points_to_voxel_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    # voxels[:, :, -3:] = voxels[:, :, :3] - \
    #     voxels[:, :, :3].sum(axis=1, keepdims=True)/num_points_per_voxel.reshape(-1, 1, 1)
    return voxels, coors, num_points_per_voxel


@numba.jit(nopython=True)
def bound_points_jit(points, upper_bound, lower_bound):
    # to use nopython=True, np.bool is not supported. so you need
    # convert result to np.bool after this function.
    N = points.shape[0]
    ndim = points.shape[1]
    keep_indices = np.zeros((N, ), dtype=np.int32)
    success = 0
    for i in range(N):
        success = 1
        for j in range(ndim):
            if points[i, j] < lower_bound[j] or points[i, j] >= upper_bound[j]:
                success = 0
                break
        keep_indices[i] = success
    return keep_indices



def read_point_cloud(nusc, sample_rec):
    """
    Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
    :param pc_path: Path of the pointcloud file on disk.
    :return: point cloud instance (x, y, z, reflectance).
    """
    pc_path = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])[
        'filename']

    pc_path = os.path.join(nusc.dataroot, pc_path)

    assert pc_path.endswith('.bin'), 'Unsupported filetype {}'.format(pc_path)

    nbr_dims = LidarPointCloud.nbr_dims()

    scan = np.fromfile(pc_path, dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :nbr_dims]

    return points





def points_to_voxel_loop(points, cfg):

    B = points.shape[0]
    voxels = []
    coors = []
    num_points = []

    points = points.numpy()
    for i in range(B):
        v, c, n = points_to_voxel(points[i],
                                  cfg['voxel_size'],
                                  cfg['pc_range'],
                                  cfg['max_points_voxel'],
                                  True,
                                  cfg['max_voxels'])

        c = torch.from_numpy(c)
        c = torch.nn.functional.pad(c, (1, 0), 'constant', i)
        voxels.append(torch.from_numpy(v))
        coors.append(c)
        num_points.append(torch.from_numpy(n))

    voxels = torch.cat(voxels).float().cuda()
    coors = torch.cat(coors).float().cuda()
    num_points = torch.cat(num_points).float().cuda()

    return voxels, coors, num_points



def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((5, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor',
                          ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                       inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(
            os.path.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get(
            'ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get(
            'calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(
            np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)
        # breakpoint()
        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points


@numba.jit(nopython=True)
def lidar_visibility_map(angles, ranges, grid_size, resolution=0.5, observer_ij=None):

    '''
    lidar_visibility_map_ij(observer_ij, angles, ranges, map_size, resolution) -> numpy.array
    
    Modified from original Cython syntax to run with Numba JIT:
    https://github.com/danieldugas/pymap2d/blob/master/CMap2D.pyx#L1496

    Takes the angles and ranges from a 2D point cloud as well as 
    the grid_size and the resolution and outputs the visibility map.

    Inputs:        
        - angles : array containing the angle value for each point in the 
                   2D point cloud. 
                   angles = atan2(horizontal_axis, vertical_axis)

        - range : array containing the distance value for each point in the 
                  2D point cloud from the observer coordinates
                  range = sqrt(horizontal_axis**2 + vertical_axis**2)

        - grid_size : tuple of visibility map width and height in cells

        - resolution : resolution for the visibility map in meters/cell

        - observer_ij : cell coordinates for the position of the observer 
                        with respect to the map. If it is not given, we 
                        assume that it is the center of the grid.

    Output:
        - visibility_map : 2D array encoding if a cell is visible (1) or not (-1) 
    '''
    
    assert len(angles) == len(ranges), "Ranges and Angles arrays need to have the same size"

    if observer_ij is None:
        observer_ij = [grid_size[0]//2, grid_size[1]//2]

    visibility_map = np.zeros(grid_size)
    
    o_i = observer_ij[0]
    o_j = observer_ij[1]

    for i in range(len(angles)):
        angle = angles[i]
        max_r = ranges[i] / resolution
        i_inc_unit = np.cos(angle)
        j_inc_unit = np.sin(angle)
        # Stretch the ray so that every 1 unit in the ray direction lands on a cell in i or j
        i_abs_inc = abs(i_inc_unit)
        j_abs_inc = abs(j_inc_unit)
        raystretch = 1. / i_abs_inc if i_abs_inc >= j_abs_inc else 1. / j_abs_inc
        i_inc = i_inc_unit * raystretch
        j_inc = j_inc_unit * raystretch
        # max amount of increments before crossing the grid border
        if i_inc == 0:
            max_inc = np.int64(((grid_size[1] - 1 - o_j) / j_inc) if j_inc >= 0 else (o_j / -j_inc))
        elif j_inc == 0:
            max_inc = np.int64(((grid_size[0] - 1 - o_i) / i_inc) if i_inc >= 0 else (o_i / -i_inc))
        else:
            max_i_inc = np.int64(((grid_size[1] - 1 - o_j) / j_inc) if j_inc >= 0 else (o_j / -j_inc))
            max_j_inc = np.int64(((grid_size[0] - 1 - o_i) / i_inc) if i_inc >= 0 else (o_i / -i_inc))
            max_inc = max_i_inc if max_i_inc <= max_j_inc else max_j_inc
        # Trace a ray
        n_i = o_i + 0
        n_j = o_j + 0
        for n in range(1, max_inc-1):
            n_i += i_inc
            in_i = np.int64(n_i)
            in_j = np.int64(n_j)
            di = ( in_i - o_i )
            dj = ( in_j - o_j )
            r = np.sqrt(di*di + dj*dj)
            visibility_map[in_i, in_j] = 1
            
            n_j += j_inc
            in_i = np.int64(n_i)
            in_j = np.int64(n_j)
            di = ( in_i - o_i )
            dj = ( in_j - o_j )
            r = np.sqrt(di*di + dj*dj)
            visibility_map[in_i, in_j] = 1
            
            if r > max_r:
                break

    return visibility_map

###

# Lyft Specific

def read_point_cloud_subsampled(lyft, sample_rec):
    """
    Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
    :param pc_path: Path of the pointcloud file on disk.
    :return: point cloud instance (x, y, z, reflectance).
    """
    
    pc_path = sample_rec['token']+'.npy'

    pc_path = os.path.join(lyft.data_path, 'subsampled_lidar', pc_path)

    scan = np.load(pc_path)

    return scan

def subsample_point_cloud(spherical_coords_pc, theta_min, theta_max, phi_min, phi_max, nb_theta_sectors, nb_phi_sectors):
    
    '''
    Function used to subsample the PC
    Lyft dataset has a 64 layers LIDAR, using it with models trained on Nuscenes requires subsampling the PC to 32 layers
    Spherical quadrants are created, the points are then projected onto their corresponding angular quadrant,
    then a single point from each quadrant is randomly sampled and returned 

    Inputs:

       spherical_coords_pc [np.array] -> (N,4) array containing the N points in spherical
                                         coordinates (rho, theta, phi, intensity)

       theta_min [float] -> minimum value of theta (azimuth) to consider for sampling 

       theta_max [float] -> maximum value of theta (azimuth) to consider for sampling

       phi_min [float] -> minimum value of phi (inclination) to consider for sampling

       phi_max [float] -> maximum value of phi (inclination) to consider for sampling

       nb_theta_sectors [float] -> Number of subdivisions in theta, can be seen also as 
                                   the number of rings in the LiDAR
    
       nb_phi_sectors [float] -> Number of subdivisions in phi  
    '''
   
    
    N, _ = spherical_coords_pc.shape

    subsampled_pc_spherical = np.zeros((nb_phi_sectors*nb_theta_sectors, 4))
    delta_theta = (theta_max-theta_min) / nb_theta_sectors
    delta_phi = (phi_max-phi_min) / nb_phi_sectors

    theta_segments = np.arange(theta_min, theta_max, delta_theta)
    phi_segments = np.arange(phi_min, phi_max, delta_phi)
    edges = [theta_segments, phi_segments]

        
    #https://stackoverflow.com/questions/70717111/how-can-i-efficiently-bin-3d-point-cloud-data-based-on-point-coordinates-and-des
    #2D binning of spherical_coords with the bins phi_segments and theta_segments
    #Returned values is array of indices mapping each lidar point to its corresponding bin
    
    # Needs to be allocated in order to work with numba

    
    coords = np.full((N, len(edges)), 0)

    for i, b in enumerate(edges):
        coords[:, i] = np.digitize(spherical_coords_pc[:,i+1], b, right=True)
    
    indices_sorted_array = np.lexsort((coords[:,1], coords[:,0]))
    coords = coords[indices_sorted_array] #Sort the angular sector array 
    spherical_coords_pc = spherical_coords_pc[indices_sorted_array]
    points_inside_current_bin = [np.array((0.,0.))]
    j=0

    for i, iterat in enumerate(coords):
            
        if (iterat != points_inside_current_bin[-1]).any(): #If the last element is different from current
            # ipdb.set_trace()
            # print(f"There are {len(points_inside_current_bin)} points in the sector nb :{points_inside_current_bin[-1]}")
            random_index = np.random.randint(0, len(points_inside_current_bin))
            subsampled_pc_spherical[j, :] = spherical_coords_pc[i-random_index,:] #Pick a random point from the list of all points within a quadrant
            points_inside_current_bin = [iterat]

            j = j+1
        else:
            points_inside_current_bin.append(iterat)

    subsampled_pc_spherical = subsampled_pc_spherical[~np.all(subsampled_pc_spherical ==0., axis=1)]

    # print(f"Input PC had {spherical_coords_pc.shape[0]} points, truncated PC has {subsampled_pc_spherical.shape[0]} points")

    return subsampled_pc_spherical  

#@numba.jit(nopython=True)
def spherical_to_cartesian(np_spherical): #assumes input [rho, theta , phi, i]
    rho = np_spherical[0]
    theta = np_spherical[1]
    phi = np_spherical[2]
    I = np_spherical[3]

    X = rho * np.cos(phi)*np.sin(theta)
    Y = rho * np.sin(phi)*np.sin(theta)
    Z = rho * np.cos(theta)

    pc_cartesian = np.vstack((X,Y,Z, I))

    return pc_cartesian

@numba.jit(nopython=True)
def cartesian_to_spherical(np_cartesian): #assumes input [x,y,z,i]
    X = np_cartesian[0]
    Y = np_cartesian[1]
    Z = np_cartesian[2]
    I = np_cartesian[3]

    rho = np.sqrt(X**2 + Y**2 + Z**2) 
    theta = np.arctan2(np.sqrt(X**2+Y**2), Z)
    phi = np.arctan2(Y,X)

    pc_spherical = np.vstack((rho, theta, phi, I))

    return pc_spherical
