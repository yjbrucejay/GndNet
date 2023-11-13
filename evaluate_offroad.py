import argparse
import os
import shutil
import yaml
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import open3d as o3d

from model import GroundEstimatorNet
from utils.point_cloud_ops import points_to_voxel
from utils.utils import lidar_to_img, lidar_to_heightmap, segment_cloud
from utils.point_cloud_ops import points_to_voxel
from numba import jit,types

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

if use_cuda:
    print('setting gpu on gpu_id: 0')

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--config', default='config/config_kittiSem.yaml', type=str, metavar='PATH',
                    help='path to config file (default: none)')
parser.add_argument('-v', '--visualize', dest='visualize', action='store_true',
                    help='visualize model on validation set')
parser.add_argument('-gnd', '--visualize_gnd', dest='visualize_gnd', action='store_true',
                    help='visualize ground elevation')
parser.add_argument('--data_dir',
                    default="360_deg_fov/",
                    type=str, metavar='PATH', help='path to config file (default: none)')
args = parser.parse_args()

if os.path.isfile(args.config):
    print("using config file:", args.config)
    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    class ConfigClass:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    cfg = ConfigClass(**config_dict)  # convert python dict to class for ease of use
else:
    print("=> no config file found at '{}'".format(args.config))

print("setting batch_size to 1")
cfg.batch_size = 1

if args.visualize:
    # Ros Includes
    import rospy
    from utils.ros_utils import np2ros_pub_2, gnd_marker_pub
    from sensor_msgs.msg import PointCloud2
    from visualization_msgs.msg import Marker

    rospy.init_node('gnd_data_provider', anonymous=True)
    pcl_pub = rospy.Publisher("/kitti/velo/pointcloud", PointCloud2, queue_size=10)
    marker_pub_2 = rospy.Publisher("/kitti/gnd_marker_pred", Marker, queue_size=10)

model = GroundEstimatorNet(cfg)
optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=0.0005)


def get_GndSeg(sem_label, GndClasses):
    index = np.isin(sem_label, GndClasses)
    GndSeg = np.ones(sem_label.shape)
    GndSeg[index] = 0
    index = np.isin(sem_label, [0,1])
    GndSeg[index] = -1
    return GndSeg


@jit(nopython=True)
def remove_outliers(pred_GndSeg, GndSeg): # removes the points outside grid and unlabled points
    index = pred_GndSeg >= 0
    pred_GndSeg = pred_GndSeg[index]
    GndSeg = GndSeg[index]

    index = GndSeg >=0
    pred_GndSeg = pred_GndSeg[index]
    GndSeg = GndSeg[index]
    return 1-pred_GndSeg, 1-GndSeg


@jit(nopython=True)
def _shift_cloud(cloud, height):
    cloud += np.array([0,0,height,0], dtype=np.float32)
    return cloud


def InferGround(cloud):
    cloud = _shift_cloud(cloud[:,:4], cfg.lidar_height)
    voxels, coors, num_points = points_to_voxel(cloud, cfg.voxel_size, cfg.pc_range, cfg.max_points_voxel, True, cfg.max_voxels)
    voxels = torch.from_numpy(voxels).float().to(device)
    coors = torch.from_numpy(coors)
    coors = F.pad(coors, (1,0), 'constant', 0).float().to(device)
    num_points = torch.from_numpy(num_points).float().to(device)
    with torch.no_grad():
            output = model(voxels, coors, num_points)
    return output


def evaluate(data_dir):
    frames = os.listdir(data_dir)
    print(len(frames))
    for f in range(len(frames)):
        points_path = os.path.join(data_dir, "pcd_out_%06d.pcd" % f)
        point_clouds = o3d.io.read_point_cloud(points_path)
        points = np.asarray(point_clouds.points)
        padding = np.ones((points.shape[0], 1))
        points = np.concatenate((points, padding), axis=1)
        pred_gnd = InferGround(points)
        pred_gnd = pred_gnd.cpu().numpy()
        pred_GndSeg = segment_cloud(points.copy(), np.asarray(cfg.grid_range), cfg.voxel_size[0],
                                    elevation_map=pred_gnd.T, threshold=0.2)
        
        nonground_indices = np.where(pred_GndSeg == 1)[0]
        ground_indices = np.where(pred_GndSeg == 0)[0]
        nonground_points = points[nonground_indices, :3]  # 只选取前三列（x, y, z）
        ground_points = points[ground_indices, :3]

        pcd_nonground = o3d.geometry.PointCloud()
        pcd_nonground.points = o3d.utility.Vector3dVector(nonground_points)
        o3d.io.write_point_cloud("nonground_indice.ply", pcd_nonground)

        pcd_ground = o3d.geometry.PointCloud()
        pcd_ground.points = o3d.utility.Vector3dVector(ground_points)
        o3d.io.write_point_cloud("ground_indices.ply", pcd_ground)
        
        print(np.sum(pred_GndSeg))
        if args.visualize:
            np2ros_pub_2(points, pcl_pub, None, pred_GndSeg)
            if args.visualize_gnd:
                gnd_marker_pub(pred_gnd, marker_pub_2, cfg, color="red")


def main():
    # rospy.init_node('pcl2_pub_example', anonymous=True)
    global args
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        raise Exception('please specify checkpoint to load')

    evaluate(args.data_dir)


if __name__ == '__main__':
    main()