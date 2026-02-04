#!/usr/bin/env python3
"""Quick test script to verify elevation_mapping_cupy integration only."""
import os
import numpy as np
import cv2
import rosbag
import ros_numpy
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
from Helpers.traversability_helpers import yaw_from_quaternion
from elevation_mapping_cupy import ElevationMap, Parameter

# Config
BAG_PATH = "/mnt/sbackup/Server_3/harshr/m2p2_data_bags/G3_elev_dlio/G3_main_bags/BL_2024-09-04_19-25-36_chunk0000_processed.bag"
OUTPUT_DIR = "/tmp/elev_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Transforms
T_base_lidar = np.eye(4)
T_base_lidar[:3, 3] = [0.240, 0.000, 0.476]

# BEV Config
grid_size = 100
grid_res = 0.1
map_length = 20.0
max_canopy_height = 2.0

# Setup elevation mapping
cprint("Setting up elevation mapping...", 'yellow')
weight_file = "/home/robotixx/elevation_ws/src/elevation_mapping_cupy/elevation_mapping_cupy/config/core/weights.dat"
plugin_config_file = "/home/robotixx/elevation_ws/src/elevation_mapping_cupy/elevation_mapping_cupy/config/core/plugin_config.yaml"

param = Parameter(use_chainer=False, weight_file=weight_file, plugin_config_file=plugin_config_file)
param.map_length = map_length
param.resolution = grid_res
param.update()

elevation_map = ElevationMap(param)
param.enable_drift_compensation = False
cprint("Elevation Mapping Initialized!", 'green')

# Step 1: Pre-scan odom
cprint("Pass 1: Pre-scanning DLIO Odometry...", 'yellow')
bag = rosbag.Bag(BAG_PATH)
all_odom_data = []
for topic, msg, t in bag.read_messages(topics=['/dlio/odom_node/odom']):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    all_odom_data.append({
        'ts': msg.header.stamp.to_sec(),
        'x': p.x, 'y': p.y, 'z': p.z,
        'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w
    })
bag.close()
all_odom_data.sort(key=lambda x: x['ts'])
odom_timestamps = np.array([x['ts'] for x in all_odom_data])
cprint(f"Loaded {len(all_odom_data)} odom messages.", 'green')

def get_odom_at_time(ts):
    if not all_odom_data: return None
    idx = np.searchsorted(odom_timestamps, ts)
    candidates = []
    if idx < len(all_odom_data): candidates.append(idx)
    if idx > 0: candidates.append(idx - 1)
    if not candidates: return None
    best_idx = min(candidates, key=lambda i: abs(all_odom_data[i]['ts'] - ts))
    return all_odom_data[best_idx]

def update_elevation_map(lidar_points, timestamp):
    closest_odom = get_odom_at_time(timestamp)
    if closest_odom is None: return
    dt = abs(closest_odom['ts'] - timestamp)
    if dt > 0.2: return

    r = R.from_quat([closest_odom['qx'], closest_odom['qy'], closest_odom['qz'], closest_odom['qw']])
    T_odom_base = np.eye(4)
    T_odom_base[:3, :3] = r.as_matrix()
    T_odom_base[:3, 3] = [closest_odom['x'], closest_odom['y'], closest_odom['z']]
    
    T_odom_lidar = T_odom_base @ T_base_lidar
    R_lidar = T_odom_lidar[:3, :3]
    t_lidar = T_odom_lidar[:3, 3]

    map_center = t_lidar.copy()
    map_center[2] = 0.0
    elevation_map.move_to(map_center, np.eye(3))

    pts_clean = np.ascontiguousarray(lidar_points.reshape(-1, 3)).astype(np.float32)
    dist = np.linalg.norm(pts_clean[:, :2], axis=1)
    valid_mask = (dist > 1.0) & (dist < 15.0)
    pts_clean = pts_clean[valid_mask]
    
    lidar_z_threshold = max_canopy_height - T_base_lidar[2, 3]
    height_mask = pts_clean[:, 2] < lidar_z_threshold
    pts_clean = pts_clean[height_mask]

    if len(pts_clean) > 0:
        elevation_map.input_pointcloud(pts_clean, ['x', 'y', 'z'], R_lidar, t_lidar, 0.005, 0.005)
    elevation_map.update_variance()
    elevation_map.update_time()

def get_synced_elevation_bev(target_ts):
    closest_odom = get_odom_at_time(target_ts)
    if closest_odom is None: return None

    elev_data = elevation_map.get_layer("elevation")
    if hasattr(elev_data, 'get'):
        elev_data = elev_data.get()
    elev_data = np.nan_to_num(elev_data, nan=-999.0)

    map_n, map_m = elev_data.shape
    cx, cy = map_m / 2.0, map_n / 2.0
    
    ryaw = yaw_from_quaternion(closest_odom['qx'], closest_odom['qy'], closest_odom['qz'], closest_odom['qw'])
    angle_deg = -np.degrees(ryaw) + 180
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    
    rotated_map = cv2.warpAffine(elev_data.astype(np.float32), M, (map_m, map_n),
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=-999.0)
    
    robot_row_in_image = int(grid_size * 0.90)
    start_row = int(cy - robot_row_in_image)
    end_row = int(start_row + grid_size)
    start_col = int(cx - (grid_size // 2))
    end_col = int(cx + (grid_size // 2))

    if start_row < 0 or end_row > map_n or start_col < 0 or end_col > map_m:
        return None

    bev_elev = rotated_map[start_row:end_row, start_col:end_col]
    bev_elev[bev_elev == -999.0] = np.nan
    if not np.all(np.isnan(bev_elev)):
        bev_elev -= closest_odom['z']
        bev_elev += T_base_lidar[2, 3]
    return bev_elev.astype(np.float32)

# Step 2: Process LiDAR to build map, save samples
cprint("Pass 2: Processing LiDAR and saving elevation samples...", 'yellow')
bag = rosbag.Bag(BAG_PATH)
lidar_count = 0
saved_count = 0

for topic, msg, t in bag.read_messages(topics=['/sensor_suite/ouster/points']):
    lidar_points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
    if len(lidar_points.shape) > 2:
        lidar_points = lidar_points.reshape(-1, 3)
    
    ts = msg.header.stamp.to_sec()
    update_elevation_map(lidar_points, ts)
    lidar_count += 1
    
    # Save every 50th frame
    if lidar_count % 50 == 0:
        bev = get_synced_elevation_bev(ts)
        if bev is not None:
            # Save as npy
            npy_path = os.path.join(OUTPUT_DIR, f"elev_{lidar_count:05d}.npy")
            np.save(npy_path, bev)
            
            # Save as image for visualization
            elev_clean = np.nan_to_num(bev, nan=0.0)
            norm_elev = cv2.normalize(elev_clean, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img_path = os.path.join(OUTPUT_DIR, f"elev_{lidar_count:05d}.png")
            cv2.imwrite(img_path, norm_elev)
            
            saved_count += 1
            cprint(f"Saved frame {lidar_count} -> min={np.nanmin(bev):.2f}, max={np.nanmax(bev):.2f}", 'cyan')
    
    if lidar_count % 100 == 0:
        print(f"Processed {lidar_count} LiDAR scans...")

bag.close()
cprint(f"\nDone! Saved {saved_count} elevation samples to {OUTPUT_DIR}", 'green')
cprint("Check the PNG files to verify elevation data is working!", 'green')
