#!/usr/bin/env python3
import rospy
import os
import copy
import pickle
import numpy as np
import cv2
import rosbag
import argparse
# --- Imports required by Original Script 1 ---
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, CompressedImage, Joy, Imu, MagneticField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped # Needed if cmd_vel is TwistStamped
from std_msgs.msg import Float32MultiArray # Assuming /status might be this, adjust if not
# --- Imports needed for Elevation ---
from grid_map_msgs.msg import GridMap
# --- Imports for Elevation Mapping Cupy ---
from scipy.spatial.transform import Rotation as R
from elevation_mapping_cupy import ElevationMap, Parameter
# --- Matplotlib setup for non-interactive backend ---
import matplotlib
matplotlib.use('Agg') # <<< IMPORTANT: Set backend BEFORE importing pyplot
import matplotlib.pyplot as plt
# --- End Matplotlib setup ---
from mpl_toolkits.axes_grid1 import make_axes_locatable
# --- Original Script 1 Helpers ---
from cv_bridge import CvBridge, CvBridgeError
from termcolor import cprint
import Helpers.depth_utiles as image_processing
from Helpers.sensor_fusion import imu_processor
from Helpers.data_calculation import DataCollection
from Helpers.traversability_helpers import extract_poses_from_path, process_traversability_single

import threading
import ros_numpy
import torch
import warnings
import traceback 
import logging
import gc

# Suppress RuntimeWarning from numpy related to NaN comparisons if needed
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in reduce")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in true_divide") # For normalization
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice") # For nanmean
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib') # Suppress Matplotlib UserWarning about threads

class ListenRecordData:
    def __init__(self, bag_path, bag_name_prefix, time_threshold=0.1, logger= None):
        
        self.logger = logger if logger else logging.getLogger(bag_name_prefix)
        
        self.bag_name_prefix = bag_name_prefix
        self.time_threshold = time_threshold

        self.data_collection = DataCollection()
        self.imu_processor = imu_processor()

        # History arrays (from Script1)
        self.odom_msgs = np.zeros((200, 3), dtype=np.float16)
        self.gyro_msgs = np.zeros((400, 3), dtype=np.float32)
        self.accel_msgs = np.zeros((400, 3), dtype=np.float32)
        self.velocity_msgs = np.zeros((5, 2), dtype=np.float32)
        self.cmd_vel_history = np.zeros((10,2), dtype=np.float32)
        self.roll_pitch_yaw = np.zeros((400, 3), dtype=np.float32)

        # Latest messages holders (from Script1, RGB removed)
        self.cmd_vel = None
        self.thermal = None
        self.depth = None
        self.depth_map = None
        self.odom = None
        self.joy_msg = None
        self.lidar_points = None
        self.dlio_final_msg = None

        self.counter = 0
        self.imu_counter = 0
        self.depth_counter = 0
        self.husky_msg = [0.0, 0.0, 0.0, 0.0] # Keep status placeholder

        # GPS placeholders (from Script1)
        self.previous_itow = 0
        self.previous_pose = np.zeros(3)

        # Constants (from Script1)
        self.WEIGHT_ACCEL_Z_JERK = 0.075
        self.WEIGHT_GYRO_X_ACCEL = 0.475
        self.WEIGHT_GYRO_Y_ACCEL = 0.475

        # Raw data storage
        self.thermal_data_raw = [] # Stores {'timestamp':..., 'msg':...}
        self.elevation_data_raw = [] # Add storage for raw elevation maps

        # Transformation (from Script1)
        self.transform_matrix = np.array([
            [0.007983, -0.999967, -0.00133,  0.000709],
            [0.02735,   0.001548, -0.999625, -0.072334],
            [0.999594,  0.007944,  0.027362, -0.208889],
            [0.0,       0.0,       0.0,       1.0      ]
        ])

        # ROI settings for elevation map
        self.roi_width = 100  # Width of the ROI in grid cells
        self.roi_height = 100  # Height of the ROI in grid cells
        self.roi_offset_x = -70  # Offset from the center in x-direction (positive moves ROI forward)
        self.roi_offset_y = 0

        # --- Elevation Mapping Cupy Configuration ---
        self.T_base_lidar = np.eye(4)
        self.T_base_lidar[:3, 3] = [0.240, 0.000, 0.476]  # Lidar position relative to base
        
        # BEV Grid Config (matching reference code)
        self.grid_size = 100
        self.grid_res = 0.1
        self.map_length = 20.0
        self.max_canopy_height = 2.0  # Filter points above this height (relative to robot)
        
        # DLIO Odom Storage for efficient lookup
        self.all_odom_data = []  # sorted list of dicts
        self.odom_timestamps = np.array([])  # for fast search
        self.elevation_map = None  # Will be initialized in setup

        # Aggregated data dictionary (Script1 MINUS RGB)
        self.msg_data = {
            'thermal_msg': [],
            'depth_msg': [],
            'sparse_depth': [],
            'odom_msg': [],
            'odom_1sec_msg': [],
            'cmd_vel_msg': [],
            'velocity_msg': [],
            'just_velocity_msg': [],
            'accel_msg': [],
            'time_stamp': [],
            'roll_pitch_yaw': [],
            'gyro_msg': [],
            'husky_msg': [],
            'lat_lon_heading_msg': [],
            'joy': []
        }

        self.bridge = CvBridge()
        self.cam_info_msg = None
        self.k_matrix_orig = None
        self.original_shape_orig = None

        # Setup elevation mapping cupy before processing bag
        self.setup_elevation_mapping()
        
        # Pre-process bag for DLIO odom (first pass)
        self.preprocess_odom(bag_path)
        
        self.process_bag(bag_path)
        self.elevation_data_raw.sort(key=lambda x: x['timestamp'])
        rospy.loginfo(f"[{self.bag_name_prefix}] Collected {len(self.thermal_data_raw)} raw thermal messages.")
        rospy.loginfo(f"[{self.bag_name_prefix}] Collected {len(self.elevation_data_raw)} raw elevation messages.")

    def setup_elevation_mapping(self):
        """Initialize elevation_mapping_cupy with parameters."""
        try:
            # Config file paths for elevation mapping cupy
            weight_file = "/home/robotixx/elevation_ws/src/elevation_mapping_cupy/elevation_mapping_cupy/config/core/weights.dat"
            plugin_config_file = "/home/robotixx/elevation_ws/src/elevation_mapping_cupy/elevation_mapping_cupy/config/core/plugin_config.yaml"
            
            self.param = Parameter(
                use_chainer=False, 
                weight_file=str(weight_file), 
                plugin_config_file=str(plugin_config_file)
            )
            
            self.param.map_length = self.map_length
            self.param.resolution = self.grid_res
            self.param.update()
            
            self.elevation_map = ElevationMap(self.param)
            self.param.enable_drift_compensation = False
            
            cprint(f"[{self.bag_name_prefix}] Elevation Mapping Cupy Initialized.", 'green')
            
        except Exception as e:
            cprint(f"[{self.bag_name_prefix}] Failed to init Elevation Mapping: {e}", 'red')
            traceback.print_exc()
            self.elevation_map = None

    def preprocess_odom(self, bag_path):
        """Pass 1: Read all DLIO odometry for efficient future lookup and 3D pose."""
        cprint(f"[{self.bag_name_prefix}] Pass 1: Pre-scanning DLIO Odometry...", 'yellow')
        try:
            bag = rosbag.Bag(bag_path)
        except rosbag.bag.BagException as e:
            rospy.logerr(f"Error opening bag file {bag_path}: {e}")
            return
            
        odom_topic = '/dlio/odom_node/odom'
        
        self.all_odom_data = []
        
        for topic, msg, t in bag.read_messages(topics=[odom_topic]):
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            self.all_odom_data.append({
                'ts': msg.header.stamp.to_sec(),
                'x': p.x, 'y': p.y, 'z': p.z,
                'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w
            })
            
        bag.close()
        
        # Sort just in case
        self.all_odom_data.sort(key=lambda x: x['ts'])
        self.odom_timestamps = np.array([x['ts'] for x in self.all_odom_data])
        cprint(f"[{self.bag_name_prefix}] Loaded {len(self.all_odom_data)} DLIO odom messages.", 'green')

    def get_odom_at_time(self, ts):
        """Find closest DLIO odom (efficient sorted search)."""
        if not self.all_odom_data: 
            return None
        
        idx = np.searchsorted(self.odom_timestamps, ts)
        candidates = []
        if idx < len(self.all_odom_data): 
            candidates.append(idx)
        if idx > 0: 
            candidates.append(idx - 1)
        
        if not candidates: 
            return None
        
        best_idx = min(candidates, key=lambda i: abs(self.all_odom_data[i]['ts'] - ts))
        return self.all_odom_data[best_idx]

    def update_elevation_map(self, lidar_points, timestamp):
        """Update elevation map with LiDAR scan using DLIO pose."""
        if self.elevation_map is None:
            return
            
        closest_odom = self.get_odom_at_time(timestamp)
        if closest_odom is None: 
            return 
        
        dt = abs(closest_odom['ts'] - timestamp)
        if dt > 0.2:
            return 

        # Build rotation matrix from quaternion using scipy
        r = R.from_quat([closest_odom['qx'], closest_odom['qy'], closest_odom['qz'], closest_odom['qw']])
        
        T_odom_base = np.eye(4)
        T_odom_base[:3, :3] = r.as_matrix()
        T_odom_base[:3, 3] = [closest_odom['x'], closest_odom['y'], closest_odom['z']]
        
        T_odom_lidar = T_odom_base @ self.T_base_lidar
        R_lidar = T_odom_lidar[:3, :3]
        t_lidar = T_odom_lidar[:3, 3]

        map_center = t_lidar.copy()
        map_center[2] = 0.0 
        
        # IMPORTANT: Use np.eye(3), NOT the robot rotation!
        self.elevation_map.move_to(map_center, np.eye(3))

        pts_clean = np.ascontiguousarray(lidar_points[:, :3].reshape(-1, 3)).astype(np.float32)
        dist = np.linalg.norm(pts_clean[:, :2], axis=1)
        valid_mask = (dist > 1.0) & (dist < 15.0)
        pts_clean = pts_clean[valid_mask]
        
        # Canopy Filtering
        lidar_z_threshold = self.max_canopy_height - self.T_base_lidar[2, 3]
        height_mask = pts_clean[:, 2] < lidar_z_threshold
        pts_clean = pts_clean[height_mask]

        if len(pts_clean) > 0:
            self.elevation_map.input_pointcloud(
                pts_clean, 
                ['x', 'y', 'z'], 
                R_lidar, 
                t_lidar, 
                0.005, 
                0.005 
            )
            
        self.elevation_map.update_variance()
        self.elevation_map.update_time()

    def get_synced_elevation_bev(self, target_ts):
        """Get robot-centric BEV elevation map at the given timestamp."""
        if self.elevation_map is None:
            return None
            
        try:
            closest_odom = self.get_odom_at_time(target_ts)
            if closest_odom is None: 
                return None

            # Get raw map data
            elev_data = self.elevation_map.get_layer("elevation")
            if hasattr(elev_data, 'get'):
                elev_data = elev_data.get()
            elev_data = np.nan_to_num(elev_data, nan=-999.0)

            map_n, map_m = elev_data.shape 
            cx = map_m / 2.0
            cy = map_n / 2.0
            
            # Use Helper: yaw_from_quaternion
            from Helpers.traversability_helpers import yaw_from_quaternion
            ryaw = yaw_from_quaternion(closest_odom['qx'], closest_odom['qy'], closest_odom['qz'], closest_odom['qw'])
            
            angle_deg = -np.degrees(ryaw) + 180 
            M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
            
            rotated_map = cv2.warpAffine(elev_data.astype(np.float32), M, (map_m, map_n), 
                                         flags=cv2.INTER_NEAREST, 
                                         borderMode=cv2.BORDER_CONSTANT, 
                                         borderValue=-999.0)
            
            robot_row_in_image = int(self.grid_size * 0.90) 
            
            start_row = int(cy - robot_row_in_image)
            end_row   = int(start_row + self.grid_size)
            start_col = int(cx - (self.grid_size // 2))
            end_col   = int(cx + (self.grid_size // 2))

            if start_row < 0 or end_row > map_n or start_col < 0 or end_col > map_m:
                return None

            bev_elev = rotated_map[start_row:end_row, start_col:end_col]
            
            bev_elev[bev_elev == -999.0] = np.nan
            if not np.all(np.isnan(bev_elev)):
                bev_elev -= closest_odom['z']
                lidar_height = self.T_base_lidar[2, 3] 
                bev_elev += lidar_height
                
            return bev_elev.astype(np.float32)

        except Exception as e:
            traceback.print_exc()
            return None


    def process_bag(self, bag_path):
        try:
             bag = rosbag.Bag(bag_path)
        except rosbag.bag.BagException as e:
             rospy.logerr(f"Error opening bag file {bag_path}: {e}")
             return

        topics = [
            '/sensor_suite/lwir/lwir/image_raw/compressed', # Thermal
            '/sensor_suite/ouster/points',                 # Lidar
            '/sensor_suite/lwir/lwir/camera_info',         # Thermal Calib
            '/odometry/filtered',                          # Odom
            '/husky_velocity_controller/cmd_vel_out',      # Cmd Vel
            '/joy_teleop/joy',                             # Joy
            '/sensor_suite/witmotion_imu/imu',             # IMU
            '/sensor_suite/witmotion_imu/magnetometer',    # Mag
            '/status',                                     # Husky Status
            '/elevation_mapping/elevation_map_raw',        # **** ADDED ****
            '/dlio/odom_node/path'
        ]
        topics = [t for t in topics if t]

        total_messages = bag.get_message_count(topic_filters=topics)
        processed_count = 0
        cprint(f"Processing bag: {os.path.basename(bag_path)} ({total_messages} relevant messages)")

        for topic, msg, t in bag.read_messages(topics=topics):
            try:
                if topic == '/sensor_suite/witmotion_imu/imu': self.imu_callback(msg, t)
                elif topic == '/sensor_suite/lwir/lwir/image_raw/compressed': self.thermal_image_callback(msg, t)
                elif topic == '/sensor_suite/ouster/points': self.lidar_callback(msg, t)
                elif topic == '/sensor_suite/lwir/lwir/camera_info': self.calib_callback(msg, t)
                elif topic == '/husky_velocity_controller/cmd_vel_out': self.cmd_vel_callback(msg, t)
                elif topic == '/odometry/filtered': self.odom_callback(msg, t)
                elif topic == '/sensor_suite/witmotion_imu/magnetometer': self.mag_callback(msg, t)
                elif topic == '/joy_teleop/joy': self.joy_callback(msg, t)
                elif topic == '/status': self.status_callback(msg, t)
                elif topic == '/elevation_mapping/elevation_map_raw': self.elevation_map_callback(msg, t)
                elif topic == '/dlio/odom_node/path': self.dlio_path_callback(msg, t)

                processed_count += 1
                if processed_count % 1000 == 0: # Print progress every 1000 messages
                    cprint(f"[{self.bag_name_prefix}] Processed {processed_count}/{total_messages} messages...", 'magenta', end='\r')

            except Exception as e:
                 rospy.logwarn(f"[{self.bag_name_prefix}] Error processing message from topic {topic} at time {t.to_sec()}: {e}")
                 # traceback.print_exc() # Uncomment for more detail if needed

        cprint(f"\n[{self.bag_name_prefix}] Finished reading bag.", 'green')
        bag.close()

    def thermal_image_callback(self, msg: CompressedImage, t):
        try:
            self.thermal = msg
            ts = msg.header.stamp.to_sec()
            
            # Capture elevation BEV at this moment (while map state is current)
            bev_elev = self.get_synced_elevation_bev(ts)
            
            self.thermal_data_raw.append({
                'timestamp': ts, 
                'msg': msg,
                'elevation_bev': bev_elev  # Store BEV captured at this moment
            })
        except Exception as e:
            rospy.logerr(f"[{self.bag_name_prefix}] Error in thermal_image_callback: {e}")

    def lidar_callback(self, msg: PointCloud2, t):
        try:
            point_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
            intensity_field_exists = any(field.name == 'intensity' for field in msg.fields)
            coords = np.column_stack((
                point_cloud['x'].flatten(), point_cloud['y'].flatten(), point_cloud['z'].flatten()
            ))
            if intensity_field_exists:
                 intensities = point_cloud['intensity'].flatten()
                 if intensities.ndim > 1: intensities = intensities[:,0]
                 self.lidar_points = np.column_stack((coords, intensities))
            else:
                 self.lidar_points = coords

            # Update elevation map with this LiDAR scan
            timestamp = msg.header.stamp.to_sec()
            self.update_elevation_map(self.lidar_points, timestamp)

            if self.cam_info_msg is not None and self.thermal is not None:
                self.process_depth_maps() # Original logic
        except Exception as e:
            rospy.logwarn(f"[{self.bag_name_prefix}] Error in lidar_callback: {e}")
            self.lidar_points = None

    def calib_callback(self, msg: CameraInfo, t):
        self.cam_info_msg = msg

    def imu_callback(self, msg: Imu, t):
        try:
            if self.imu_counter <= 600:
                self.imu_processor.beta = 0.8;
                self.imu_counter += 1
            else:
                self.imu_processor.beta = 0.05

            self.imu_processor.imu_update(msg)
            self.gyro_msgs = np.roll(self.gyro_msgs, -1, axis=0)
            self.accel_msgs = np.roll(self.accel_msgs, -1, axis=0)
            self.gyro_msgs[-1] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            self.accel_msgs[-1] = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
            self.roll_pitch_yaw = np.roll(self.roll_pitch_yaw, -1, axis=0)
            self.roll_pitch_yaw[-1] = np.radians([self.imu_processor.roll, self.imu_processor.pitch, self.imu_processor.heading])
            self.data_collection.imu_buffer.append({
                'timestamp': t.to_sec(), 'gyro': self.gyro_msgs[-1].copy(), 'accel': self.accel_msgs[-1].copy()
            })
        except Exception as e:
            rospy.logwarn(f"[{self.bag_name_prefix}] Error in imu_callback: {e}")

    def cmd_vel_callback(self, msg, t):
        try:
            # Roll the cmd_vel history array and store the latest cmd_vel
            self.cmd_vel_history = np.roll(self.cmd_vel_history, -1, axis=0)
            self.cmd_vel_history[-1] = np.array([
                msg.twist.linear.x, 
                msg.twist.angular.z
            ])
            self.cmd_vel = msg

            # Don't print every time, too verbose
            # print(f"Received messages :: {self.counter} -----")

            self.velocity_msgs = np.roll(self.velocity_msgs, -1, axis=0)
            if self.odom_msgs.shape[0] > 0: # Ensure odom_msgs is not empty
                 self.velocity_msgs[-1] = [self.odom_msgs[-1, 0], self.odom_msgs[-1, 2]]
            else:
                 self.velocity_msgs[-1] = [0.0, 0.0] # Default if no odom yet

            # Check if all necessary messages have arrived *at least once*
            # The exact timing match happens during saving
            if (
                self.cmd_vel is not None and
                self.thermal is not None and
                self.depth is not None and
                self.depth_map is not None and
                self.odom is not None
            ):
                # Append copies to avoid mutation issues
                self.msg_data['thermal_msg'].append(self.thermal)
                self.msg_data['depth_msg'].append(self.depth)
                self.msg_data['sparse_depth'].append(self.depth_map)
                self.msg_data['odom_msg'].append(self.odom)
                self.msg_data['odom_1sec_msg'].append(self.odom_msgs.flatten())
                self.msg_data['accel_msg'].append(self.accel_msgs.flatten())
                self.msg_data['gyro_msg'].append(self.gyro_msgs.flatten())
                self.msg_data['roll_pitch_yaw'].append(self.roll_pitch_yaw.flatten())
                self.msg_data['velocity_msg'].append(self.velocity_msgs.flatten())
                self.msg_data['just_velocity_msg'].append(self.velocity_msgs[-1])
                self.msg_data['time_stamp'].append(self.cmd_vel.header.stamp.to_sec() if self.cmd_vel else 0.0)
                self.msg_data['cmd_vel_msg'].append(self.cmd_vel_history.flatten())
                self.msg_data['husky_msg'].append(self.husky_msg[:]) # Ensure copy
                self.msg_data['lat_lon_heading_msg'].append([0.0, 0.0, 0.0, 0.0]) # Placeholder
                self.msg_data['joy'].append(self.joy_msg) # Append deep copy

                self.counter += 1
        except Exception as e:
            rospy.logwarn(f"[{self.bag_name_prefix}] Error in cmd_vel_callback: {e}")
            
    def odom_callback(self, msg: Odometry, t):
        try:
            self.odom = msg
            twist = msg.twist.twist; pose = msg.pose.pose
            self.odom_msgs = np.roll(self.odom_msgs, -1, axis=0)
            self.odom_msgs[-1] = [twist.linear.x, twist.linear.y, twist.angular.z]
            self.data_collection.odom_buffer.append({
                'timestamp': t.to_sec(),
                'position': (pose.position.x, pose.position.y),
                'linear_velocity' : twist.linear.x
            })
        except Exception as e:
            rospy.logwarn(f"[{self.bag_name_prefix}] Error in odom_callback: {e}")

    def mag_callback(self, msg: MagneticField, t):
        try: self.imu_processor.mag_update(msg)
        except Exception as e: rospy.logwarn(f"[{self.bag_name_prefix}] Error in mag_callback: {e}")

    def joy_callback(self, msg: Joy, t):
        self.joy_msg = msg 

    def status_callback(self, msg, t):
         try:
              if isinstance(msg, Float32MultiArray) and len(msg.data) >= 4:
                   self.husky_msg = list(msg.data[:4])
         except Exception as e:
              rospy.logwarn(f"[{self.bag_name_prefix}] Error processing /status message: {e}")
              self.husky_msg = [0.0, 0.0, 0.0, 0.0] # Reset to default

    def elevation_map_callback(self, msg: GridMap, t):
        try:
            # Store a copy to prevent modification issues if msg is reused
            self.elevation_data_raw.append({'timestamp': t.to_sec(), 'msg': (msg)})
        except Exception as e:
             rospy.logwarn(f"[{self.bag_name_prefix}] Error in elevation_map_callback: {e}")

    # --- process_depth_maps - Reverted to Original Script1 Logic ---
    def process_depth_maps(self):
        try:
            if self.lidar_points is None or self.thermal is None or self.cam_info_msg is None:
                self.depth = None; self.depth_map = None; return

            thermal_cv = self.bridge.compressed_imgmsg_to_cv2(self.thermal, desired_encoding='bgr8')
            # thermal_tensor = torch.from_numpy(thermal_cv).float()
            # processed_thermal_tensor = image_processing.preprocess_thermal(thermal_tensor)
            # processed_thermal_numpy = (processed_thermal_tensor.numpy() * 255).astype(np.uint8)
            # Undistort
            undistorted_thermal, self.k_matrix_orig, self.d_matrix = image_processing.undistort_image(
                self.cam_info_msg, thermal_cv #processed_thermal_numpy
            )

            rotation_matrix = self.transform_matrix[:3, :3]
            translation_matrix = self.transform_matrix[:3, 3]
            # lidar_points_to_process = self.lidar_points.copy()

            if self.lidar_points.shape[1] > 3:
                min_intensity, max_intensity = 60.0, 6000.0
                intensities = self.lidar_points[:, 3]
                if intensities.ndim > 1: intensities = intensities[:,0]
                intensity_mask = (intensities >= min_intensity) & (intensities <= max_intensity)
                self.lidar_points = self.lidar_points[intensity_mask]
            if self.lidar_points.shape[0] == 0:
                 self.depth = None; self.depth_map = None; return

            points_cam = (rotation_matrix @ self.lidar_points[:, :3].T).T + translation_matrix
            points_cam = points_cam[points_cam[:, 2] > 0.1]
            if points_cam.shape[0] == 0:
                 self.depth = None; self.depth_map = None; return

            max_distance = 30.0
            points_cam[:, 2] = np.clip(points_cam[:, 2], 0.1, max_distance)
            # Lidar projected onto the thermal image plane
            projections = image_processing.project_points(points_cam, self.k_matrix_orig, self.d_matrix)
            # Generate sparse depth map
            self.depth_map = image_processing.generate_depth_map(projections, points_cam, undistorted_thermal.shape[:2])
            # Generate dense depth map
            dense_depth_map = image_processing.create_dense_depth_map(self.depth_map)
            # Saving depth dense map
            if isinstance(dense_depth_map, np.ndarray):
                self.depth = np.nan_to_num(dense_depth_map, nan = 0.0).astype(np.float32)
                
                #  Commented because this removes metric depth
                #  dense_depth_norm = np.nan_to_num(dense_depth_map, nan=0.0).astype(np.uint8)
                #  self.depth = image_processing.enhance_depth_image(dense_depth_norm)
            else:
                 rospy.logwarn(f"[{self.bag_name_prefix}] create_dense_depth_map did not return a numpy array.")
                 self.depth = None; self.depth_map = None; return
            
            if self.original_shape_orig is None:
                self.original_shape_orig = undistorted_thermal.shape[:2]

            self.depth_counter +=1
            if self.depth_counter % 500 == 0:
                # Get min and max values 
                min_depth = np.nanmin(self.depth)
                max_depth = np.nanmax(self.depth)
                cprint(f"[{self.bag_name_prefix}] Processed depth map with min: {min_depth:.2f}, max: {max_depth:.2f}", 'green')  

        except CvBridgeError as e:
             rospy.logerr(f"[{self.bag_name_prefix}] CvBridge error processing depth map: {e}")
             self.depth = None; self.depth_map = None
        except Exception as e:
            rospy.logwarn(f"[{self.bag_name_prefix}] Failed to process depth maps: {e}")
            # traceback.print_exc() # Uncomment for more detail
            self.depth = None; self.depth_map = None
    
    def dlio_path_callback(self, msg, t):
        try:
             self.dlio_final_msg = msg
        except Exception as e:
             rospy.logerr(f"[{self.bag_name_prefix}] Error processing DLIO path message: {e}")
        
    # --- save_data - Combines Script1 saving + RAW Elevation Matching/Saving ---
    def save_data(self, msg_data, pickle_file_name, data_folder, just_the_name):
        final_data = {}
        self.thermal_data_raw.sort(key=lambda x: x['timestamp']) # Ensure thermal is sorted

        # --- Initialize lists ---
        final_data['thermal_paths'] = []
        final_data['thermal_timestamps'] = []
        final_data['depth_paths'] = []
        final_data['sparse_depth_paths'] = []
        # ---> NEW Elevation keys
        final_data['elevation_raw_paths'] = []      # Path to raw .npy file
        final_data['elevation_timestamps'] = []     # Timestamp of the matched elevation map
        final_data['elevation_image_paths'] = []    # Path to 256x256 uint8 image

        # --- Copy aggregated data ---
        final_data['cmd_vel_msg'] = msg_data.get('cmd_vel_msg', []) # Use .get for safety
        data_length = len(final_data['cmd_vel_msg'])
        if data_length == 0:
             cprint(f"[{just_the_name}] No data aggregated in msg_data (based on cmd_vel_msg). Skipping save.", 'red')
             return False # Indicate failure/skip
        cprint(f'[{just_the_name}] Processing {data_length} aggregated data points for saving...', 'cyan')

        # Ensure all required keys exist and slice them to the determined data_length
        required_keys = ['odom_1sec_msg', 'accel_msg', 'gyro_msg', 'time_stamp', 'roll_pitch_yaw']
        for key in required_keys:
            if key not in msg_data:
                cprint(f"[{just_the_name}] Key '{key}' missing in aggregated msg_data. Cannot proceed.", 'red', attrs=['bold'])
                return False # Critical missing data
            final_data[key] = msg_data[key][:data_length]

        # --- Process with DataCollection helpers ---
        final_data['odom'] = self.data_collection.process_odom_vel_data(msg_data) # Needs {'velocity_msg', 'just_velocity_msg'}
        odom_buffer_timestamps = np.array([p.get('timestamp', np.nan) for p in self.data_collection.odom_buffer])
        odom_buffer_pos = [p.get('position', (np.nan, np.nan)) for p in self.data_collection.odom_buffer]
        odom_buffer_vx = np.array([p.get('linear_velocity', np.nan) for p in self.data_collection.odom_buffer])

        final_data['odom_pose'] = []
        if len(odom_buffer_timestamps) > 0 and np.any(~np.isnan(odom_buffer_timestamps)):
             valid_odom_indices = np.where(~np.isnan(odom_buffer_timestamps))[0]
             valid_odom_timestamps = odom_buffer_timestamps[valid_odom_indices]
             for ts in final_data['time_stamp']:
                 time_diffs = np.abs(valid_odom_timestamps - ts)
                 closest_valid_idx_in_subset = np.argmin(time_diffs)
                 # Check if the closest match is within a threshold (e.g., 0.1s)
                 if time_diffs[closest_valid_idx_in_subset] < 0.1:
                     original_index = valid_odom_indices[closest_valid_idx_in_subset]
                     final_data['odom_pose'].append(odom_buffer_pos[original_index])
                 else:
                     final_data['odom_pose'].append((np.nan, np.nan)) # No close match found
        else: # Handle case with no valid odom data
            final_data['odom_pose'] = [(np.nan, np.nan)] * data_length

        final_data['imu_accel_1m'] = []
        final_data['imu_gyro_1m'] = []
        final_data['velocity_at_1m'] = []

        # --- Setup folders ---
        thermal_folder = os.path.join(data_folder, f'thermal_{just_the_name}')
        depth_folder = os.path.join(data_folder, f'depth_{just_the_name}')
        sparse_depth_folder = os.path.join(data_folder, f'sparse_depth_{just_the_name}')

        # ---> NEW Elevation folders
        elevation_raw_npy_folder = os.path.join(data_folder, f'raw_elevation_npy_{just_the_name}')
        elevation_image_folder = os.path.join(data_folder, f'elevation_images_{just_the_name}')
        traversability_masks_folder = os.path.join(data_folder, f'traversability_masks_{just_the_name}')
        traversability_footprints_folder = os.path.join(data_folder, f'traversability_footprints_{just_the_name}')
        os.makedirs(thermal_folder, exist_ok=True); os.makedirs(depth_folder, exist_ok=True)
        os.makedirs(sparse_depth_folder, exist_ok=True);
        os.makedirs(elevation_raw_npy_folder, exist_ok=True)
        os.makedirs(elevation_image_folder, exist_ok=True) 
        os.makedirs(traversability_masks_folder, exist_ok=True)
        os.makedirs(traversability_footprints_folder, exist_ok=True)

        # --- Image Saving & Future State Loop ---
        target_height = 185; target_width = 256; crop_y_start = 40; crop_y_end = 225
        black_image_bgr = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        black_image_gray = np.zeros((target_height, target_width), dtype=np.uint8)
        # Calculating the new K matrix since we are resizing and cropping
        # K_final_cropped = None
        # if self.k_matrix_orig is None or self.original_shape_orig is None:
        #     cprint(f"[{just_the_name}] CRITICAL: Original K matrix or shape was not set. Cannot calculate new K.", "red")
        #     return False
        # else:
        #     # 1. Get original parameters
        #     H_orig, W_orig = self.original_shape_orig
        #     fx_orig = self.k_matrix_orig[0, 0]
        #     fy_orig = self.k_matrix_orig[1, 1]
        #     cx_orig = self.k_matrix_orig[0, 2]
        #     cy_orig = self.k_matrix_orig[1, 2]

        #     # 2. Get your target resize/crop parameters
        #     H_resized, W_resized = 256, 256
        #     # 3. Calculate scaling factors
        #     scale_x = W_resized / W_orig
        #     scale_y = H_resized / H_orig

        #     # 4. Calculate K for the 256x256 *resized* image
        #     fx_scaled = fx_orig * scale_x
        #     fy_scaled = fy_orig * scale_y
        #     cx_scaled = cx_orig * scale_x
        #     cy_scaled = cy_orig * scale_y

        #     # 5. Calculate K for the 185x256 *cropped* image
        #     # (Focal lengths don't change, principal point shifts up)
        #     fx_final = fx_scaled
        #     fy_final = fy_scaled
        #     cx_final = cx_scaled
        #     cy_final = cy_scaled - crop_y_start

        #     K_final_cropped = np.array([
        #         [fx_final, 0,        cx_final],
        #         [0,        fy_final, cy_final],
        #         [0,        0,        1       ]
        #     ])
            
        #     cprint(f"[{just_the_name}] Original K matrix:\n{self.k_matrix_orig}", "blue")
        #     cprint(f"[{just_the_name}] Calculated NEW K matrix for 185x256 crop:\n{K_final_cropped}", "green")
            

        cprint(f"[{just_the_name}] Starting image/future state processing loop ({data_length} items)...", "yellow")
        # Use msg_data lengths for safety in loop
        thermal_msgs_agg = msg_data.get('thermal_msg', [])
        depth_msgs_agg = msg_data.get('depth_msg', [])
        sparse_depth_agg = msg_data.get('sparse_depth', [])

        # Pre-initialize elevation lists with None to match data_length
        final_data['elevation_raw_paths'] = [None] * data_length
        final_data['elevation_timestamps'] = [None] * data_length
        final_data['elevation_image_paths'] = [None] * data_length

        # **** Pre-initialize traversability lists ****
        final_data['traversability_mask_paths'] = [None] * data_length
        final_data['traversability_footprint_paths'] = [None] * data_length

        # Pre-extract valid odom timestamps and values for faster lookup inside the loop
        valid_odom_timestamps_for_future = []
        valid_odom_vx_for_future = []
        if len(odom_buffer_timestamps) > 0:
            valid_mask = ~np.isnan(odom_buffer_timestamps) & ~np.isnan(odom_buffer_vx)
            valid_odom_timestamps_for_future = odom_buffer_timestamps[valid_mask]
            valid_odom_vx_for_future = odom_buffer_vx[valid_mask]

        for index in range(data_length):
            timestamp = final_data['time_stamp'][index]
            # Future IMU/Velocity Calc
            start_timestamp_B = self.data_collection.get_imu_data_after_distance(timestamp, 1.9) # Find time after ~1m travel
            future_vx = 0.0 # Default future velocity
            if start_timestamp_B is not None:
                imu_data_B_window = self.data_collection.get_imu_data_over_time_window(start_timestamp_B)
                accel_array_B, gyro_array_B = self.data_collection.process_imu_data_window(imu_data_B_window)
                # Find closest valid odom velocity near start_timestamp_B
                if len(valid_odom_timestamps_for_future) > 0:
                    time_diffs_future = np.abs(valid_odom_timestamps_for_future - start_timestamp_B)
                    closest_future_odom_idx = np.argmin(time_diffs_future)
                    # Use a threshold (e.g., 0.1s) to ensure the match is relevant
                    if time_diffs_future[closest_future_odom_idx] < 0.1:
                        future_vx = valid_odom_vx_for_future[closest_future_odom_idx]
            else: # If no future timestamp found (e.g., end of bag)
                accel_array_B = np.zeros((400, 3), dtype=np.float32)
                gyro_array_B = np.zeros((400, 3), dtype=np.float32)

            final_data['imu_accel_1m'].append(accel_array_B.flatten())
            final_data['imu_gyro_1m'].append(gyro_array_B.flatten())
            final_data['velocity_at_1m'].append(future_vx)

            # --- Thermal Saving ---
            thermal_path = os.path.join(thermal_folder, f'{index}.png'); final_data['thermal_paths'].append(thermal_path)
            current_thermal_ts_saved = 0.0 # Initialize timestamp saved for this index
            if index < len(thermal_msgs_agg) and thermal_msgs_agg[index] is not None:
                try:
                    thermal_msg = thermal_msgs_agg[index]
                    # Check if header and stamp exist before accessing
                    if thermal_msg.header and thermal_msg.header.stamp:
                        current_thermal_ts_saved = thermal_msg.header.stamp.to_sec()
                    else:
                        rospy.logwarn(f"[{just_the_name}] Thermal message at index {index} missing header/stamp.")

                    thermal_img = self.bridge.compressed_imgmsg_to_cv2(thermal_msg, desired_encoding='bgr8')
                    cv2.imwrite(thermal_path, thermal_img) # Save original size
                except AttributeError as ae: # Catch cases where thermal_msg might not be a full message object
                    rospy.logwarn(f"[{just_the_name}] Attribute error processing thermal {index}: {ae}. Saving black image.")
                    cv2.imwrite(thermal_path, black_image_bgr) # Use the BGR black image
                    current_thermal_ts_saved = 0.0 # Reset timestamp if save failed
                except Exception as e:
                    rospy.logwarn(f"[{just_the_name}] Error saving thermal {index}: {e}. Saving black image.")
                    # traceback.print_exc() # Uncomment for details
                    cv2.imwrite(thermal_path, black_image_bgr) # Use the BGR black image
                    current_thermal_ts_saved = 0.0 # Reset timestamp if save failed
            else:
                # Save black image if no thermal message was aggregated for this index
                cv2.imwrite(thermal_path, black_image_bgr) # Use the BGR black image
            final_data['thermal_timestamps'].append(current_thermal_ts_saved)


            # --- Depth Saving ---
            depth_path = os.path.join(depth_folder, f'{index}.npy'); final_data['depth_paths'].append(depth_path)
            sparse_path = os.path.join(sparse_depth_folder, f'{index}_sparse.npy'); final_data['sparse_depth_paths'].append(sparse_path)
            # Check index bounds and non-None values
            if index < len(depth_msgs_agg) and depth_msgs_agg[index] is not None and \
               index < len(sparse_depth_agg) and sparse_depth_agg[index] is not None:
                try:
                    sparse_depth = sparse_depth_agg[index]; depth_img = depth_msgs_agg[index]
                    # Ensure inputs are numpy arrays before processing
                    if not isinstance(sparse_depth, np.ndarray) or not isinstance(depth_img, np.ndarray):
                        raise TypeError(f"Expected numpy arrays for depth/sparse, got {type(sparse_depth)} and {type(depth_img)}")

                    # Apply crop/resize as in original Script1
                    # Resize first, then crop
                    # sparse_depth_resized = cv2.resize(sparse_depth, (target_width, 256), interpolation=cv2.INTER_NEAREST)
                    # sparse_depth_cropped = sparse_depth_resized[crop_y_start:crop_y_end, :]
                    np.save(sparse_path, sparse_depth.astype(np.float16))

                    # depth_img_resized = cv2.resize(depth_img, (target_width, 256), interpolation=cv2.INTER_NEAREST)
                    # depth_cropped = depth_img_resized[crop_y_start:crop_y_end, :]
                    # cv2.imwrite(depth_path, depth_cropped)
                    np.save(depth_path, depth_img.astype(np.float16))
                               
                except Exception as e:
                    rospy.logwarn(f"[{just_the_name}] Error saving depth/sparse {index}: {e}")
                    
            else:
                # Save black/zero placeholders if data is missing
                 print("Data missing! ")
        print(depth_img.shape, "DEPTH SHAPE") 
        cprint(f"[{just_the_name}] Finished image/future state processing loop.", "yellow")

        traversability_success_count = 0
        all_poses_for_traversability = [] # Default empty list

        # Extract poses ONCE if the DLIO message exists
        if self.dlio_final_msg is not None:
            all_poses_for_traversability = extract_poses_from_path(self.dlio_final_msg)
            if not all_poses_for_traversability:
                cprint(f"[{just_the_name}] Warning: DLIO message found, but no poses extracted.", "magenta")
            

        # Check if we have poses to work with
        if not all_poses_for_traversability:
            cprint(f"[{just_the_name}] No valid poses extracted from DLIO path or DLIO message missing. Skipping all traversability generation.", "yellow")
        else:
            # Get the lists populated in the first loop
            thermal_timestamps_list = final_data.get('thermal_timestamps', [])
            thermal_paths_list = final_data.get('thermal_paths', [])
            current_data_length = len(thermal_timestamps_list) # Get length *after* first loop

            # Ensure lists are consistent before iterating
            if len(thermal_paths_list) == current_data_length:
                cprint(f"[{just_the_name}] Processing traversability for {current_data_length} entries...", "cyan")
                for idx in range(current_data_length): # Iterate through the collected data
                    current_thermal_ts = thermal_timestamps_list[idx]
                    current_thermal_path = thermal_paths_list[idx]

                    mask_path = None
                    footprint_path = None

                    # Check if the thermal data for this index is valid before attempting processing
                    if current_thermal_ts is not None and current_thermal_ts > 0 and \
                    current_thermal_path is not None and os.path.exists(current_thermal_path):
                        try:
                            # Call the helper function for this specific index/timestamp
                            mask_path, footprint_path = process_traversability_single(
                                thermal_ts=current_thermal_ts,
                                thermal_image_path=current_thermal_path,
                                #dlio_path_msg=self.dlio_final_msg,           # Pass the single final DLIO message
                                all_past_thermal_ts=thermal_timestamps_list, # Pass the *complete* list
                                all_poses_from_path=all_poses_for_traversability, # Pass the *complete* pose list
                                output_footprints_dir=traversability_footprints_folder,
                                output_masks_dir=traversability_masks_folder,
                                index=idx,                                   # Pass the current index
                                logger=None
                            )
                            if mask_path is not None: # Increment count if successful
                                traversability_success_count += 1

                        except Exception as e:
                            rospy.logerr(f"[{just_the_name}] Error in traversability processing for index {idx}: {e}")
                            # Ensure paths are None on error
                            mask_path = None
                            footprint_path = None
                    # else: Thermal ts or path was invalid, mask/footprint remain None

                    if 'traversability_mask_paths' in final_data and idx < len(final_data['traversability_mask_paths']):
                        final_data['traversability_mask_paths'][idx] = mask_path
                    if 'traversability_footprint_paths' in final_data and idx < len(final_data['traversability_footprint_paths']):
                        final_data['traversability_footprint_paths'][idx] = footprint_path

            else:
                cprint(f"[{just_the_name}] Mismatch between thermal timestamps ({len(thermal_timestamps_list)}) and paths ({len(thermal_paths_list)}). Skipping traversability.", "red")

        cprint(f"[{just_the_name}] Finished traversability processing. Generated {traversability_success_count} sets.", 'cyan')

        elevation_match_count = 0
        thermal_timestamps_to_match = final_data['thermal_timestamps'] # Already populated
        # Find indices where thermal timestamp is valid (not None and > 0)
        valid_thermal_indices = [i for i, ts in enumerate(thermal_timestamps_to_match) if ts is not None and ts > 0.0]

        if not valid_thermal_indices:
            # Silently skip if no valid thermal timestamps to match
            pass
        else:
            # Loop only through indices with valid thermal timestamps
            for index in valid_thermal_indices:
                current_thermal_ts = thermal_timestamps_to_match[index]

                # --- Use cached BEV from thermal_data_raw (captured at processing time) ---
                try:
                    # Find the matching thermal_data_raw entry for this index
                    if index < len(self.thermal_data_raw):
                        bev_elev = self.thermal_data_raw[index].get('elevation_bev', None)
                    else:
                        bev_elev = None
                    
                    if bev_elev is not None and bev_elev.shape[0] > 0 and bev_elev.shape[1] > 0:
                        elevation_match_count += 1
                        matched_elev_ts = current_thermal_ts  # Use thermal timestamp as the matched timestamp
                        
                        # The BEV is already robot-relative from get_synced_elevation_bev
                        rotated_grid = bev_elev.astype(np.float16)

                        # --- 1. Save Robot-Relative Raw Elevation Data (.npy) ---
                        elevation_raw_npy_path = os.path.join(elevation_raw_npy_folder, f'{index}_raw.npy')
                        np.save(elevation_raw_npy_path, rotated_grid)
                        # Store results for this index
                        final_data['elevation_raw_paths'][index] = elevation_raw_npy_path
                        final_data['elevation_timestamps'][index] = matched_elev_ts

                        # --- 2. Create and Save 256x256 uint8 Image ---
                        elevation_image_path = os.path.join(elevation_image_folder, f'{index}_image.png')
                        # Use nested try-except for image generation/saving robustness
                        try:
                            # Save as-is without flip (matches reference behaviour)
                            img_data_to_scale = rotated_grid.copy()

                            valid_mask_img = ~np.isnan(img_data_to_scale)
                            # Initialize float16 image (e.g., black)
                            scaled_image_fl16 = np.zeros(img_data_to_scale.shape, dtype=np.float16)

                            if np.any(valid_mask_img): # Process only if there's valid data
                                min_val = np.nanmin(img_data_to_scale)
                                max_val = np.nanmax(img_data_to_scale)

                                if max_val > min_val: # Avoid division by zero
                                    # Normalize valid data to 0-255
                                    scaled_data = ((img_data_to_scale[valid_mask_img] - min_val) / (max_val - min_val)) * 255.0
                                    scaled_image_fl16[valid_mask_img] = scaled_data.astype(np.float16)
                                elif not np.isnan(min_val): # Handle flat ground case (single valid value)
                                    scaled_image_fl16[valid_mask_img] = 128 # Assign mid-gray

                            # Resize the scaled float16 image to 256x256
                            resized_scaled_image = cv2.resize(scaled_image_fl16, (256, 256), interpolation=cv2.INTER_NEAREST)

                            # Convert to uint8 before saving
                            resized_scaled_image_uint8 = resized_scaled_image.astype(np.uint8)

                            # Save the 256x256 image
                            save_success_img = cv2.imwrite(elevation_image_path, resized_scaled_image_uint8)
                            if save_success_img:
                                final_data['elevation_image_paths'][index] = elevation_image_path # Store path only on success
                            # else: Image save failed, path remains None

                        except Exception:
                            pass

                except Exception as e:
                    rospy.logerr(f"[{self.bag_name_prefix}] Elevation processing failed for index {index}: {e}")
                    traceback.print_exc()
                    final_data['elevation_raw_paths'][index] = None
                    final_data['elevation_timestamps'][index] = None
                    final_data['elevation_image_paths'][index] = None
                    pass

                # else: No suitable elevation match found, paths remain None

        cprint(f"[{just_the_name}] Finished elevation processing. Matched {elevation_match_count}/{len(valid_thermal_indices)} valid thermal timestamps.", 'cyan')
        # --- Roughness Calc & Save ---
        # Process CURRENT IMU data accumulated in msg_data
        final_data['processed_gyro'] = self.data_collection.process_gyro_msg(msg_data)
        final_data['processed_accel'] = self.data_collection.process_accl_msg(msg_data)

        # Calculate CURRENT roughness score
        if final_data['processed_accel'].ndim == 2 and final_data['processed_accel'].shape[0] == data_length and \
        final_data['processed_gyro'].ndim == 2 and final_data['processed_gyro'].shape[0] == data_length:
            ajz = final_data['processed_accel'][:, 5]; gax = final_data['processed_gyro'][:, 3]; gay = final_data['processed_gyro'][:, 4]
            roughness_score_raw = (self.WEIGHT_ACCEL_Z_JERK * np.abs(ajz) + self.WEIGHT_GYRO_X_ACCEL * np.abs(gax) + self.WEIGHT_GYRO_Y_ACCEL * np.abs(gay))
            window_size = 100
            if window_size > 1 :
                roughness_score_filtered = np.convolve(roughness_score_raw, np.ones(window_size)/window_size, mode='same')
            else:
                roughness_score_filtered = roughness_score_raw

            # ---> Checkpoint 1: Are ajz, gax, gay non-zero? Is roughness_score_raw non-zero? Is roughness_score_filtered non-zero?
            print(f"[{just_the_name}] Current Raw Means: ajz={np.mean(np.abs(ajz)):.4f}, gax={np.mean(np.abs(gax)):.4f}, gay={np.mean(np.abs(gay)):.4f}")
            print(f"[{just_the_name}] Current Roughness Means: raw={np.mean(roughness_score_raw):.4f}, filtered={np.mean(roughness_score_filtered):.4f}")

            # Apply clipping and normalization
            final_data['roughness_score'] = (np.clip(roughness_score_filtered, 0.01, 0.15) - 0.01) / (0.15 - 0.01)
            # ---> Checkpoint 2: What is the mean value AFTER normalization?
            print(f"[{just_the_name}] Current Final Mean: {np.mean(final_data['roughness_score']):.4f}")
        else:
            # ---> Checkpoint 3: Does this message print? If so, the input arrays are invalid.
            print(f"[{just_the_name}] Could not calculate CURRENT roughness score due to shape mismatch/invalid input.")
            print(f"  Processed Accel Shape: {final_data['processed_accel'].shape if 'processed_accel' in final_data else 'N/A'}")
            print(f"  Processed Gyro Shape: {final_data['processed_gyro'].shape if 'processed_gyro' in final_data else 'N/A'}")
            final_data['roughness_score'] = np.zeros(data_length, dtype=np.float32) # Assign zeros if calculation failed

        # Recreate dict structure expected by processing functions
        msg_data_1m = {'accel_msg': final_data['imu_accel_1m'], 'gyro_msg': final_data['imu_gyro_1m']}
        final_data['processed_accel_1m'] = self.data_collection.process_accl_msg(msg_data_1m)
        final_data['processed_gyro_1m'] = self.data_collection.process_gyro_msg(msg_data_1m)

        # Calculate roughness score only if processed future IMU data is valid
        if final_data['processed_accel_1m'].ndim == 2 and final_data['processed_accel_1m'].shape[0] == data_length and \
           final_data['processed_gyro_1m'].ndim == 2 and final_data['processed_gyro_1m'].shape[0] == data_length:
             ajz = final_data['processed_accel_1m'][:, 5]; gax = final_data['processed_gyro_1m'][:, 3]; gay = final_data['processed_gyro_1m'][:, 4]
             roughness_score_raw = (self.WEIGHT_ACCEL_Z_JERK * np.abs(ajz) + self.WEIGHT_GYRO_X_ACCEL * np.abs(gax) + self.WEIGHT_GYRO_Y_ACCEL * np.abs(gay))
             window_size = 100 # Prevent window larger than data
             if window_size > 1 :
                 roughness_score_filtered = np.convolve(roughness_score_raw, np.ones(window_size)/window_size, mode='same')
             else: # Avoid convolution with window size 1
                 roughness_score_filtered = roughness_score_raw
             # Clip and normalize roughness score
             final_data['roughness_score_1m'] = (np.clip(roughness_score_filtered, 0.01, 0.15) - 0.01) / (0.15 - 0.01)
        else:
            rospy.logwarn(f"[{just_the_name}] Could not calculate roughness score due to invalid shape of processed future IMU data. Accel shape: {final_data['processed_accel_1m'].shape}, Gyro shape: {final_data['processed_gyro_1m'].shape}. Expected ({data_length}, ...)")
            # final_data['roughness_score_1m'] = np.zeros(data_length, dtype=np.float32) # Assign zeros if calculation failed

        cprint(f'[{just_the_name}] Final data length check before assertion: {data_length}', 'green')
        
        original_data_length = data_length # Store the length *before* filtering
        indices_to_keep = [
            i for i in range(original_data_length) # Iterate up to original length
            if final_data['elevation_raw_paths'][i] is not None
        ]

        new_data_length = len(indices_to_keep)

        if new_data_length < original_data_length:
            cprint(f"[{just_the_name}] Found {original_data_length - new_data_length} entries missing elevation data. Filtering...", "magenta")
            filtered_final_data = {}
            for key, value in final_data.items(): # Iterate through items (key, value pairs)
                # *** Check if the value is a list/array AND has the original length ***
                if isinstance(value, (list, np.ndarray)) and len(value) == original_data_length:
                    # Apply filtering based on type
                    if isinstance(value, list):
                        # Use list comprehension for lists
                        filtered_final_data[key] = [value[i] for i in indices_to_keep]
                    elif isinstance(value, np.ndarray):
                        # Use advanced numpy indexing for arrays (more efficient)
                        filtered_final_data[key] = value[indices_to_keep]
                else:
                    # *** If not a list/array of the original length, copy it directly ***
                    # This handles metadata or lists potentially modified by DataCollection
                    filtered_final_data[key] = value

            # Replace the original dictionary with the filtered one
            final_data = filtered_final_data
            # Update data_length for subsequent assertions and logging
            data_length = new_data_length
            cprint(f"[{just_the_name}] Filtering complete. New data length: {data_length}", "green")
        else:
            cprint(f"[{just_the_name}] No elevation filtering needed. All {data_length} entries have elevation data.", "green")

        # --- Final Assertions ---
        final_keys_to_check = [
            'cmd_vel_msg', 'accel_msg', 'gyro_msg', 'time_stamp', 'roll_pitch_yaw',
            'imu_accel_1m', 'imu_gyro_1m', 'velocity_at_1m', 'roughness_score', 'roughness_score_1m',
            'thermal_paths', 'thermal_timestamps', 'depth_paths', 'sparse_depth_paths',
            'elevation_raw_paths', 'elevation_timestamps', 'elevation_image_paths',
            'traversability_mask_paths', 'traversability_footprint_paths'
        ]
        try:
            for key in final_keys_to_check:
                 assert key in final_data, f"Key '{key}' is missing in final_data dictionary."
                 assert len(final_data[key]) == data_length, f"Length mismatch for key '{key}'. Expected {data_length}, got {len(final_data[key])}."
            cprint(f'[{just_the_name}] All final data structure assertions passed.', 'green')
        except AssertionError as e:
             cprint(f"[{just_the_name}] SAVE FAILED: FINAL DATA STRUCTURE ASSERTION ERROR! {e}", 'red', attrs=['bold'])
             # Optionally print lengths for debugging:
             for k in final_keys_to_check:
                  if k in final_data: cprint(f"Length of '{k}': {len(final_data[k])}", 'red')
                  else: cprint(f"Key '{k}' missing", 'red')
             return False # Indicate save failure

        # --- Save to Pickle ---
        pickle_path = os.path.join(data_folder, pickle_file_name)
        cprint(f'[{just_the_name}] Saving final data ({data_length} entries) to {pickle_path}...', 'yellow')
        final_keys_sorted = sorted(list(final_data.keys())); cprint(f"[{just_the_name}] Final keys in data: {final_keys_sorted}", 'magenta')
        try:
            with open(pickle_path, 'wb') as f: pickle.dump(final_data, f)
            cprint(f'[{just_the_name}] Saved data successfully.', 'green', attrs=['blink'])
        except Exception as e:
             cprint(f"[{just_the_name}] Error saving pickle file: {e}", 'red', attrs=['bold'])
             # traceback.print_exc() # Uncomment for details
             return False # Indicate save failure
        return True # Indicate save success


def threading_function(bag_path, output_folder, just_the_name, time_threshold):
    cprint(f"Thread {threading.current_thread().name} starting for: {bag_path}", "blue")
    try:
        recorder = ListenRecordData(bag_path=bag_path,
                                    bag_name_prefix=just_the_name, time_threshold=time_threshold, logger = None)
        
        if not recorder.msg_data or not recorder.msg_data.get('cmd_vel_msg'):
             cprint("No aggregated data found. Skipping save.", "red")
             return 
        msg_data_copy = copy.deepcopy(recorder.msg_data)

        success = recorder.save_data(msg_data=msg_data_copy, pickle_file_name=f"{just_the_name}.pkl",
                                     data_folder=output_folder, just_the_name=just_the_name)
        if success:
             cprint(f"Thread {threading.current_thread().name} finished successfully for: {just_the_name}", "green")
        else:
             cprint(f"Thread {threading.current_thread().name} finished with save errors for: {just_the_name}", "red")

    except Exception as e:
        cprint(f"!!! Critical error in thread {threading.current_thread().name} for {bag_path}: {e}", "red", attrs=['bold'])
        traceback.print_exc() # Print detailed traceback for thread errors

if __name__ == '__main__':
    # Initialize ROS node only if not already initialized
    if not rospy.core.is_initialized():
        try:
            rospy.init_node('offline_bag_processor', anonymous=True, log_level=rospy.INFO)
            cprint("ROS node 'offline_bag_processor' initialized.", 'green')
        except rospy.exceptions.ROSInitException as e:
            cprint(f"Failed to initialize ROS node: {e}. Proceeding without ROS.", 'yellow')
    else:
         cprint("ROS node already initialized.", 'blue')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Process ROS bag files sequentially.')
    parser.add_argument('-f', '--file', type=str, required=True, help='Base output folder path.')
    parser.add_argument('-b', '--folder', type=str, required=True, help='Input folder containing .bag files.')
    parser.add_argument('-t', '--time_threshold', type=float, default=0.1, help='Max time difference (seconds) for synchronization.')
    args = parser.parse_args()

    save_data_path = args.file
    bag_folder = args.folder
    sync_threshold = args.time_threshold

    cprint(f"Input ROS bag folder: {bag_folder}", 'blue')
    cprint(f"Output data folder: {save_data_path}", 'blue')
    cprint(f"Using time threshold for sync: {sync_threshold}s", 'blue')

    if not os.path.isdir(bag_folder):
        cprint(f"Error: Bag folder not found: '{bag_folder}'", 'red', attrs=['bold'])
        exit(1)

    try:
        list_of_bags = sorted([f for f in os.listdir(bag_folder) if f.endswith('.bag')])
    except OSError as e:
         cprint(f"Error accessing bag folder '{bag_folder}': {e}", 'red', attrs=['bold'])
         exit(1)

    if not list_of_bags:
        cprint(f"No .bag files found in '{bag_folder}'", 'red')
        exit(0)

    cprint(f"Found {len(list_of_bags)} bag files to process sequentially.", 'blue')

    try:
        os.makedirs(save_data_path, exist_ok=True)
    except OSError as e:
         cprint(f"Error creating output directory '{save_data_path}': {e}", 'red', attrs=['bold'])
         exit(1)

    # --- Sequential Processing Loop ---
    for i, bag_filename in enumerate(list_of_bags):
        just_the_name = os.path.splitext(bag_filename)[0]
        full_bag_path = os.path.join(bag_folder, bag_filename)

        cprint(f"\nProcessing bag {i+1}/{len(list_of_bags)}: {bag_filename}", 'yellow')
        try:
            # Directly call the processing function for the current bag
            threading_function(full_bag_path, save_data_path, just_the_name, sync_threshold)
            cprint(f"--- Finished processing: {bag_filename}", 'green')
        except Exception as e:
            cprint(f"!!! ERROR processing {bag_filename}: {e}", 'red', attrs=['bold'])
            traceback.print_exc()
            cprint(f"--- Skipping to next bag due to error ---\n", 'red')
            continue # Move to the next bag file

    cprint('\nAll bag processing completed!', 'green', attrs=['bold'])
    exit(0)