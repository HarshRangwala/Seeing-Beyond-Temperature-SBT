#!/usr/bin/env python3
import rospy
import tf2_ros
import json
import os
import copy
import pickle
import numpy as np
import cv2
import rosbag
import threading
import argparse
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from termcolor import cprint
import Helpers.image_processing as image_processing
from Helpers.sensor_fusion import imu_processor
import Helpers.data_calculation as data_calculation
from collections import deque

class ListenRecordData:
    def __init__(self, bag_path, transform_file, output_folder):
        self.output_folder = output_folder

        # Initialize data structures with deep copies
        self.msg_data = {
            'image_left_msg': deque(),
            'image_right_msg': deque(),
            'thermal_msg': deque(),
            'lidar_msg': deque(),
            'depth_map': deque(),
            'dense_depth_map': deque(),
            'overlay_image': deque(),
            'odom_msg': deque(),
            'odom_1sec_msg': deque(),
            'velocity_msg': deque(),
            'just_velocity_msg': deque(),
            'cmd_vel_msg': deque(),
            'time_stamp': deque(),
            'accel_msg': deque(),
            'roll_pitch_yaw': deque(),
            'gyro_msg': deque(),
            'hunter_msg': deque(),
            'lat_lon_heading_msg': deque(),
            'imu_dash_stats': deque(),
            'joy': deque()
        }

        # Queues for synchronized processing
        self.image_left_queue = deque()
        self.image_right_queue = deque()
        self.thermal_queue = deque()
        self.lidar_queue = deque()

        # Other initializations
        self.odom_msgs = np.zeros((200, 3), dtype=np.float32)
        self.gyro_msgs = np.zeros((400, 3), dtype=np.float32)
        self.accel_msgs = np.zeros((400, 3), dtype=np.float32)
        self.velocity_msgs = np.zeros((5, 2), dtype=np.float32)
        self.cmd_vel_history = np.zeros((10,2), dtype=np.float32)
        self.roll_pitch_yaw = np.zeros((400, 3), dtype=np.float32)
        self.cmd_vel = None
        self.image_left = None
        self.image_right = None
        self.odom = None
        self.thermal_image = None
        self.lidar_points = None
        self.counter = 0
        self.imu_processor = imu_processor()
        self.bridge = CvBridge()
        self.imu_counter = 0
        self.hunter_msg = [0.0, 0.0, 0.0, 0.0]
        self.previous_nano = 0
        self.joy_msg = None
        self.cam_info_msg = None

        # Create necessary output directories
        self.create_output_directories()

        # Process the bag
        self.process_bag(bag_path)

    def load_transform(self, points, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        translation = np.array([
            data['translation']['x'],
            data['translation']['y'],
            data['translation']['z']
        ])
        rotation = self.quaternion_to_rotation_matrix(data['rotation'])
        points_transformed = (rotation @ points.T).T + translation
        rospy.loginfo(f"Loaded transformation matrix from {filepath}")
        return points_transformed

    def quaternion_to_rotation_matrix(self, q):
        """Converts a geometry_msgs/Quaternion to a rotation matrix."""
        x, y, z, w = q['x'], q['y'], q['z'], q['w']
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        x /= norm
        y /= norm
        z /= norm
        w /= norm
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])
        return R

    def create_output_directories(self):
        subdirs = [
            'image_left',
            'image_right',
            'thermal',
            'depth_map',
            'dense_depth_map',
            'overlay_image'
        ]
        for subdir in subdirs:
            path = os.path.join(self.output_folder, subdir)
            os.makedirs(path, exist_ok=True)
            cprint(f"Created directory: {path}", 'green')

    def process_bag(self, bag_path):
        bag = rosbag.Bag(bag_path)
        all_topics = [
            '/sensor_suite/lwir/lwir/image_raw/compressed',
            '/sensor_suite/ouster/points',
            '/sensor_suite/lwir/lwir/camera_info',
            '/sensor_suite/left_camera_optical/image_color/compressed',
            '/sensor_suite/right_camera_optical/image_color/compressed',
            '/sensor_suite/f9p_rover/navpvt',
            '/husky_velocity_controller/cmd_vel_out',
            '/sensor_suite/witmotion_imu/imu',
            '/sensor_suite/witmotion_imu/magnetometer',
            '/status',  # hunter_status
            '/joy_teleop/joy'
        ]
        rospy.loginfo(f"Starting processing of bag: {bag_path}")
        for topic, msg, t in bag.read_messages(topics=all_topics):
            if topic == '/husky_velocity_controller/cmd_vel_out':
                self.cmd_vel_callback(msg)
            elif topic == '/odometry/filtered':
                self.odom_callback(msg)
            elif topic == '/sensor_suite/witmotion_imu/imu':
                self.imu_callback(msg)
            elif topic == '/sensor_suite/left_camera_optical/image_color/compressed':
                self.image_left_callback(msg)
            elif topic == '/sensor_suite/right_camera_optical/image_color/compressed':
                self.image_right_callback(msg)
            elif topic == '/sensor_suite/witmotion_imu/magnetometer':
                self.mag_callback(msg)
            elif topic == '/sensor_suite/f9p_rover/navpvt':
                self.callback(msg)
            elif topic == '/joy_teleop/joy':
                self.joy_callback(msg)
            elif topic == '/sensor_suite/lwir/lwir/image_raw/compressed':
                self.thermal_image_callback(msg)
            elif topic == '/sensor_suite/ouster/points':
                self.lidar_callback(msg)
            elif topic == '/sensor_suite/lwir/lwir/camera_info':
                self.calib_callback(msg)
        bag.close()
        rospy.loginfo(f"Finished processing bag: {bag_path}")
        self.save_data()

    def calib_callback(self, msg):
        self.cam_info_msg = msg
        rospy.loginfo("Camera info updated.")

    def cmd_vel_callback(self, msg):
        if not hasattr(self, 'cmd_vel_history'):
            self.cmd_vel_history = np.zeros((10, 2), dtype=np.float32)
        self.cmd_vel_history = np.roll(self.cmd_vel_history, -1, axis=0)
        self.cmd_vel_history[-1] = np.array([msg.twist.linear.x, msg.twist.angular.z])
        self.cmd_vel = msg

    def image_left_callback(self, msg):
        self.image_left_queue.append((msg.header.stamp, msg))

    def image_right_callback(self, msg):
        self.image_right_queue.append((msg.header.stamp, msg))

    def thermal_image_callback(self, msg):
        try:
            # Decompress the thermal image
            thermal_cv = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rospy.loginfo(f"Received thermal image with shape: {thermal_cv.shape}")
            # Deep copy to prevent reference issues
            self.thermal_queue.append((msg.header.stamp, thermal_cv.copy()))
        except CvBridgeError as e:
            rospy.logwarn(f"Thermal image decompression failed: {e}")
            self.thermal_queue.append((msg.header.stamp, None))

    def lidar_callback(self, msg):
        lidar_points = []
        try:
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                lidar_points.append(point)
            self.lidar_points = np.array(lidar_points)
            rospy.loginfo(f"Received {len(self.lidar_points)} LIDAR points.")
            self.lidar_queue.append((msg.header.stamp, self.lidar_points.copy()))
        except Exception as e:
            rospy.logwarn(f"LIDAR point extraction failed: {e}")
            self.lidar_queue.append((msg.header.stamp, None))
        # Process depth maps after receiving LIDAR points
        self.process_depth_maps()

    def hunter_callback(self, msg):
        # Assuming msg has attributes: linear_velocity, steering_angle, motor_states
        try:
            self.hunter_msg = [
                msg.linear_velocity,
                msg.steering_angle,
                msg.motor_states[1]['rpm'],
                msg.motor_states[2]['rpm']
            ]
        except Exception as e:
            rospy.logwarn(f"Hunter callback parsing failed: {e}")
            self.hunter_msg = [0.0, 0.0, 0.0, 0.0]

    def callback(self, msg):
        new_nano = msg.nano
        if abs(new_nano - self.previous_nano) > 100 and self.hunter_msg[0] >= 0.0:
            self.previous_nano = new_nano
            rospy.loginfo(f"Received velocity message at counter {self.counter}")
            if not hasattr(self, 'velocity_msgs'):
                self.velocity_msgs = np.zeros((5, 2), dtype=np.float32)
            self.velocity_msgs = np.roll(self.velocity_msgs, -1, axis=0)
            self.velocity_msgs[-1] = np.array([msg.velN * 1e-3, msg.velE * 1e-3])

            cmd_vel = self.cmd_vel_history
            odom_msg = self.odom_msgs

            # Synchronize messages based on closest timestamps
            synchronized = self.synchronize_messages(msg.header.stamp)
            if synchronized:
                image_left_msg, image_right_msg, thermal_msg, lidar_msg = synchronized
                if image_left_msg and image_right_msg and thermal_msg and lidar_msg:
                    self.save_synchronized_data(image_left_msg, image_right_msg, thermal_msg, lidar_msg)
                    self.counter += 1

    def synchronize_messages(self, lidar_stamp):
        """Synchronize image and LIDAR messages based on timestamps."""
        if not self.image_left_queue or not self.image_right_queue or not self.thermal_queue or not self.lidar_queue:
            return None

        # Find the closest image messages to the LIDAR timestamp
        image_left_msg = self.get_closest_message(self.image_left_queue, lidar_stamp)
        image_right_msg = self.get_closest_message(self.image_right_queue, lidar_stamp)
        thermal_msg = self.get_closest_message(self.thermal_queue, lidar_stamp)
        lidar_msg = self.get_closest_message(self.lidar_queue, lidar_stamp)

        return image_left_msg, image_right_msg, thermal_msg, lidar_msg

    def get_closest_message(self, queue, target_stamp):
        """Retrieve and remove the message closest in time to the target timestamp."""
        if not queue:
            return None
        closest = min(queue, key=lambda x: abs(x[0].to_sec() - target_stamp.to_sec()))
        queue.remove(closest)
        return closest[1]

    def save_synchronized_data(self, image_left_msg, image_right_msg, thermal_msg, lidar_msg):
        """Process and save synchronized data."""
        try:
            # Undistort images
            image_left_cv = self.bridge.compressed_imgmsg_to_cv2(image_left_msg, desired_encoding='bgr8')
            undistorted_left, K_left, dist_coeffs_left = image_processing.undistort_image(self.cam_info_msg, image_left_cv.copy())
            
            image_right_cv = self.bridge.compressed_imgmsg_to_cv2(image_right_msg, desired_encoding='bgr8')
            undistorted_right, K_right, dist_coeffs_right = image_processing.undistort_image(self.cam_info_msg, image_right_cv.copy())
            
            undistorted_thermal, K_thermal, dist_coeffs_thermal = image_processing.undistort_image(self.cam_info_msg, thermal_msg.copy())

            # Transform LIDAR points to camera frame
            point_cam = self.load_transform(lidar_msg, 'transform.json')
            rospy.loginfo("Transformed LIDAR points to camera frame.")

            # Project points onto image plane
            projections = image_processing.project_points(point_cam, K_thermal, dist_coeffs_thermal)
            rospy.loginfo(f"Projected {len(projections)} LIDAR points onto image plane.")

            # Generate depth maps
            depth_map = image_processing.generate_depth_map(projections, point_cam, undistorted_thermal.shape)
            rospy.loginfo("Generated sparse depth map.")

            depth_colored = image_processing.convert_depth_to_color(depth_map)
            dense_depth_map = image_processing.create_dense_depth_map(depth_map)
            rospy.loginfo("Generated dense depth map.")

            # Overlay points on thermal image
            overlay_image = image_processing.overlay_points(undistorted_thermal.copy(), projections, color=(0, 255, 0))
            rospy.loginfo("Created overlay image.")

            # Generate unique filename using timestamp
            timestamp = int(self.lidar_stamp.to_sec() * 1e6)

            # Save images
            self.save_image('depth_map', f"{timestamp}.png", image_processing.convert_depth_to_color(depth_map))
            self.save_image('dense_depth_map', f"{timestamp}.npy", dense_depth_map)
            self.save_image('overlay_image', f"{timestamp}.png", overlay_image)
            self.save_image('thermal', f"{timestamp}.png", undistorted_thermal)
            self.save_image('image_left', f"{timestamp}.png", undistorted_left)
            self.save_image('image_right', f"{timestamp}.png", undistorted_right)

            # Append to msg_data with deep copies
            self.msg_data['depth_map'].append(depth_map.copy())
            self.msg_data['dense_depth_map'].append(dense_depth_map.copy())
            self.msg_data['overlay_image'].append(overlay_image.copy())
            self.msg_data['thermal_msg'].append(undistorted_thermal.copy())
            self.msg_data['image_left_msg'].append(undistorted_left.copy())
            self.msg_data['image_right_msg'].append(undistorted_right.copy())

            rospy.loginfo(f"Saved synchronized data for timestamp {timestamp}")

        except Exception as e:
            rospy.logwarn(f"Failed to save synchronized data: {e}")

    def save_image(self, folder, filename, image):
        """Save image to the specified folder with the given filename."""
        path = os.path.join(self.output_folder, folder, filename)
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                if image.ndim == 2:
                    cv2.imwrite(path, image)
                else:
                    cv2.imwrite(path, image)
            elif image.dtype == np.uint8:
                cv2.imwrite(path, image)
            else:
                rospy.logwarn(f"Unsupported image data type for {path}")
        elif isinstance(image, (float, int)):
            np.save(path, image)
        else:
            rospy.logwarn(f"Unsupported image format for {path}")

    def imu_callback(self, msg):
        if self.imu_counter <= 600:
            self.imu_processor.beta = 0.8
            self.imu_counter += 1
        else:
            self.imu_processor.beta = 0.05
        self.imu_processor.imu_update(msg)
        self.roll_pitch_yaw = np.roll(self.roll_pitch_yaw, -1, axis=0)
        self.roll_pitch_yaw[-1] = np.radians([
            self.imu_processor.roll,
            self.imu_processor.pitch,
            self.imu_processor.heading
        ])
        self.gyro_msgs = np.roll(self.gyro_msgs, -1, axis=0)
        self.accel_msgs = np.roll(self.accel_msgs, -1, axis=0)
        self.gyro_msgs[-1] = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        self.accel_msgs[-1] = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

    def mag_callback(self, msg):
        self.imu_processor.mag_update(msg)

    def joy_callback(self, msg):
        self.joy_msg = msg

    def odom_callback(self, msg):
        self.odom = msg
        self.odom_msgs = np.roll(self.odom_msgs, -1, axis=0)
        twist = msg.twist.twist
        self.odom_msgs[-1] = np.array([twist.linear.x, twist.linear.y, twist.angular.z])

    def process_depth_maps(self):
        # Placeholder if additional processing is needed
        pass

    def save_data(self):
        # Define directories
        dirs = [
            'image_left',
            'image_right',
            'thermal',
            'depth_map',
            'dense_depth_map',
            'overlay_image'
        ]
        for d in dirs:
            os.makedirs(os.path.join(self.output_folder, d), exist_ok=True)

        # Process velocity and odometry data
        data = {}
        data_length = len(self.msg_data['image_left_msg'])

        # Process resultant velocity, omega, roll, etc.
        data['res_vel_omega_roll_slde_bump'], data['triplets'] = data_calculation.process_resultant_vel(self.msg_data)

        # Ensure all data lists are trimmed to the same length
        data_length = len(data['res_vel_omega_roll_slde_bump'])
        data['cmd_vel_msg'] = list(self.msg_data['cmd_vel_msg'])[:data_length]
        data['odom_1sec_msg'] = list(self.msg_data['odom_1sec_msg'])[:data_length]
        data['odom'] = data_calculation.process_odom_vel_data(self.msg_data)
        data['velocity_msg'] = list(self.msg_data['velocity_msg'])[:data_length]
        data['poses'] = list(self.msg_data['poses'])[:data_length]
        data['accel_msg'] = list(self.msg_data['accel_msg'])[:data_length]
        data['gyro_msg'] = list(self.msg_data['gyro_msg'])[:data_length]
        data['time_stamp'] = list(self.msg_data['time_stamp'])[:data_length]
        data['roll_pitch_yaw'] = list(self.msg_data['roll_pitch_yaw'])[:data_length]
        data['lidar_points'] = list(self.msg_data['lidar_msg'])[:data_length]
        data['thermal_cam_info'] = self.cam_info_msg

        # Process patches using image_processing.py
        data['patches'], data['patches_found'] = image_processing.process_bev_image_and_patches(self.msg_data)

        # Save the data as a pickle file with unique filename
        path = os.path.join(self.output_folder, f"processed_data_{self.counter}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        rospy.loginfo(f"Saved data successfully to {path}")
        cprint('Data length: ' + str(data_length), 'green', attrs=['bold'])

def threading_function(bag_path, transform_file, output_folder, just_the_name):
    data_recorder = ListenRecordData(bag_path, transform_file, output_folder)
    if (len(data_recorder.msg_data['image_left_msg']) > 0 and
        len(data_recorder.msg_data['image_right_msg']) > 0 and
        len(data_recorder.msg_data['thermal_msg']) > 0 and
        len(data_recorder.msg_data['lidar_msg']) > 0):
        data_recorder.save_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Process ROS bag files offline and save aligned data.'
    )
    parser.add_argument(
        '-b', '--bag_folder', type=str, required=True,
        help='Folder containing the ROS bag files to process.'
    )
    parser.add_argument(
        '-t', '--transform_file', type=str, required=True,
        help='Path to the transformation matrix JSON file.'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='output_data',
        help='Output directory to save processed data.'
    )
    args = parser.parse_args()

    if not os.path.exists(args.bag_folder):
        cprint(args.bag_folder, 'red', attrs=['bold'])
        raise FileNotFoundError('ROS bag folder not found')
    else:
        list_of_bags = [f for f in os.listdir(args.bag_folder) if f.endswith('.bag')]
        threading_array = []
        for each in list_of_bags:
            just_the_name = each.split('.')[0]
            # Create subdirectories for each bag file
            bag_output_folder = os.path.join(args.output, just_the_name)
            os.makedirs(os.path.join(bag_output_folder, 'image_left'), exist_ok=True)
            os.makedirs(os.path.join(bag_output_folder, 'image_right'), exist_ok=True)
            os.makedirs(os.path.join(bag_output_folder, 'thermal'), exist_ok=True)
            os.makedirs(os.path.join(bag_output_folder, 'depth_map'), exist_ok=True)
            os.makedirs(os.path.join(bag_output_folder, 'dense_depth_map'), exist_ok=True)
            os.makedirs(os.path.join(bag_output_folder, 'overlay_image'), exist_ok=True)
            each_path = os.path.join(args.bag_folder, each)
            threading_array.append(threading.Thread(
                target=threading_function,
                args=(each_path, args.transform_file, bag_output_folder, just_the_name)
            ))
            threading_array[-1].start()
            print(f"Processing bag: {each} in thread: {threading_array[-1].name}")
        for thread in threading_array:
            thread.join()
        cprint("All bags have been processed.", 'green', attrs=['bold'])
        exit(0)
