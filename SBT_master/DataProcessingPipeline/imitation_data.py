#!/usr/bin/env python3
import os
import pickle
import numpy as np
import argparse
import rospy
import rosbag
import cv2
from termcolor import cprint
from tf.transformations import euler_from_quaternion

# Assuming your helper file is in a 'Helpers' subdirectory
from Helpers import traversability_helpers as trav_helpers

class DatasetGenerator:
    """
    processes and synchronizes ROS bag data for behavioral cloning.

    This class uses the command velocity message as the synchronization trigger,
    ensuring that each state (image, pose history) is correctly paired with
    the action (command) that resulted from it.
    """
    def __init__(self, bag_path, out_folder, pose_seq_len=20, time_threshold=0.1):
        self.bag_path = bag_path
        self.bag_prefix = os.path.splitext(os.path.basename(bag_path))[0]
        self.out_folder = out_folder
        self.pose_seq_len = pose_seq_len
        self.time_threshold = time_threshold # Max allowed delay between state and action

        # Data stores for pre-processed bag topics
        self.thermal_data = [] # List of (timestamp, msg)
        self.odom_data = []    # List of (timestamp, [x,y,z,r,p,y])
        self.cmdvel_data = []  # List of (timestamp, [vx, wz])

        # Final synchronized samples
        self.synchronized_samples = []
        cprint(f"[{self.bag_prefix}] Initialized DatasetGenerator.", "green")

    @staticmethod
    def _get_euler_from_quat(orientation):
        """Helper to convert geometry_msgs/Quaternion to Euler angles."""
        q = orientation
        return euler_from_quaternion([q.x, q.y, q.z, q.w])

    @staticmethod
    def _compressed_img_to_cv2(compressed_msg):
        """Helper to decode a compressed image message."""
        np_arr = np.frombuffer(compressed_msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def _preprocess_data_from_bag(self):
        """
        Efficiently reads all required topics from the bag file into memory once.
        This avoids slow, repetitive bag parsing.
        """
        cprint(f"[{self.bag_prefix}] Pre-processing data from bag file...", "yellow")
        try:
            with rosbag.Bag(self.bag_path, 'r') as bag:
                topics = [
                    '/sensor_suite/lwir/lwir/image_raw/compressed',
                    '/odometry/filtered',
                    '/husky_velocity_controller/cmd_vel_out'
                ]
                for topic, msg, t in bag.read_messages(topics=topics):
                    timestamp = msg.header.stamp.to_sec()
                    if topic == '/sensor_suite/lwir/lwir/image_raw/compressed':
                        self.thermal_data.append((timestamp, msg))
                    elif topic == '/odometry/filtered':
                        p = msg.pose.pose.position
                        roll, pitch, yaw = self._get_euler_from_quat(msg.pose.pose.orientation)
                        pose6d = np.array([p.x, p.y, p.z, roll, pitch, yaw], dtype=np.float32)
                        self.odom_data.append((timestamp, pose6d))
                    elif topic == '/husky_velocity_controller/cmd_vel_out':
                        cmd = np.array([msg.twist.linear.x, msg.twist.angular.z], dtype=np.float32)
                        self.cmdvel_data.append((timestamp, cmd))
        except Exception as e:
            cprint(f"[{self.bag_prefix}] Error during pre-processing: {e}", "red")
            return False

        if not self.cmdvel_data or not self.thermal_data or not self.odom_data:
            cprint(f"[{self.bag_prefix}] A required data topic was empty. Aborting.", "red")
            return False

        cprint(f"[{self.bag_prefix}] Pre-processing complete. Found:", "green")
        cprint(f"  - {len(self.thermal_data)} Thermal Images", "green")
        cprint(f"  - {len(self.odom_data)} Odometry Messages", "green")
        cprint(f"  - {len(self.cmdvel_data)} Command Velocities", "green")
        return True

    def _find_closest_in_past(self, target_ts, data_timestamps):
        """
        Finds the index of the latest item in `data_timestamps` that occurred
        BEFORE or AT the `target_ts`, within the time threshold.
        """
        # searchsorted finds the insertion point to maintain order.
        # The item just before this point is the closest in the past.
        idx = np.searchsorted(data_timestamps, target_ts, side='right')

        if idx == 0:
            return None # No items in the past

        closest_idx = idx - 1
        time_diff = target_ts - data_timestamps[closest_idx]

        if 0 <= time_diff <= self.time_threshold:
            return closest_idx
        return None

    def _synchronize_and_create_samples(self):
        """
        The core logic. Iterates through commands and finds corresponding states.
        """
        if not self.cmdvel_data: return

        cprint(f"[{self.bag_prefix}] Synchronizing samples...", "yellow")
        # Create NumPy arrays of timestamps for efficient searching
        thermal_timestamps = np.array([t for t, msg in self.thermal_data])
        odom_timestamps = np.array([t for t, pose in self.odom_data])
        odom_poses = np.array([pose for t, pose in self.odom_data])

        for cmd_ts, cmd_vel in self.cmdvel_data:
            # For each command (Action), find the corresponding state (Image)
            img_idx = self._find_closest_in_past(cmd_ts, thermal_timestamps)
            if img_idx is None:
                continue # No recent enough image for this command

            # Now find the pose history leading up to that image's timestamp
            img_ts = thermal_timestamps[img_idx]
            odom_idx = self._find_closest_in_past(img_ts, odom_timestamps)
            if odom_idx is None or odom_idx < self.pose_seq_len - 1:
                continue # Not enough odom history for a full sequence

            # We have a valid, synchronized sample. Extract the data.
            start_idx = odom_idx - self.pose_seq_len + 1
            pose_history_global = odom_poses[start_idx : odom_idx + 1]

            # Convert the pose sequence to the robot's reference frame
            ref_pose = pose_history_global[-1]
            pose_history_robot_frame = trav_helpers.to_robot_numpy(
                np.tile(ref_pose, (self.pose_seq_len, 1)),
                pose_history_global
            )

            sample = {
                'thermal_msg': self.thermal_data[img_idx][1],
                'pose_sequence': pose_history_robot_frame.astype(np.float32),
                'cmd_vel': cmd_vel
            }
            self.synchronized_samples.append(sample)

        cprint(f"[{self.bag_prefix}] Synchronization complete. Found {len(self.synchronized_samples)} valid samples.", "green")

    def save_dataset(self):
        """Saves the synchronized samples to disk (images and a pickle file)."""
        if not self.synchronized_samples:
            cprint(f"[{self.bag_prefix}] No samples to save.", "magenta")
            return

        cprint(f"[{self.bag_prefix}] Saving dataset...", "yellow")
        img_folder = os.path.join(self.out_folder, "thermal_images")
        os.makedirs(img_folder, exist_ok=True)

        # Final lists for the pickle file
        final_thermal_paths = []
        final_pose_sequences = []
        final_cmd_vels = []

        for idx, sample in enumerate(self.synchronized_samples):
            try:
                img = self._compressed_img_to_cv2(sample['thermal_msg'])
                fname = f"{self.bag_prefix}_{idx:06d}.png"
                fpath = os.path.join(img_folder, fname)
                relative_path = os.path.join("thermal_images", fname) # Store relative path
                cv2.imwrite(fpath, img)

                final_thermal_paths.append(relative_path)
                final_pose_sequences.append(sample['pose_sequence'])
                final_cmd_vels.append(sample['cmd_vel'])
            except Exception as e:
                cprint(f"[{self.bag_prefix}] Failed to save image for sample {idx}: {e}", "red")
                continue

        dataset = {
            'thermal_paths': final_thermal_paths,
            'pose_sequences': np.array(final_pose_sequences, dtype=np.float32),
            'cmd_vels': np.array(final_cmd_vels, dtype=np.float32)
        }

        pickle_path = os.path.join(self.out_folder, f"{self.bag_prefix}_dataset.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(dataset, f)

        cprint(f"[{self.bag_prefix}] Successfully saved dataset to {pickle_path} with {len(final_thermal_paths)} samples.", "green", attrs=['bold'])

    def run(self):
        """Executes the full pipeline for a single bag file."""
        if self._preprocess_data_from_bag():
            self._synchronize_and_create_samples()
            self.save_dataset()

def main():
    """Main function to handle command-line arguments and process all bags."""
    parser = argparse.ArgumentParser(description="Generate a correctly synchronized dataset for behavioral cloning.")
    parser.add_argument('-b', '--bag_folder', required=True, help="Input folder containing ROS bag files.")
    parser.add_argument('-o', '--out_folder', required=True, help="Root output folder for the dataset.")
    parser.add_argument('-t', '--time_threshold', type=float, default=0.1, help="Max allowed time difference (seconds) between state and action for synchronization.")
    parser.add_argument('-s', '--seq_len', type=int, default=20, help="Length of the pose history sequence.")
    args = parser.parse_args()

    # It's good practice to run rospy.init_node when interacting with ROS bag files
    try:
        rospy.init_node('dataset_generator_node', anonymous=True, disable_signals=True)
    except rospy.exceptions.ROSInitException:
        cprint("ROS node already initialized or not available. Proceeding...", "magenta")


    os.makedirs(args.out_folder, exist_ok=True)
    rosbag_files = sorted([f for f in os.listdir(args.bag_folder) if f.endswith(".bag")])

    if not rosbag_files:
        cprint(f"No .bag files found in {args.bag_folder}", "red")
        return

    cprint(f"Found {len(rosbag_files)} bags to process. Starting...", "cyan", attrs=['bold'])
    for bag_file in rosbag_files:
        bag_path = os.path.join(args.bag_folder, bag_file)
        cprint(f"\n" + "="*50, "blue")
        cprint(f"Processing bag: {bag_file}", "blue", attrs=['bold'])
        cprint("="*50, "blue")
        processor = DatasetGenerator(bag_path, args.out_folder, pose_seq_len=args.seq_len, time_threshold=args.time_threshold)
        processor.run()

    cprint("\nAll bags processed successfully!", "green", attrs=['bold', 'blink'])

if __name__ == '__main__':
    main()