import os
import rosbag  # Make sure you have rosbag installed: `pip install rosbags`
import numpy as np
import pandas as pd
from tqdm import tqdm
from termcolor import cprint

def split_poses_by_bag(
    bag_root_dir: str,
    combined_poses_file: str,
    output_dir: str,
    topic_name: str = "/sensor_suite/lwir/lwir/image_raw/compressed",
) -> None:
    """
    Splits a combined poses CSV file into individual CSV files per bag file,
    based on timestamp matching.  Prints start and end times for both bag files
    and corresponding pose files, as well as the total pose count.

    Args:
        bag_root_dir: Root directory containing bag files (or folders of bags).
        combined_poses_file: Path to the combined poses CSV file (from DLIO).
        output_dir: Directory to save the individual pose CSV files.
        topic_name: Name of the image topic for selecting the timestamp.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load the combined poses CSV into a pandas DataFrame
    poses_df = pd.read_csv(combined_poses_file)
    poses_df['timestamp'] = pd.to_numeric(poses_df['timestamp'], errors='coerce')
    poses_df.sort_values('timestamp', inplace=True)
    poses_df.dropna(subset=['timestamp'], inplace=True)
    total_poses_combined = len(poses_df)  # Get total pose count *before* splitting
    cprint(f"Total poses in combined file ({combined_poses_file}): {total_poses_combined}", "magenta")


    # Find all bag files (recursively)
    bag_files = []
    for root, _, files in os.walk(bag_root_dir):
        for file in files:
            if file.endswith(".bag"):
                bag_files.append(os.path.join(root, file))
    bag_files.sort()  # Ensure consistent ordering

    if not bag_files:
        cprint(f"Error: No bag files found in {bag_root_dir}", "red")
        return

    for bag_file in tqdm(bag_files, desc="Processing bag files"):
        bag_name = os.path.basename(bag_file)
        bag_prefix = os.path.splitext(bag_name)[0]  # e.g., "my_bag_file"
        output_pose_file = os.path.join(output_dir, f"pose_{bag_prefix}.csv")

        # 1. Get Bag File Timestamp Range
        try:
            bag = rosbag.Bag(bag_file, 'r')
            # Get start and end time using timestamps from the image topic
            start_time = None
            end_time = None

            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                if start_time is None:
                    start_time = t.to_sec()
                end_time = t.to_sec()  # Update end_time in each iteration
            bag.close()

            if start_time is None or end_time is None:
                cprint(f"Warning: Could not determine time range for {bag_file}. Skipping.", "yellow")
                continue
            cprint(f"Bag file: {bag_name}  Start: {start_time:.3f}, End: {end_time:.3f}", "cyan")

        except rosbag.ROSBagException as e:
            cprint(f"Error opening {bag_file}: {e}. Skipping.", "red")
            continue

        # 2. Filter Poses DataFrame
        filtered_poses_df = poses_df[
            (poses_df['timestamp'] >= start_time) & (poses_df['timestamp'] <= end_time)
        ]

        # 3. Save Filtered Poses
        if not filtered_poses_df.empty:
            filtered_poses_df.to_csv(output_pose_file, index=False)
            cprint(f"Saved poses for {bag_name} to {output_pose_file}", "green")
            cprint(f"\tPose file: Start Time: {filtered_poses_df['timestamp'].iloc[0]:.3f}, End Time: {filtered_poses_df['timestamp'].iloc[-1]:.3f}, Total Poses: {len(filtered_poses_df)}", "green")
        else:
            cprint(f"Warning: No poses found within the time range of {bag_file}.  No CSV created.", "yellow")


# --- Example Usage ---
if __name__ == "__main__":
    bag_root_directory = "/media/harshr/Data/wc_m2p2_decomp/3"  # e.g., '/data/my_bags'
    combined_poses_csv = "/media/harshr/Data/wc_m2p2_decomp/parsed_poses/poses_loop_3.csv"  # e.g., '/data/dlio_poses/poses_1.csv'
    output_directory = "/home/harshr/NV_cahsor/data/west_campus_data/3/"   # e.g., '/data/processed_poses'

    split_poses_by_bag(bag_root_directory, combined_poses_csv, output_directory)

    cprint("Finished splitting pose files!", "green")