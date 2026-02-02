#!/usr/bin/env python3

import os
import argparse
import csv
import rosbag
from rosbag import Bag
import time

TOPICS_OF_INTEREST = [
    '/sensor_suite/left_camera_optical/image_color/compressed',
    '/sensor_suite/right_camera_optical/image_color/compressed',
    '/sensor_suite/lwir/lwir/image_raw/compressed',
    '/sensor_suite/ouster/points'
]

def process_bag_file(bag_path):
    """Process a single ROS bag file and extract duration/topic counts"""
    with Bag(bag_path, 'r') as bag:
        start_time = bag.get_start_time()
        end_time = bag.get_end_time()
        duration = end_time - start_time
        topic_info = bag.get_type_and_topic_info()[1]  # Get topic statistics
        
        counts = {}
        for topic in TOPICS_OF_INTEREST:
            counts[topic] = topic_info[topic].message_count if topic in topic_info else 0

        return {
            'filename': os.path.basename(bag_path),
            'duration': duration,
            **counts
        }


def format_duration(seconds):
    """Convert seconds to HH:MM:SS format"""
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def main():
    parser = argparse.ArgumentParser(description='Analyze ROS bag files')
    parser.add_argument('input_dir', help='Directory containing bag files')
    args = parser.parse_args()

    # Find all bag files recursively
    bag_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.bag'):
                bag_files.append(os.path.join(root, file))

    if not bag_files:
        print("No .bag files found in specified directory")
        return

    # Process all bags
    results = []
    for bag_path in bag_files:
        try:
            results.append(process_bag_file(bag_path))
        except Exception as e:
            print(f"Error processing {bag_path}: {str(e)}")

    # Write CSV report
    csv_file = 'rosbag_summary.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['filename', 'duration (s)'] + TOPICS_OF_INTEREST
        writer.writerow(headers)
        
        for res in results:
            row = [res['filename'], res['duration']] + [res[t] for t in TOPICS_OF_INTEREST]
            writer.writerow(row)

    # Calculate totals
    total_duration = sum(r['duration'] for r in results)
    total_counts = {t: sum(r[t] for r in results) for t in TOPICS_OF_INTEREST}

    # Print summary
    print(f"\n{' Analysis Summary ':=^80}")
    print(f"Processed {len(results)} bag files")
    print(f"\nTotal duration: {total_duration:.2f} sec ({format_duration(total_duration)})")
    
    print("\nMessage counts:")
    for topic in TOPICS_OF_INTEREST:
        print(f"  {topic}: {total_counts[topic]:,}")

    print(f"\nCSV report saved to {csv_file}")

if __name__ == '__main__':
    main()