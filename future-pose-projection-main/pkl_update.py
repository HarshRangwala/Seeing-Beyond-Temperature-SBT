import os
import pickle
import numpy as np
from collections import OrderedDict

# --- CONFIGURATION (Change these to your new system's paths) ---
file_path = '/home/harsh/Downloads/bags/ch0001/WC1_2024-08-27_19-59-04_chunk0001.pkl'  # Original pickle file
new_thermal_base_dir = '/home/harsh/Downloads/bags/ch0001/thermal_WC1_2024-08-27_19-59-04_chunk0001/'  # Base directory for thermal images
output_file_path = '/home/harsh/Downloads/bags/ch0001/WC1_2024-08-27_19-59-04_chunk0001_updated.pkl'  # Updated pickle file


# import matplotlib.pyplot as plt

# def display_imu_samples(pickle_file, n_samples=10):
#     """
#     Display N samples of IMU data from a pickle file
    
#     Args:
#         pickle_file (str): Path to the pickle file containing IMU data
#         n_samples (int): Number of samples to display
#     """
#     # Check if file exists
#     if not os.path.exists(pickle_file):
#         print(f"Error: File {pickle_file} not found")
#         return
    
#     # Load the pickle file
#     try:
#         with open(pickle_file, 'rb') as f:
#             data = pickle.load(f)
#     except Exception as e:
#         print(f"Error loading pickle file: {e}")
#         return
    
#     # Check if IMU data exists in the pickle file
#     if 'imu_poses' not in data:
#         print("No IMU data found in the pickle file")
#         return
    
#     # Get IMU data
#     imu_data = data['imu_poses']
    
#     # Limit to N samples
#     samples_to_show = min(n_samples, len(imu_data))
    
#     # Display the samples
#     print(f"\nShowing {samples_to_show} IMU samples out of {len(imu_data)} total samples:\n")
#     print("-" * 80)
    
#     for i in range(samples_to_show):
#         sample = imu_data[i]
#         print(f"Sample {i+1}:")
#         print(f"  Timestamp: {sample['timestamp']:.6f}")
#         print(f"  Roll: {sample['roll']:.6f}")
#         print(f"  Pitch: {sample['pitch']:.6f}")
#         print(f"  Heading: {sample['heading']:.6f}")
#         print(f"  Acceleration (m/s²): X={sample['accel_x']:.6f}, Y={sample['accel_y']:.6f}, Z={sample['accel_z']:.6f}")
#         print("-" * 80)
    
#     # Plot the first N samples
#     timestamps = [sample['timestamp'] for sample in imu_data[:n_samples]]
    
#     # Create a figure with subplots
#     fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    
#     # Plot orientation
#     axs[0, 0].plot(timestamps, [sample['roll'] for sample in imu_data[:n_samples]])
#     axs[0, 0].set_title('Roll')
#     axs[0, 0].set_xlabel('Timestamp')
#     axs[0, 0].set_ylabel('Radians')
    
#     axs[0, 1].plot(timestamps, [sample['pitch'] for sample in imu_data[:n_samples]])
#     axs[0, 1].set_title('Pitch')
#     axs[0, 1].set_xlabel('Timestamp')
#     axs[0, 1].set_ylabel('Radians')
    
#     axs[1, 0].plot(timestamps, [sample['heading'] for sample in imu_data[:n_samples]])
#     axs[1, 0].set_title('Heading')
#     axs[1, 0].set_xlabel('Timestamp')
#     axs[1, 0].set_ylabel('Radians')
    
#     # Plot acceleration
#     axs[1, 1].plot(timestamps, [sample['accel_x'] for sample in imu_data[:n_samples]])
#     axs[1, 1].set_title('Acceleration X')
#     axs[1, 1].set_xlabel('Timestamp')
#     axs[1, 1].set_ylabel('m/s²')
    
#     axs[2, 0].plot(timestamps, [sample['accel_y'] for sample in imu_data[:n_samples]])
#     axs[2, 0].set_title('Acceleration Y')
#     axs[2, 0].set_xlabel('Timestamp')
#     axs[2, 0].set_ylabel('m/s²')
    
#     axs[2, 1].plot(timestamps, [sample['accel_z'] for sample in imu_data[:n_samples]])
#     axs[2, 1].set_title('Acceleration Z')
#     axs[2, 1].set_xlabel('Timestamp')
#     axs[2, 1].set_ylabel('m/s²')
    
#     plt.tight_layout()
#     plt.show()

# # Example usage
# if __name__ == "__main__":
#     pickle_file = '/home/harsh/Downloads/bags/BL_2024-09-04_19-18-16_chunk0000.pkl'
#     try:
#         n_samples = int(input("Enter the number of samples to display: "))
#     except ValueError:
#         print("Invalid input. Using default value of 10 samples.")
#         n_samples = 10
    
#     display_imu_samples(pickle_file, n_samples)



# --- Update the pickle file ---
try:
    # Load the original pickle file
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    # Analyze the data structure
    print("\n--- Original Data Structure Analysis ---")
    for key in data.keys():
        if isinstance(data[key], list):
            print(f"Key: {key}, Type: {type(data[key])}, Length: {len(data[key])}")
            
            # Show sample data for each key
            if len(data[key]) > 0:
                if key == 'thermal_npaths':
                    print(f"  Sample path: {data[key][0]}")
                elif key == 'thermal_timestamps' or key == 'odom_timestamps':
                    print(f"  Sample timestamp: {data[key][0]}")
                elif key == 'odom_poses':
                    print(f"  Sample odom data: {data[key][0]}")
                    print(f"  Odom data keys: {list(data[key][0].keys())}")
                elif key == 'imu_poses':
                    print(f"  Sample IMU data: {data[key][0]}")
                    print(f"  IMU data keys: {list(data[key][0].keys())}")
        else:
            print(f"Key: {key}, Type: {type(data[key])}")
    
    # Update the thermal image paths
    old_thermal_paths = data['thermal_npaths']
    new_thermal_paths = []
    
    for old_path in old_thermal_paths:
        # Extract the filename from the old path
        filename = os.path.basename(old_path)
        
        # Construct the new path
        new_path = os.path.join(new_thermal_base_dir, filename)
        new_thermal_paths.append(new_path)
    
    # Replace old paths with new paths
    data['thermal_npaths'] = new_thermal_paths
    
    # Save the modified data
    with open(output_file_path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"\nPickle file updated. Saved to: {output_file_path}")
    
    # --- Check if the images are accessible ---
    accessible_paths = []
    inaccessible_paths = []
    
    for path in new_thermal_paths:
        if os.path.exists(path):
            accessible_paths.append(path)
        else:
            inaccessible_paths.append(path)
    
    print(f"\n--- Accessibility Check Results ---")
    print(f"Total paths: {len(new_thermal_paths)}")
    print(f"Accessible paths: {len(accessible_paths)}")
    print(f"Inaccessible paths: {len(inaccessible_paths)}")
    
    if inaccessible_paths:
        print("\nSample of inaccessible paths:")
        for path in inaccessible_paths[:5]:  # Show first 5 inaccessible paths
            print(f"  - {path}")
        
        if len(inaccessible_paths) > 5:
            print(f"  ... and {len(inaccessible_paths) - 5} more")
    
    # --- Analyze data alignment ---
    print("\n--- Data Alignment Analysis ---")
    
    # Check if all data arrays have the same length
    data_lengths = OrderedDict([
        ('thermal_npaths', len(data['thermal_npaths'])),
        ('thermal_timestamps', len(data['thermal_timestamps'])),
        ('odom_poses', len(data['odom_poses'])),
        ('odom_timestamps', len(data['odom_timestamps'])),
    ])
    
    if 'imu_poses' in data:
        data_lengths['imu_poses'] = len(data['imu_poses'])
    
    print("Data lengths:")
    for key, length in data_lengths.items():
        print(f"  {key}: {length}")
    
    # Check timestamp ranges
    if len(data['thermal_timestamps']) > 0:
        print("\nTimestamp ranges:")
        print(f"  Thermal: {min(data['thermal_timestamps']):.3f} to {max(data['thermal_timestamps']):.3f}")
        print(f"  Odometry: {min(data['odom_timestamps']):.3f} to {max(data['odom_timestamps']):.3f}")
        
        if 'imu_poses' in data and len(data['imu_poses']) > 0:
            imu_timestamps = [item['timestamp'] for item in data['imu_poses']]
            print(f"  IMU: {min(imu_timestamps):.3f} to {max(imu_timestamps):.3f}")
    
except Exception as e:
    print(f"Error: {e}")
