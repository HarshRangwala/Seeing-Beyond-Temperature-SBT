#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from tqdm import tqdm

# Constants for roughness score calculation
WEIGHT_ACCEL_Z_JERK = 0.075  # Weight for |dz| from accel (vertical impact sharpness)
WEIGHT_GYRO_X_ACCEL = 0.475 # Weight for |dx| from gyro (roll change sharpness)
WEIGHT_GYRO_Y_ACCEL = 0.475

# Annotation settings
ANNOTATION_THRESHOLD_PERCENTILE = 70 # Show filenames for events in the top % of roughness score
MAX_ANNOTATIONS = 45 # Limit the number of annotations to avoid clutter

class IMUProcessor:
    def __init__(self, window_size=15):
        self.window_size = window_size
        
    def process_accl_msg(self, msg_data):
        acc_data = np.array(msg_data['accel_msg']).reshape(len(msg_data['accel_msg']), 400, 3)[:,-20:,:]
        x = np.clip(acc_data[:, :, 0], -10, 10)
        y = np.clip(acc_data[:, :, 1], -10, 10)
        z = np.clip(acc_data[:, :, 2]-10.1, -10, 10)

        # Use the configurable window size
        window = np.ones(self.window_size)/self.window_size
        
        x = np.median(np.convolve(x.flatten(), window, mode='same').reshape(x.shape[0], 20), axis=1)
        y = np.median(np.convolve(y.flatten(), window, mode='same').reshape(y.shape[0], 20), axis=1)
        z = np.median(np.convolve(z.flatten(), window, mode='same').reshape(z.shape[0], 20), axis=1)
        
        dx = np.zeros(x.shape[0])
        dy = np.zeros(y.shape[0])
        dz = np.zeros(z.shape[0])
        
        dx[1:] = np.convolve(np.diff(x, axis=0), window, mode='same') / 2
        dy[1:] = np.convolve(np.diff(y, axis=0), window, mode='same') / 2
        dz[1:] = np.convolve(np.diff(z, axis=0), window, mode='same') / 1.2
        
        x = np.clip(x,-10,10) / 10
        y = np.clip(y, -10,10) / 9
        z = np.clip(z, -10,10) / 7
        
        return np.array([x, y, z, dx, dy, dz]).T

    def process_gyro_msg(self, msg_data):
        gyro_data = np.array(msg_data['gyro_msg']).reshape(len(msg_data['gyro_msg']), 400, 3)[:,-20:,:]
        x = np.clip(gyro_data[:, :, 0], -10, 10)
        y = np.clip(gyro_data[:, :, 1], -10, 10)
        z = np.clip(gyro_data[:, :, 2], -10, 10)
        
        # Use the configurable window size
        window = np.ones(self.window_size)/self.window_size
        
        x = np.median(np.convolve(x.flatten(), window, mode='same').reshape(x.shape[0], 20), axis=1)
        y = np.median(np.convolve(y.flatten(), window, mode='same').reshape(y.shape[0], 20), axis=1)
        z = np.median(np.convolve(z.flatten(), window, mode='same').reshape(z.shape[0], 20), axis=1)
        
        dx = np.zeros(x.shape[0])
        dy = np.zeros(y.shape[0])
        dz = np.zeros(z.shape[0])
        
        dx[1:] = np.clip(np.convolve(np.diff(x, axis=0), window, mode='same') * 3.8, -1, 1)
        dy[1:] = np.clip(np.convolve(np.diff(y, axis=0), window, mode='same') * 2.2, -1, 1)
        dz[1:] = np.clip(np.convolve(np.diff(z, axis=0), window, mode='same') * 2, -1, 1)
        
        x = np.clip(x,-10,10) / 2 
        y = np.clip(y,-10,10) / 2
        z = np.clip(z,-10,10) / 2.4

        return np.array([x, y, z, dx, dy, dz]).T

def plot_imu_roughness_analysis(roughness_score2, processed_gyro, processed_accel, timestamps, thermal_paths, title_base, window_size):
    """
    Plots rates of change, calculates a combined roughness score, and 
    annotates high-score events with thermal image filenames.
    
    Args:
        processed_gyro: Processed gyroscope data
        processed_accel: Processed accelerometer data
        timestamps: Time stamps for the data points
        thermal_paths: Paths to thermal images
        title_base: Base name for the plot title
    """
    # Basic validation
    if not (timestamps.ndim == 1 and
            processed_gyro.ndim == 2 and processed_gyro.shape[1] == 6 and
            processed_accel.ndim == 2 and processed_accel.shape[1] == 6 and
            len(timestamps) == len(processed_gyro) == len(processed_accel) == len(thermal_paths)):
        print("Error: Data shape mismatch or inconsistency.")
        print(f"  Timestamps: {timestamps.shape}")
        print(f"  Gyro Proc: {processed_gyro.shape}")
        print(f"  Accel Proc: {processed_accel.shape}")
        print(f"  Thermal Paths: {len(thermal_paths)}")
        
        # Attempt to trim to min length if lengths mismatch slightly
        min_len = min(len(timestamps), len(processed_gyro), len(processed_accel), len(thermal_paths))
        if min_len > 0 and min_len < len(timestamps):
             print(f"Warning: Trimming data to minimum length: {min_len}")
             timestamps = timestamps[:min_len]
             processed_gyro = processed_gyro[:min_len]
             processed_accel = processed_accel[:min_len]
             thermal_paths = thermal_paths[:min_len]
        else:
             return # Exit if shapes are fundamentally wrong or empty

    if len(timestamps) == 0:
        print("Error: No data points found.")
        return

    # --- 2. Extract Data Components ---
    relative_time = timestamps - timestamps[0]

    # Accel components (ajx = accel jerk x, etc.)
    ajx = processed_accel[:, 3]
    ajy = processed_accel[:, 4]
    ajz = processed_accel[:, 5]

    # Gyro components (gax = gyro accel x, etc.)
    gax = processed_gyro[:, 3]
    gay = processed_gyro[:, 4]
    gaz = processed_gyro[:, 5]

    # --- 3. Calculate Combined Roughness Score ---
    # Using absolute values as roughness causes large deviations in either direction
    roughness_score = (WEIGHT_ACCEL_Z_JERK * np.abs(ajz) +
                       WEIGHT_GYRO_X_ACCEL * np.abs(gax) +
                       WEIGHT_GYRO_Y_ACCEL * np.abs(gay))
    roughness_score = np.convolve(roughness_score, np.ones(100)/100, mode='same')
    roughness_score =  (np.clip(roughness_score, 0.01, 0.15) -0.01)/(0.15-0.01)
    # --- 4. Create Compact Plot ---
    # 7 rows: ajx, ajy, ajz, gax, gay, gaz, score
    fig, axes = plt.subplots(8, 1, figsize=(18, 14), sharex=True, gridspec_kw={'hspace': 0.4})
    fig.suptitle(f'IMU Rate of Change Analysis: {title_base}', fontsize=16, y=0.99)

    plot_params = {'linewidth': 1.2, 'alpha': 0.9}
    grid_params = {'linestyle': '--', 'alpha': 0.6}
    label_fontsize = 9
    title_fontsize = 11
    tick_fontsize = 8

    # Plot Accel Jerk
    axes[0].plot(relative_time, ajx, label='Accel Jerk X (ajx)', color='royalblue', **plot_params)
    axes[0].set_title('Linear Jerk (Accel Rate of Change)', fontsize=title_fontsize, loc='left')
    axes[0].set_ylabel('X', fontsize=label_fontsize)
    axes[0].grid(**grid_params)
    axes[0].tick_params(axis='y', labelsize=tick_fontsize)

    axes[1].plot(relative_time, ajy, label='Accel Jerk Y (ajy)', color='darkorange', **plot_params)
    axes[1].set_ylabel('Y', fontsize=label_fontsize)
    axes[1].grid(**grid_params)
    axes[1].tick_params(axis='y', labelsize=tick_fontsize)

    axes[2].plot(relative_time, ajz, label='Accel Jerk Z (ajz)', color='forestgreen', **plot_params)
    axes[2].set_ylabel('Z', fontsize=label_fontsize)
    axes[2].grid(**grid_params)
    axes[2].tick_params(axis='y', labelsize=tick_fontsize)

    # Plot Gyro Accel
    axes[3].plot(relative_time, gax, label='Gyro Accel X (gax)', color='crimson', **plot_params)
    axes[3].set_title('Angular Acceleration (Gyro Rate of Change)', fontsize=title_fontsize, loc='left')
    axes[3].set_ylabel('Roll (X)', fontsize=label_fontsize)
    axes[3].grid(**grid_params)
    axes[3].tick_params(axis='y', labelsize=tick_fontsize)

    axes[4].plot(relative_time, gay, label='Gyro Accel Y (gay)', color='darkviolet', **plot_params)
    axes[4].set_ylabel('Pitch (Y)', fontsize=label_fontsize)
    axes[4].grid(**grid_params)
    axes[4].tick_params(axis='y', labelsize=tick_fontsize)

    axes[5].plot(relative_time, gaz, label='Gyro Accel Z (gaz)', color='saddlebrown', **plot_params)
    axes[5].set_ylabel('Yaw (Z)', fontsize=label_fontsize)
    axes[5].grid(**grid_params)
    axes[5].tick_params(axis='y', labelsize=tick_fontsize)

    # Plot Combined Roughness Score
    axes[6].plot(relative_time, roughness_score, label='Combined Score', color='black', linewidth=1.5)
    axes[6].set_title('Combined Roughness Indicator Score', fontsize=title_fontsize, loc='left')
    axes[6].set_ylabel('Score', fontsize=label_fontsize)
    axes[6].grid(**grid_params)
    axes[6].tick_params(axis='y', labelsize=tick_fontsize)

    axes[7].plot(relative_time, roughness_score2, label='Combined Score', color='black', linewidth=1.5)
    axes[7].set_title('Combined Roughness2 Indicator Score', fontsize=title_fontsize, loc='left')
    axes[7].set_ylabel('Score', fontsize=label_fontsize)
    axes[7].set_xlabel('Time (s)', fontsize=12)
    axes[7].grid(**grid_params)
    axes[7].tick_params(axis='both', labelsize=tick_fontsize)
    axes[7].legend(loc='upper right', fontsize=label_fontsize-1)

    # --- 5. Add Annotations for Significant Events ---
    threshold = np.percentile(roughness_score2, ANNOTATION_THRESHOLD_PERCENTILE)
    event_indices = np.where(roughness_score2 > threshold)[0]

    if len(event_indices) > 0:
        # Select a limited number of points to annotate, prioritizing highest scores
        scores_at_events = roughness_score2[event_indices]
        sorted_event_indices = event_indices[np.argsort(scores_at_events)[::-1]] # Sort by score descending
        indices_to_annotate = sorted_event_indices[:min(MAX_ANNOTATIONS, len(sorted_event_indices))]

        # Sort by time for plotting order
        indices_to_annotate = np.sort(indices_to_annotate)

        print(f"\nAnnotating {len(indices_to_annotate)} events above threshold {threshold:.3f} (Top {100-ANNOTATION_THRESHOLD_PERCENTILE}%):")

        last_annotated_time = -np.inf
        min_time_diff = (relative_time[-1] - relative_time[0]) / (MAX_ANNOTATIONS * 1.5) # Heuristic for spacing

        for i, idx in enumerate(indices_to_annotate):
            t = relative_time[idx]
            score_val = roughness_score2[idx]
            filename = os.path.basename(thermal_paths[idx])

            # Basic check to avoid cluttering annotations too close in time
            if t < last_annotated_time + min_time_diff and i > 0:
                continue

            print(f"  Time: {t:.2f}s, Score: {score_val:.3f}, Image: {filename}")

            # Add annotation to the score plot (axes[6])
            axes[7].annotate(filename,
                             xy=(t, score_val), # Point to annotate
                             xytext=(0, 15 + (i % 3) * 10), # Text offset (pixels). Stagger vertically.
                             textcoords="offset points",
                             ha='center', va='bottom',
                             fontsize=7,
                             arrowprops=dict(arrowstyle="->", color='gray', lw=0.5,
                                             connectionstyle="arc3,rad=-0.1")) # Slight curve
            # Add a vertical line across all plots
            for ax in axes:
                ax.axvline(t, color='red', linestyle=':', linewidth=0.7, alpha=0.8)

            last_annotated_time = t

    # --- 6. Final Touches & Save ---
    for ax in axes[:-1]: # Hide x-tick labels on all but the bottom plot
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout

    output_filename = f"plot_imu_analysis_{title_base}_window{window_size}.png"
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process and visualize IMU data with configurable window sizes')
    parser.add_argument('pickle_file', type=str, help='Path to the pickle file containing IMU data')
    parser.add_argument('--window-size', type=int, default=15, help='Window size for convolution (default: 15)')
    parser.add_argument('--approach', type=str, default='1m', choices=['1sec', '1m', 'const'], 
                        help='Which IMU data approach to use (default: 1m)')
    args = parser.parse_args()

    # Load the pickle file
    print(f"Loading data from {args.pickle_file}...")
    with open(args.pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # Print available keys for debugging
    print(f"Available keys in pickle file: {data.keys()}")
    
    # Extract base name for plot title
    title_base = os.path.splitext(os.path.basename(args.pickle_file))[0]
    window_size = args.window_size
    approach = args.approach
    
    # Create processor with specified window size
    processor = IMUProcessor(window_size=window_size)
    
    # Prepare data for processing based on selected approach
    print(f"Processing IMU data with window size {window_size} using approach '{approach}'...")
    
    msg_data = {
        'accel_msg': data[f'imu_accel_{approach}'],
        'gyro_msg': data[f'imu_gyro_{approach}']
    }
    
    # Process the data
    processed_accel = processor.process_accl_msg(msg_data)
    processed_gyro = processor.process_gyro_msg(msg_data)
    roughness_score2 = data['roughness_score_1m']
    # Plot the results
    print("Generating plot...")
    plot_imu_roughness_analysis(
        roughness_score2 = roughness_score2,
        processed_gyro=processed_gyro,
        processed_accel=processed_accel,
        timestamps=np.array(data['time_stamp']),
        thermal_paths=data['thermal_paths'],
        title_base=f"{title_base}_{approach}",
        window_size = window_size
    )

if __name__ == "__main__":
    main()
