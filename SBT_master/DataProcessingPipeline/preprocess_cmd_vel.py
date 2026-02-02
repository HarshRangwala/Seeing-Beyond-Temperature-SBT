import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def smooth_cmd_vel(cmd_vel_data, f_size=7):
    """
    Apply smoothing to command velocity data similar to the reference implementation.
    
    Args:
        cmd_vel_data: List or array of command velocity data
        f_size: Filter size for the moving average
        
    Returns:
        Smoothed command velocity data
    """
    cmd_vel = np.array(cmd_vel_data, dtype=np.float32)
    
    # Check if we have enough data to smooth
    if cmd_vel.shape[0] < f_size:
        return cmd_vel
    
    # Create output array (will be smaller due to 'valid' convolution)
    cmd_vel_filtered = np.zeros(
        (cmd_vel.shape[0] - f_size + 1, cmd_vel.shape[1]), dtype=np.float32
    )
    
    # Apply smoothing to each dimension
    for i in range(cmd_vel.shape[1]):
        cmd_vel_filtered[:, i] = np.convolve(
            cmd_vel[:, i], np.ones(f_size) / f_size, mode="valid"
        )
        
    return cmd_vel_filtered

def calculate_stats(all_smoothed_cmd_vel):
    """Calculate statistics for the smoothed command velocities"""
    all_smoothed_cmd_vel = np.vstack(all_smoothed_cmd_vel)
    
    stats = {
        "sm_cmd_vel_mean": np.mean(all_smoothed_cmd_vel, axis=0),
        "sm_cmd_vel_std": np.std(all_smoothed_cmd_vel, axis=0),
        "sm_cmd_vel_min": np.min(all_smoothed_cmd_vel, axis=0),
        "sm_cmd_vel_max": np.max(all_smoothed_cmd_vel, axis=0),
    }
    
    return stats

def visualize_smoothing(raw_cmd_vel, smoothed_cmd_vel, filename, output_dir):
    """
    Create visualizations comparing raw and smoothed command velocities
    
    Args:
        raw_cmd_vel: Original command velocity data
        smoothed_cmd_vel: Smoothed command velocity data
        filename: Name of the file being processed (for plot title)
        output_dir: Directory to save the plot
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Prepare time axis for both raw and smoothed data
    # For valid mode convolution, the smoothed data is shorter
    raw_time = np.arange(len(raw_cmd_vel))
    smoothed_time = np.arange(len(smoothed_cmd_vel)) + (len(raw_cmd_vel) - len(smoothed_cmd_vel))//2
    
    # Create figure with 2 subplots (linear and angular velocity)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot linear velocity
    axes[0].plot(raw_time, raw_cmd_vel[:, 0], 'b-', alpha=0.5, label='Raw Linear')
    axes[0].plot(smoothed_time, smoothed_cmd_vel[:, 0], 'r-', linewidth=2, label='Smoothed Linear')
    axes[0].set_ylabel('Linear Velocity (m/s)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot angular velocity
    axes[1].plot(raw_time, raw_cmd_vel[:, 1], 'b-', alpha=0.5, label='Raw Angular')
    axes[1].plot(smoothed_time, smoothed_cmd_vel[:, 1], 'r-', linewidth=2, label='Smoothed Angular')
    axes[1].set_ylabel('Angular Velocity (rad/s)')
    axes[1].set_xlabel('Time Steps')
    axes[1].legend()
    axes[1].grid(True)
    
    # Set title for the entire figure
    fig.suptitle(f'Command Velocity Smoothing: {Path(filename).stem}', fontsize=16)
    
    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save the figure
    plot_path = output_dir / f"{Path(filename).stem}_cmd_vel_smoothing.png"
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def create_summary_plot(all_data, output_dir):
    """
    Create a summary plot showing all smoothed command velocities
    
    Args:
        all_data: List of tuples (filename, raw_cmd_vel, smoothed_cmd_vel)
        output_dir: Directory to save the plot
    """
    output_dir = Path(output_dir)
    
    # Create a large figure
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Linear velocity subplot (all files)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("Linear Velocities (All Files)", fontsize=14)
    ax1.set_ylabel("Linear Velocity (m/s)")
    ax1.grid(True)
    
    # Angular velocity subplot (all files)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title("Angular Velocities (All Files)", fontsize=14)
    ax2.set_ylabel("Angular Velocity (rad/s)")
    ax2.set_xlabel("Time Steps (Normalized)")
    ax2.grid(True)
    
    # Plot each file's data with different colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_data)))
    
    for i, (filename, _, smoothed) in enumerate(all_data):
        # Normalize time to 0-1 range for comparison
        norm_time = np.linspace(0, 1, len(smoothed))
        label = Path(filename).stem[:20] + "..." if len(Path(filename).stem) > 20 else Path(filename).stem
        
        # Plot linear velocity
        ax1.plot(norm_time, smoothed[:, 0], color=colors[i], alpha=0.7, linewidth=1.5, label=label)
        
        # Plot angular velocity
        ax2.plot(norm_time, smoothed[:, 1], color=colors[i], alpha=0.7, linewidth=1.5, label=label)
    
    # Add legends
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=2)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=2)
    
    plt.tight_layout()
    
    # Save the summary plot
    summary_path = output_dir / "cmd_vel_smoothing_summary.png"
    plt.savefig(summary_path)
    plt.close()
    
    return summary_path

def process_cmd_vel_data(data_dir, output_stats_file, f_size=7, visualize=True, plots_dir=None):
    """
    Process all pickle files in the directory, smooth cmd_vel data,
    and generate statistics.
    
    Args:
        data_dir: Directory containing pickle files
        output_stats_file: Path to save the statistics
        f_size: Filter size for smoothing
        visualize: Whether to create visualization plots
        plots_dir: Directory to save visualization plots
    """
    data_dir = Path(data_dir)
    pkl_files = list(data_dir.glob("*.pkl"))
    
    print(f"Found {len(pkl_files)} pickle files in {data_dir}")
    
    all_smoothed_cmd_vel = []
    visualization_data = []
    
    for pkl_file in tqdm(pkl_files, desc="Processing files"):
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            
            # Check if the key exists
            if 'cmd_vel_msg' not in data:
                print(f"Warning: 'cmd_vel_msg' not found in {pkl_file}")
                continue
            
            # Get the raw command velocity data
            raw_cmd_vel = np.array(data['cmd_vel_msg'], dtype=np.float32)
            
            # Smooth command velocities
            smoothed_cmd_vel = smooth_cmd_vel(raw_cmd_vel, f_size)
            
            # Store the smoothed data back
            data['sm_cmd_vel'] = smoothed_cmd_vel
            
            # Add to collection for statistics
            all_smoothed_cmd_vel.append(smoothed_cmd_vel)
            
            # Save for visualization
            if visualize:
                visualization_data.append((pkl_file.name, raw_cmd_vel, smoothed_cmd_vel))
            
            # Save the updated data
            with open(pkl_file, "wb") as f:
                pickle.dump(data, f)
                
            print(f"Processed {pkl_file.name}: Added 'sm_cmd_vel' with shape {smoothed_cmd_vel.shape}")
            
            # Create individual file visualization
            if visualize and plots_dir:
                plot_path = visualize_smoothing(raw_cmd_vel, smoothed_cmd_vel, pkl_file.name, plots_dir)
                print(f"  Visualization saved to {plot_path}")
            
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
    
    # Calculate statistics
    if all_smoothed_cmd_vel:
        stats = calculate_stats(all_smoothed_cmd_vel)
        
        # Save statistics
        with open(output_stats_file, "wb") as f:
            pickle.dump(stats, f)
        
        print(f"Statistics saved to {output_stats_file}")
        print("Statistics summary:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
        # Create summary visualization
        if visualize and plots_dir and visualization_data:
            summary_path = create_summary_plot(visualization_data, plots_dir)
            print(f"Summary visualization saved to {summary_path}")
    else:
        print("No data processed. Cannot calculate statistics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process command velocity data with smoothing')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing pickle files')
    parser.add_argument('--output_stats', type=str, required=True,
                        help='Output file path for statistics')
    parser.add_argument('--filter_size', type=int, default=7,
                        help='Filter size for smoothing (default: 7)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--plots_dir', type=str, default='cmd_vel_plots',
                        help='Directory to save visualization plots')
    
    args = parser.parse_args()
    
    process_cmd_vel_data(
        args.data_dir, 
        args.output_stats, 
        args.filter_size,
        args.visualize,
        args.plots_dir
    )
