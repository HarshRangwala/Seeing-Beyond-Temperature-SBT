import os
import pickle
import glob
from collections import OrderedDict
import time
from tqdm import tqdm
from termcolor import cprint, colored
import sys

def process_pickle_files(source_dir, thermal_base_dir, output_dir):
    """
    Process all pickle files in the source directory, updating their thermal image paths
    and saving to the output directory.
    
    Args:
        source_dir (str): Directory containing original pickle files
        thermal_base_dir (str): Base directory containing thermal image folders
        output_dir (str): Directory to save updated pickle files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all pickle files
    pickle_files = glob.glob(os.path.join(source_dir, "*.pkl"))
    
    cprint(f"Found {len(pickle_files)} pickle files to process.", "cyan")
    
    success_count = 0
    warning_count = 0
    error_count = 0
    
    # Create progress bar
    with tqdm(total=len(pickle_files), desc=colored("Processing files", "blue"), 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        
        for pkl_file in pickle_files:
            # Get the base filename without extension
            base_filename = os.path.basename(pkl_file)
            file_prefix = os.path.splitext(base_filename)[0]  # Remove .pkl extension
            
            # Construct thermal directory path
            thermal_dir = os.path.join(thermal_base_dir, f"thermal_{file_prefix}")
            
            # Output file path
            output_file = os.path.join(output_dir, f"{file_prefix}_updated.pkl")
            
            pbar.set_postfix_str(file_prefix)
            
            # Check if thermal directory exists
            if not os.path.exists(thermal_dir):
                warning_count += 1
                cprint(f"  Warning: Thermal directory not found: {thermal_dir}", "yellow")
                pbar.update(1)
                continue
                
            try:
                # Load the original pickle file
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                
                # Update the thermal image paths
                if 'thermal_npaths' in data:
                    old_thermal_paths = data['thermal_npaths']
                    new_thermal_paths = []
                    
                    for old_path in old_thermal_paths:
                        # Extract the filename from the old path
                        filename = os.path.basename(old_path)
                        
                        # Construct the new path
                        new_path = os.path.join(thermal_dir, filename)
                        new_thermal_paths.append(new_path)
                    
                    # Replace old paths with new paths
                    data['thermal_npaths'] = new_thermal_paths
                    
                    # Save the modified data
                    with open(output_file, "wb") as f:
                        pickle.dump(data, f)
                    
                    # Verify a few paths exist
                    sample_size = min(5, len(new_thermal_paths))
                    existing = sum(1 for path in new_thermal_paths[:sample_size] if os.path.exists(path))
                    
                    if existing == 0 and sample_size > 0:
                        warning_count += 1
                        cprint(f"  Warning: None of the sampled thermal paths exist for {file_prefix}", "yellow")
                    else:
                        success_count += 1
                else:
                    error_count += 1
                    cprint(f"  Error: No 'thermal_npaths' key found in {file_prefix}", "red")
                    
            except Exception as e:
                error_count += 1
                cprint(f"  Error processing {base_filename}: {e}", "red")
            
            pbar.update(1)
    
    # Print final statistics with blinking effect if supported
    cprint("\nProcessing complete!", "green", attrs=["bold"])
    
    # Print statistics
    total_processed = success_count + warning_count + error_count
    cprint(f"\n--- Final Statistics ---", "cyan", attrs=["bold"])
    cprint(f"Total files processed: {total_processed}", "white")
    cprint(f"Successful: {success_count}", "green")
    cprint(f"With warnings: {warning_count}", "yellow")
    cprint(f"With errors: {error_count}", "red")
    
    # Create blinking effect for completion message (if in terminal)
    if sys.stdout.isatty():
        for _ in range(5):
            cprint("\rAll pickle files have been processed! ✓", "green", attrs=["bold", "blink"], end="")
            time.sleep(0.5)
            cprint("\r                                      ", end="")
            time.sleep(0.5)
        cprint("\rAll pickle files have been processed! ✓", "green", attrs=["bold"])
    else:
        cprint("All pickle files have been processed! ✓", "green", attrs=["bold"])

def main():
    # Configuration - update these paths to match your system
    source_dir = "/home/harshr/NV_cahsor/data/traversability/data"  # Directory with original pickle files
    thermal_base_dir = "/home/harshr/NV_cahsor/data/traversability/data"  # Base directory containing thermal folders
    output_dir = "/home/harshr/NV_cahsor/data/traversability/data/updated_pkl"  # Output directory
    
    cprint("=== Thermal Path Updater ===", "magenta", attrs=["bold"])
    cprint(f"Source directory: {source_dir}", "blue")
    cprint(f"Thermal base directory: {thermal_base_dir}", "blue")
    cprint(f"Output directory: {output_dir}", "blue")
    
    # Process all pickle files
    process_pickle_files(source_dir, thermal_base_dir, output_dir)

if __name__ == "__main__":
    main()
