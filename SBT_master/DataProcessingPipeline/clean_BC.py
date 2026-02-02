import pickle
import numpy as np
from pathlib import Path
import os

def clean_pickle_file(input_path: Path, output_path: Path):
    """
    Loads a pickle file, synchronizes data lengths based on 'sm_cmd_vel',
    and saves the cleaned data to the output path.

    Specifically, it truncates 'thermal_paths' and any other list/array
    that has the same initial length as 'thermal_paths' to match the
    length of 'sm_cmd_vel', if 'thermal_paths' is longer.
    """
    print(f"Processing: {input_path.name}")
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)

        # --- Check for essential keys ---
        if 'thermal_paths' not in data:
            print(f"  WARNING: 'thermal_paths' key missing. Skipping file.")
            return False
        if 'sm_cmd_vel' not in data:
            print(f"  WARNING: 'sm_cmd_vel' key missing. Skipping file.")
            return False

        # --- Get lengths ---
        len_thermal = len(data['thermal_paths'])
        len_sm_cmd_vel = len(data['sm_cmd_vel'])

        print(f"  Initial lengths: thermal_paths={len_thermal}, sm_cmd_vel={len_sm_cmd_vel}")

        # --- Check if cleaning is needed ---
        if len_thermal > len_sm_cmd_vel:
            print(f"  Mismatch detected. Truncating lists/arrays to length {len_sm_cmd_vel}.")
            target_len = len_sm_cmd_vel
            cleaned_data = {}
            modified = False

            for key, value in data.items():
                # Check if the item is a list or numpy array AND if its length
                # matches the *original* length of thermal_paths.
                # This assumes all these lists/arrays should be synchronized.
                if isinstance(value, (list, np.ndarray)) and len(value) == len_thermal:
                    # Truncate the list/array
                    cleaned_data[key] = value[:target_len]
                    if key == 'thermal_paths': # Mark modified only if thermal_paths was actually truncated
                         modified = True
                else:
                    # Keep other data types or lists/arrays of different lengths as is
                    cleaned_data[key] = value

            # Verify final lengths (optional but recommended)
            final_len_thermal = len(cleaned_data['thermal_paths'])
            final_len_cmd = len(cleaned_data['sm_cmd_vel'])
            print(f"  Final lengths: thermal_paths={final_len_thermal}, sm_cmd_vel={final_len_cmd}")

            if final_len_thermal != final_len_cmd:
                 print(f"  ERROR: Final lengths still mismatch after cleaning! Check logic.")
                 # Decide how to handle this error - maybe skip saving?
                 return False # Indicate failure

            # --- Save the cleaned data ---
            with open(output_path, 'wb') as f:
                pickle.dump(cleaned_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Saved cleaned data to: {output_path}")
            return True # Indicate modification happened

        elif len_thermal < len_sm_cmd_vel:
            print(f"  WARNING: 'thermal_paths' ({len_thermal}) is shorter than 'sm_cmd_vel' ({len_sm_cmd_vel}). "
                  f"This script only handles cases where thermal is longer. File not modified.")
            # Optionally copy the original file if you want all files in the output dir
            # import shutil
            # shutil.copyfile(input_path, output_path)
            # print(f"  Copied original file to: {output_path}")
            return False # Indicate no modification needed based on the primary goal

        else:
            # Lengths match, no cleaning needed for this condition
            print("  Lengths match. No modification needed for this file.")
            # Optionally copy the original file
            # import shutil
            # shutil.copyfile(input_path, output_path)
            # print(f"  Copied original file to: {output_path}")
            return False # Indicate no modification needed

    except FileNotFoundError:
        print(f"  ERROR: File not found: {input_path}")
        return False
    except Exception as e:
        print(f"  ERROR: Failed to process file {input_path.name}: {e}")
        return False


# --- Configuration ---
INPUT_DIR = Path("/mnt/sbackup/Server_3/harshr/m2p2_data/validation")
OUTPUT_DIR = Path("/mnt/sbackup/Server_3/harshr/m2p2_data/validation_dt4") # Choose a DIFFERENT directory!
FILE_PATTERN = "*_processed.pkl"

# --- Main Execution ---
if __name__ == "__main__":
    if INPUT_DIR == OUTPUT_DIR:
        raise ValueError("Input and Output directories must be different to avoid overwriting original data!")

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Find all pickle files matching the pattern
    pickle_files = sorted(list(INPUT_DIR.glob(FILE_PATTERN)))

    if not pickle_files:
        print(f"No files found matching '{FILE_PATTERN}' in {INPUT_DIR}")
    else:
        print(f"Found {len(pickle_files)} files to process.")

    num_modified = 0
    num_failed = 0
    for input_file_path in pickle_files:
        output_file_path = OUTPUT_DIR / input_file_path.name
        modified = clean_pickle_file(input_file_path, output_file_path)
        if modified:
            num_modified += 1
        # Add more sophisticated error tracking if needed based on return value

    print("\n--- Processing Summary ---")
    print(f"Total files processed: {len(pickle_files)}")
    print(f"Files modified and saved: {num_modified}")
    # Add counts for skipped/failed files if implemented

    print("Done.")