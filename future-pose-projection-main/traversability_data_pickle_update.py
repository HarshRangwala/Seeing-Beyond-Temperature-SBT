import pickle
import os
import cv2
from termcolor import cprint

def update_pickle_with_masks(pickle_file_path, base_data_dir):
    """
    Adds mask paths to a pickle file, assuming thermal paths are correct.
    Checks readability of both thermal and mask image files.

    Args:
        pickle_file_path: Path to the pickle file.
        base_data_dir: Base directory for data.
    """
    try:
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        cprint(f"Error loading pickle file {pickle_file_path}: {e}", "red")
        return

    # Construct mask folder name
    pickle_file_name = os.path.basename(pickle_file_path)
    mask_folder_name = pickle_file_name.replace('.pkl', '_masks')
    mask_folder_path = os.path.join(base_data_dir, mask_folder_name)

    if not os.path.isdir(mask_folder_path):
        cprint(f"Error: Mask folder not found: {mask_folder_path}", "red")
        return

    # We assume 'thermal_npaths' already exists and is correct,
    # and that thermal images and masks have a 1:1 correspondence.
    if 'thermal_npaths' not in data:
        cprint(f"Error: 'thermal_npaths' key not found in pickle file: {pickle_file_path}", "red")
        return

    thermal_paths = data['thermal_npaths']
    mask_paths = []

    unreadable_files = []  # Keep track of unreadable files

    for i in range(len(thermal_paths)):  # Iterate by index
        # Correct Mask File Naming:
        mask_file_name = f"mask_{i:04d}.png"  # Directly use the index
        mask_path = os.path.join(mask_folder_path, mask_file_name)

        if os.path.exists(mask_path):
             # Check readability using cv2.imread (more robust than os.access)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) #or  cv2.IMREAD_GRAYSCALE
            if mask_img is None:  # cv2.imread returns None if it can't read
                unreadable_files.append(mask_path)
            else:
                mask_paths.append(mask_path)  # ONLY add if readable.
        else:
            cprint(f"Warning: Mask file not found: {mask_path}", "yellow")
            # Don't append if the file doesn't exist


    # Add mask paths to the dictionary
    data['mask_paths'] = mask_paths

    # Check readability of thermal images (now that we're checking masks, do thermal too)
    for thermal_path in thermal_paths:  # Check *all* thermal paths
        thermal_img = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
        if thermal_img is None:
            unreadable_files.append(thermal_path)


    # Save updated pickle file
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(data, f)

    cprint(f"Updated mask paths in: {pickle_file_path}", "green")

    if unreadable_files:
        cprint(f"Warning: The following files are unreadable:", "red")
        for file_path in unreadable_files:
            cprint(f"  - {file_path}", "red")
        cprint(f"Total unreadable files: {len(unreadable_files)}", "red")

    if len(data.get('thermal_npaths', [])) != len(data.get('mask_paths', [])):  # Use .get() for safety
       cprint(f"Warning: Mismatch in number of thermal ({len(data.get('thermal_npaths', []))}) and mask ({len(data.get('mask_paths', []))}) paths in {pickle_file_path}.", "yellow")

def main():
    base_data_dir = '/home/harshr/NV_cahsor/data/traversability'
    for root, _, files in os.walk(base_data_dir):
        for file in files:
            if file.endswith('.pkl'):
                pickle_path = os.path.join(root, file)
                update_pickle_with_masks(pickle_path, base_data_dir)


if __name__ == '__main__':
    main()