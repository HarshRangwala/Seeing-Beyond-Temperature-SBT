import numpy as np
from PIL import Image
import torch
import cv2
import pickle
import os
import sys
from segment_anything import sam_model_registry, SamPredictor
from scipy.spatial.distance import cdist
from tqdm import tqdm  # Import tqdm for progress bars
from termcolor import cprint  # Import cprint for colored output


# --- SAM Setup ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAM_CHECKPOINT = "/home/harshr/NV_cahsor/CAHSOR-master/TRON/checkpoint/SAM/sam_vit_h_4b8939.pth"  # Your SAM checkpoint
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# --- Helper Functions ---

VIZ_IMAGE_SIZE = (1280, 1024)  # Or whatever your visualization size is

def gen_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

def project_points(
    xy: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    horizon, _ = xy.shape
    xyz = np.concatenate(
        [xy, -camera_height * np.ones((horizon, 1))], axis=-1
    )
    rvec = tvec = (0, 0, 0)
    xyz[:, 0] += camera_x_offset
    xyz_cv = np.stack([xyz[:, 1], -xyz[:, 2], xyz[:, 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv, rvec, tvec, camera_matrix, dist_coeffs
    )
    uv = uv.reshape(horizon, 2)
    return uv

def get_pos_pixels(
    points: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    clip: bool = True,
):
    pixels = project_points(
        points, camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]  # Adjust for your image size
    if clip:
        pixels = np.array(
            [
                [
                    np.clip(p[0], 0, VIZ_IMAGE_SIZE[0]),
                    np.clip(p[1], 0, VIZ_IMAGE_SIZE[1]),
                ]
                for p in pixels
            ]
        )
    else:
        pixels = np.array(
            [
                p
                for p in pixels
                if np.all(p > 0) and np.all(p < [VIZ_IMAGE_SIZE[0], VIZ_IMAGE_SIZE[1]])  #Adjust for the image size
            ]
        )
    return pixels

def farthest_point_sampling(points, n_samples):
    n_points = points.shape[0]
    if n_samples >= n_points:
        return np.arange(n_points)
    sampled_indices = [np.random.randint(0, n_points)]
    distances = np.full(n_points, np.inf)
    for _ in range(1, n_samples):
        last_sampled = points[sampled_indices[-1]]
        new_distances = cdist(points, last_sampled.reshape(1, -1), metric="euclidean").squeeze()
        distances = np.minimum(distances, new_distances)
        next_sample = np.argmax(distances)
        sampled_indices.append(next_sample)
        distances[next_sample] = 0
    return np.array(sampled_indices)

def transform_lg(wp, X, Y, PSI):
    R_r2i = np.array([
        [np.cos(PSI), -np.sin(PSI), X],
        [np.sin(PSI), np.cos(PSI), Y],
        [0, 0, 1]
    ])
    R_i2r = np.linalg.inv(R_r2i)
    transformed_wp = []
    for waypoint in wp:
        pi = np.array([[waypoint[0]], [waypoint[1]], [1]])
        pr = np.matmul(R_i2r, pi)
        lg = [pr[0, 0], pr[1, 0]]
        transformed_wp.append(lg)
    return np.array(transformed_wp)

def yaw_from_quaternion(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def get_traversability_mask(image, traj_pixels, predictor, sampled_indices, n_samples = 3):
    input_points = traj_pixels[sampled_indices]
    predictor.set_image(image)
    input_labels = np.ones(len(input_points))
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )
    best_mask_idx = np.argmax(scores)
    return masks[best_mask_idx]

def process_pickle_file(pickle_file_path, output_base_dir, camera_params, max_future_poses=200, n_samples=3):
    """Processes a single pickle file and saves the masks."""
    try:
        with open(pickle_file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        cprint(f"Error loading pickle file {pickle_file_path}: {e}", "red")
        return

    # Extract data from pickle
    original_odom_poses = data['odom_poses']
    thermal_images = data['thermal_npaths']
    thermal_ts_list = data['thermal_timestamps']

    # Create output directory for this pickle file
    pickle_file_name = os.path.basename(pickle_file_path)
    output_dir_masks = os.path.join(output_base_dir, pickle_file_name.replace('.pkl', '_masks'))  # Consistent naming
    os.makedirs(output_dir_masks, exist_ok=True)

    cprint(f"Processing: {pickle_file_name}", "yellow")

    # Loop through thermal images with progress bar
    for idx in tqdm(range(len(thermal_ts_list)), desc="Generating Masks", unit="image"):
        thermal_ts = thermal_ts_list[idx]

        # Find future odometry poses
        future_odom = [p for p in original_odom_poses if p['timestamp'] >= thermal_ts]
        if not future_odom:
            cprint(f"No odom poses found at or after time {thermal_ts} in {pickle_file_name}", "yellow")
            continue  # Skip to the next thermal image
        future_odom = future_odom[:max_future_poses]


        # Calculate local trajectories
        local_positions_dynamic = []
        for i, current_pose in enumerate(future_odom):
            X0, Y0 = current_pose['x'], current_pose['y']
            PSI0 = yaw_from_quaternion(current_pose['qx'], current_pose['qy'], current_pose['qz'], current_pose['qw'])
            positions_global = np.array([[p['x'], p['y']] for p in future_odom[i:]])
            positions_local = transform_lg(positions_global, X0, Y0, PSI0)
            local_positions_dynamic.append(positions_local)
        if not local_positions_dynamic:
            continue

        positions_local = local_positions_dynamic[0]

        # Load corresponding thermal image
        img_path = thermal_images[idx]
        try:
            img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img_array is None:
                raise IOError(f"Could not open or read image file: {img_path}")
        except Exception as e:
            cprint(f"Error loading image {img_path}: {e}", "red")
            continue  # Skip to next thermal image

        # Convert to RGB (required for SAM)
        if len(img_array.shape) == 2:
            img_array_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
        else:
            img_array_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        # Project points and sample
        traj_pixels = get_pos_pixels(positions_local, camera_params['height'], camera_params['x_offset'],
                                     camera_params['matrix'], camera_params['dist_coeffs'], clip=True)
        sampled_indices = farthest_point_sampling(traj_pixels, n_samples)

        # Generate and save mask
        traversability_mask = get_traversability_mask(img_array_rgb, traj_pixels, predictor, sampled_indices, n_samples)
        mask_filename = f'mask_{idx:04d}.png'
        mask_output_path = os.path.join(output_dir_masks, mask_filename)
        cv2.imwrite(mask_output_path, traversability_mask.astype(np.uint8) * 255)

    cprint(f"Finished processing: {pickle_file_name}, masks saved to: {output_dir_masks}", "green")


def main():
    # --- Configuration ---
    data_dir = '/home/harshr/NV_cahsor/data/traversability'  # Directory containing your pickle files and thermal images
    output_base_dir = '/home/harshr/NV_cahsor/data/traversability/masks_output'  # Main output directory
    os.makedirs(output_base_dir, exist_ok=True)  # Create main output directory

    # Camera parameters (replace with your actual values)
    camera_params = {
        'height': 0.409 + 0.1,
        'x_offset': 0.451,
        'fx': 935.2355857804463,
        'fy': 935.7905325732659,
        'cx': 656.1572332633887,
        'cy': 513.7144019593092,
        'matrix': None,  # Will be computed below
        'dist_coeffs': np.array([-0.08194476107782814, -0.06592640858415261, -0.0007043163003212235, 0.002577256982584405])
    }
    camera_params['matrix'] = gen_camera_matrix(camera_params['fx'], camera_params['fy'], camera_params['cx'], camera_params['cy'])


    # --- Find and Process Pickle Files ---
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.pkl'):
                pickle_file_path = os.path.join(root, file)
                process_pickle_file(pickle_file_path, output_base_dir, camera_params)


if __name__ == "__main__":
    main()