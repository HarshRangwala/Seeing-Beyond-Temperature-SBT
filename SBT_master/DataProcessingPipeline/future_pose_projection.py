import numpy as np
from PIL import Image
import torch
import cv2
import pickle
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.patches import Circle
from segment_anything import sam_model_registry, SamPredictor
from scipy.spatial.distance import cdist

# --- SAM Setup (Same as before) ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAM_CHECKPOINT = "/home/harshr/NV_cahsor/CAHSOR-master/TRON/checkpoint/SAM/sam_vit_h_4b8939.pth"  # Download from Meta
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# --- Helper Functions (Mostly same, with added FPS) ---

VIZ_IMAGE_SIZE = (1280, 1024)
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])

def numpy_to_img(arr: np.ndarray) -> Image:
    img = Image.fromarray(np.transpose(np.uint8(255 * arr), (1, 2, 0)))
    img = img.resize(VIZ_IMAGE_SIZE)
    return img

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()

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
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]
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
                if np.all(p > 0) and np.all(p < [VIZ_IMAGE_SIZE[0], VIZ_IMAGE_SIZE[1]])
            ]
        )
    return pixels

def farthest_point_sampling(points, n_samples):
    """
    Performs farthest point sampling on a set of points.
    Args:
        points: array of shape (N, D) representing N points in D dimensions.
        n_samples: number of points to sample.
    Returns:
        sampled_indices: indices of the sampled points.
    """
    n_points = points.shape[0]
    if n_samples >= n_points:
        return np.arange(n_points)  # Return all indices if n_samples is >= n_points

    sampled_indices = [np.random.randint(0, n_points)]  # Start with a random point
    distances = np.full(n_points, np.inf)

    for _ in range(1, n_samples):
        last_sampled = points[sampled_indices[-1]]
        new_distances = cdist(points, last_sampled.reshape(1, -1), metric="euclidean").squeeze()  # Distances to last sampled point
        distances = np.minimum(distances, new_distances)  # Update minimum distances
        next_sample = np.argmax(distances)  # Farthest point
        sampled_indices.append(next_sample)
        distances[next_sample] = 0 # set the distance 0 so that it won't be selected

    return np.array(sampled_indices)


def plot_trajs_and_points_on_image(
    ax: plt.Axes,
    img: np.ndarray,
    traj_pixels: np.ndarray,
    sampled_indices: np.ndarray,
    traj_color: np.ndarray = YELLOW,
):
    ax.imshow(img)
    # ax.plot(traj_pixels[:, 0], traj_pixels[:, 1], color=traj_color, lw=2.5, label='Trajectory')

    # Highlight sampled points
    sampled_points = traj_pixels[sampled_indices]
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], color='red', s=50) #, label='Sampled Points', zorder=5) #zorder make sure it plot on top

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
    ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))
    ax.legend()

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

def get_traversability_mask(image, traj_pixels, predictor, sampled_indices, n_samples =3):
    """
    Generate traversability mask using SAM and trajectory points, with FPS.
    """
    # --- Farthest Point Sampling ---
    # sampled_indices = farthest_point_sampling(traj_pixels, n_samples) #Now we are passing the sampled indices
    input_points = traj_pixels[sampled_indices]

    # --- Grayscale Conversion ---
    # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    predictor.set_image(image) #SAM takes grayscale image

    input_labels = np.ones(len(input_points))

    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True  # Get multiple mask options
    )

    best_mask_idx = np.argmax(scores) # Select the best mask
    traversability_mask = masks[best_mask_idx]
    return traversability_mask

# --- Main Processing Loop ---

# Load the pickle file
# file_path = '/home/harshr/NV_cahsor/CAHSOR-master/data/WC1_2024-08-27_19-57-34_chunk0000.pkl'
file_path ='/home/harshr/NV_cahsor/data/traversability/BL_2024-09-04_19-10-17_chunk0001.pkl'
with open(file_path, "rb") as f:
    data = pickle.load(f)

# Camera parameters
camera_height = 0.409 + 0.1
camera_x_offset = 0.451
fx, fy, cx, cy = 935.2355857804463, 935.7905325732659, 656.1572332633887, 513.7144019593092
camera_matrix = gen_camera_matrix(fx, fy, cx, cy)
dist_coeffs = np.array([-0.08194476107782814, -0.06592640858415261, -0.0007043163003212235, 0.002577256982584405])

output_dir = '/home/harshr/NV_cahsor/data/traversability/BL_2024-09-04_19-10-17_maskSAM'
os.makedirs(output_dir, exist_ok=True)

# Data
original_odom_poses = data['odom_poses']
thermal_images = data['thermal_npaths']
thermal_ts_list = data['thermal_timestamps']
odom_timestamps = [p['timestamp'] for p in original_odom_poses]
max_future_poses = 200
n_samples = 3 # Number of points to sample using FPS

# --- Main Loop ---
for idx in range(len(thermal_ts_list)):
    thermal_ts = thermal_ts_list[idx]

    # Find future odometry poses
    future_odom = [p for p in original_odom_poses if p['timestamp'] >= thermal_ts]
    if not future_odom:
        print(f"No odom poses found at or after time {thermal_ts}")
        continue
    future_odom = future_odom[:max_future_poses]


    # Calculate local trajectories
    local_positions_dynamic = []
    for i, current_pose in enumerate(future_odom):
        X0 = current_pose['x']
        Y0 = current_pose['y']
        PSI0 = yaw_from_quaternion(current_pose['qx'], current_pose['qy'],
                                 current_pose['qz'], current_pose['qw'])

        positions_global = np.array([[p['x'], p['y']] for p in future_odom[i:]])
        positions_local = transform_lg(positions_global, X0, Y0, PSI0)
        local_positions_dynamic.append(positions_local)
    if not local_positions_dynamic:
        continue

    # Get trajectory points (using first future pose as reference)
    positions_local = local_positions_dynamic[0]

    # Load and process image
    img_path = thermal_images[idx]
    img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_array is None:
        print(f"Error loading image {img_path}")
        continue

    # Convert image to RGB format (important for visualization)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
    else:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # Project trajectory points
    traj_pixels = get_pos_pixels(
        positions_local, camera_height, camera_x_offset,
        camera_matrix, dist_coeffs, clip=True
    )
    # --- FPS ---
    sampled_indices = farthest_point_sampling(traj_pixels, n_samples)

    # --- Get Traversability Mask (using FPS points and grayscale) ---
    traversability_mask = get_traversability_mask(img_array, traj_pixels, predictor, sampled_indices, n_samples)
    # --- Visualization ---
    overlay = img_array.copy()
    overlay[traversability_mask] = [255, 255, 255]  
    alpha = 0.5 # Increased the alpha for making SAM prediction more visible
    img_array = cv2.addWeighted(img_array, 1 - alpha, overlay, alpha, 0)

    fig, ax = plt.subplots(figsize=(12, 9))  # Adjusted figure size
    plot_trajs_and_points_on_image(ax, img_array, traj_pixels, sampled_indices, traj_color=RED)

    output_path = os.path.join(output_dir, f'traj_sam_{idx:04d}.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Processed image {idx}")

print(f"Saved images to {output_dir}!!")