import numpy as np
from PIL import Image
import torch
import cv2
import pickle
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.patches import Polygon
from scipy.spatial.distance import cdist

# --- No SAM Needed ---

# --- Helper Functions (Modified for footprint drawing) ---

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

def get_world_coordinates(pixel_coords, camera_matrix, camera_height, camera_x_offset):
    """Converts pixel coordinates to world coordinates (inverse of projection)."""
    # Intrinsic matrix inverse
    K_inv = np.linalg.inv(camera_matrix)

    # Convert pixel coordinates to normalized image coordinates
    homogeneous_pixel_coords = np.hstack((pixel_coords, np.ones((pixel_coords.shape[0], 1))))
    normalized_coords = (K_inv @ homogeneous_pixel_coords.T).T

    # Calculate Z (depth) in the camera frame.  We know the camera height.
    Z_c = camera_height / normalized_coords[:, 1]

    # Calculate X and Y in the camera frame
    X_c = normalized_coords[:, 0] * Z_c
    Y_c = -normalized_coords[:,1] * Z_c #Y_c is already calculated

    # Transform to world coordinates (account for camera offset)
    X_w = Z_c + camera_x_offset
    Y_w = X_c
    Z_w = np.zeros_like(X_w)  # The robot is on the ground (Z=0)

    world_coords = np.stack((X_w, Y_w, Z_w), axis=-1)
    return world_coords

def plot_footprints(ax, traj_pixels, robot_width_meters, robot_length_meters, camera_matrix, camera_height, camera_x_offset, width_scale, length_scale):
    """Plots robot footprints with perspective, correctly scaled, and with adjustable size."""

    for i in range(len(traj_pixels) - 1):
        # Get world coordinates for the current and next points
        world_coords_current = get_world_coordinates(traj_pixels[i:i+1], camera_matrix, camera_height, camera_x_offset)[0]
        world_coords_next = get_world_coordinates(traj_pixels[i+1:i+2], camera_matrix, camera_height, camera_x_offset)[0]

        # Calculate the distance (depth) for scaling
        dist_current = world_coords_current[0]
        dist_next = world_coords_next[0]

        # Scale width and length based on depth AND apply the additional scaling factors.
        width_current = (robot_width_meters * width_scale * camera_matrix[0,0] / dist_current)
        length_current = (robot_length_meters * length_scale * camera_matrix[0,0] / dist_current)
        width_next = (robot_width_meters * width_scale * camera_matrix[0,0] / dist_next)
        length_next = (robot_length_meters * length_scale * camera_matrix[0,0]/ dist_next)


        # Define footprint corners in *pixel* coordinates.
        p1 = traj_pixels[i] + np.array([-width_current / 2, 0])
        p2 = traj_pixels[i] + np.array([width_current / 2, 0])
        p3 = traj_pixels[i+1] + np.array([width_next / 2, 0])
        p4 = traj_pixels[i+1] + np.array([-width_next / 2, 0])

        # Create and plot the polygon
        polygon = Polygon([p1, p2, p3, p4], closed=True, edgecolor='blue', facecolor='blue', alpha=0.3)
        ax.add_patch(polygon)
        # Removed the out-of-bounds check. It's not strictly necessary and
        # matplotlib handles clipping automatically.  This simplifies the code.


def plot_trajs_and_points_on_image(
    ax: plt.Axes,
    img: np.ndarray,
    traj_pixels: np.ndarray,
    sampled_indices: np.ndarray,
    robot_width_meters: float,
    robot_length_meters: float,
    camera_matrix: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    width_scale: float,  # Added width scale
    length_scale: float, # Added length scale
    traj_color: np.ndarray = YELLOW,
):
    ax.imshow(img)
    # Plot footprints, passing the scaling factors
    plot_footprints(ax, traj_pixels, robot_width_meters, robot_length_meters, camera_matrix, camera_height, camera_x_offset, width_scale, length_scale)

    # Highlight sampled points
    sampled_points = traj_pixels[sampled_indices]
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], color='red', s=50, zorder=5)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
    ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))



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


# --- Main Processing Loop ---

# Load the pickle file
file_path ='/home/harsh/Downloads/bags/WC1_2024-08-27_19-57-34_chunk0000_updated.pkl' 
with open(file_path, "rb") as f:
    data = pickle.load(f)

# Camera parameters
camera_height = 0.409 + 0.1
camera_x_offset = 0.451
fx, fy, cx, cy = 935.2355857804463, 935.7905325732659, 656.1572332633887, 513.7144019593092
camera_matrix = gen_camera_matrix(fx, fy, cx, cy)
dist_coeffs = np.array([-0.08194476107782814, -0.06592640858415261, -0.0007043163003212235, 0.002577256982584405])

output_dir = '/home/harsh/Downloads/bags/WC1_2024-08-27-19-57-34_footprintT6'
os.makedirs(output_dir, exist_ok=True)

# Data
original_odom_poses = data['odom_poses']
thermal_images = data['thermal_npaths']
thermal_ts_list = data['thermal_timestamps']
odom_timestamps = [p['timestamp'] for p in original_odom_poses]
max_future_poses = 250
n_samples = 3  # For visualization
# Robot dimensions
robot_width_meters = 0.6
robot_length_meters = 1.0

# --- SCALING FACTORS ---
width_scale_factor = 3.0  # Increase to make footprints wider
length_scale_factor = 0 # Increase to make footprints longer


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

    # Convert image to RGB
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
    # FPS for visualization
    sampled_indices = farthest_point_sampling(traj_pixels, n_samples)

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(12, 9))
    # Pass the scaling factors to the plotting function.
    plot_trajs_and_points_on_image(ax, img_array, traj_pixels, sampled_indices, robot_width_meters, robot_length_meters, camera_matrix, camera_height, camera_x_offset, width_scale_factor, length_scale_factor, traj_color=RED)

    output_path = os.path.join(output_dir, f'traj_footprint_{idx:04d}.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Processed image {idx}")

print(f"Saved images to {output_dir}!!")