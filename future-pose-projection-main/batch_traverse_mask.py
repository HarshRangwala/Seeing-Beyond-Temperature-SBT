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
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm  # Import tqdm for progress bar
from termcolor import cprint  # Import cprint for colored output

# --- Helper Functions ---

VIZ_IMAGE_SIZE = (1280, 1024)
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])
WHITE = np.array([1, 1, 1])  # Define white color

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
    Y_c = -normalized_coords[:, 1] * Z_c

    # Transform to world coordinates (account for camera offset)
    X_w = Z_c + camera_x_offset
    Y_w = X_c
    Z_w = np.zeros_like(X_w)  # The robot is on the ground (Z=0)

    world_coords = np.stack((X_w, Y_w, Z_w), axis=-1)
    return world_coords

def plot_footprints(ax, traj_pixels, robot_width_meters, robot_length_meters, camera_matrix, camera_height, camera_x_offset, width_scale, length_scale):
    """Plots robot footprints with perspective, correctly scaled, and with adjustable size (WHITE)."""

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

        # Create and plot the polygon (WHITE)
        polygon = Polygon([p1, p2, p3, p4], closed=True, edgecolor='white', facecolor='white', alpha=0.3)
        ax.add_patch(polygon)

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
    width_scale: float,
    length_scale: float,
    traj_color: np.ndarray = YELLOW,
):
    ax.imshow(img)
    # Plot footprints (now in WHITE), passing the scaling factors
    plot_footprints(ax, traj_pixels, robot_width_meters, robot_length_meters, camera_matrix, camera_height, camera_x_offset, width_scale, length_scale)

    # Highlight sampled points
    sampled_points = traj_pixels[sampled_indices]
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], color='red', s=50, zorder=5)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
    ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert a quaternion to a 3x3 rotation matrix."""
    rotation = R.from_quat([qx, qy, qz, qw])
    return rotation.as_matrix()

def transform_with_gravity_alignment(waypoints, robot_pose):
    """Transform waypoints to a gravity-aligned robot frame."""
    X = robot_pose['x']
    Y = robot_pose['y']
    Z = robot_pose.get('z', 0)
    qx, qy, qz, qw = robot_pose['qx'], robot_pose['qy'], robot_pose['qz'], robot_pose['qw']
    R_full = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    T_global_to_robot = np.eye(4)
    T_global_to_robot[:3, :3] = R_full
    T_global_to_robot[:3, 3] = [X, Y, Z]
    T_robot_to_global = np.linalg.inv(T_global_to_robot)
    transformed_wp = []
    for waypoint in waypoints:
        p_global = np.array([waypoint[0], waypoint[1], waypoint[2] if len(waypoint) > 2 else 0, 1])
        p_robot = T_robot_to_global @ p_global
        transformed_wp.append(p_robot[:2])
    return np.array(transformed_wp)

def yaw_from_quaternion(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def find_closest_imu_pose(timestamp, imu_poses):
    closest_pose = min(imu_poses, key=lambda p: abs(p['timestamp'] - timestamp))
    return closest_pose

def process_pickle_file(file_path, output_base_dir):
    """Processes a single pickle file and saves the results."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Extract filename for output directory
    filename = os.path.basename(file_path).split('.')[0]
    output_dir = os.path.join(output_base_dir, filename)
    os.makedirs(output_dir, exist_ok=True)

     # Camera parameters (These could also be loaded from the pickle if they vary)
    camera_height = 0.409 + 0.1
    camera_x_offset = 0.451
    fx, fy, cx, cy = 935.2355857804463, 935.7905325732659, 656.1572332633887, 513.7144019593092
    camera_matrix = gen_camera_matrix(fx, fy, cx, cy)
    dist_coeffs = np.array([-0.08194476107782814, -0.06592640858415261, -0.0007043163003212235, 0.002577256982584405])

    # Data
    original_odom_poses = data['odom_poses']
    thermal_images = data['thermal_npaths']
    thermal_ts_list = data['thermal_timestamps']
    imu_poses = data.get('imu_poses', [])  # Load IMU data, default to empty list if not present

    max_future_poses = 150
    n_samples = 3
    robot_width_meters = 0.6
    robot_length_meters = 1.0
    width_scale_factor = 3.0
    length_scale_factor = 0.3
    NEGATIVE_THRESHOLD = -10.0
    POSITIVE_THRESHOLD = 10.0
    RATE_OF_CHANGE_THRESHOLD = 5.0
    LOOKBACK_FRAMES = 5
    previous_pitch = 0.0

    for idx in tqdm(range(len(thermal_ts_list)), desc=f"Processing {filename}", unit="image"):
        thermal_ts = thermal_ts_list[idx]

        if imu_poses: # Only run the checks if IMU data is available
            closest_imu = find_closest_imu_pose(thermal_ts, imu_poses)
            current_pitch = closest_imu['pitch']
            pitch_change = abs(current_pitch - previous_pitch)
            previous_pitch = current_pitch
            skip_image = False
            negative_tilt_detected = False

            for i in range(max(0, idx - LOOKBACK_FRAMES), idx + 1):
                if i < len(thermal_ts_list):
                    past_imu = find_closest_imu_pose(thermal_ts_list[i], imu_poses)
                    if past_imu['pitch'] < NEGATIVE_THRESHOLD:
                        negative_tilt_detected = True
                        break

            if negative_tilt_detected:
                if current_pitch > POSITIVE_THRESHOLD or pitch_change > RATE_OF_CHANGE_THRESHOLD:
                    skip_image = True
            if skip_image:
                cprint(f"Skipping image {idx} in {filename} due to sky-pointing. Pitch: {current_pitch:.2f}, Change: {pitch_change:.2f}", "yellow")
                continue  # Skip to the next image

        future_odom = [p for p in original_odom_poses if p['timestamp'] >= thermal_ts]
        if not future_odom:
            cprint(f"No odom poses found at or after time {thermal_ts} in {filename}", "yellow")
            continue
        future_odom = future_odom[:max_future_poses]

        local_positions_dynamic = []
        for i, current_pose in enumerate(future_odom):
            positions_global = np.array([[p['x'], p['y']] for p in future_odom[i:]])
            positions_local = transform_with_gravity_alignment(positions_global, current_pose) #Using the new correct transform
            local_positions_dynamic.append(positions_local)

        if not local_positions_dynamic:
            continue
        positions_local = local_positions_dynamic[0]

        img_path = thermal_images[idx]
        img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_array is None:
            cprint(f"Error loading image {img_path} in {filename}", "red")
            continue

        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        traj_pixels = get_pos_pixels(
            positions_local, camera_height, camera_x_offset,
            camera_matrix, dist_coeffs, clip=True
        )
        sampled_indices = farthest_point_sampling(traj_pixels, n_samples)

        fig, ax = plt.subplots(figsize=(12, 9))
        plot_trajs_and_points_on_image(ax, img_array, traj_pixels, sampled_indices, robot_width_meters, robot_length_meters, camera_matrix, camera_height, camera_x_offset, width_scale_factor, length_scale_factor, traj_color=RED)
        output_path = os.path.join(output_dir, f'traj_footprint_{idx:04d}.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # if imu_poses: #Print pitch info only when IMU is available
        #     cprint(f"Processed image {idx} in {filename}, Pitch: {current_pitch:.2f} degrees", "green")
        # else:
        #     cprint(f"Processed image {idx} in {filename}", "green")


def main():
    input_dir = '/home/harshr/NV_cahsor/data/traversability/data/updated_pkl'
    output_base_dir = '/home/harshr/NV_cahsor/data/traversability/data/masks'
    os.makedirs(output_base_dir, exist_ok=True)

    pickle_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    if not pickle_files:
        cprint("No pickle files found in the input directory.", "red")
        return

    for pickle_file in pickle_files:
        file_path = os.path.join(input_dir, pickle_file)
        cprint(f"Starting processing of {pickle_file}", "cyan")
        process_pickle_file(file_path, output_base_dir)
        cprint(f"Finished processing of {pickle_file}", "green", attrs=["bold"])

    cprint("All pickle files processed!", "green", attrs=["blink", "bold"])

if __name__ == "__main__":
    main()