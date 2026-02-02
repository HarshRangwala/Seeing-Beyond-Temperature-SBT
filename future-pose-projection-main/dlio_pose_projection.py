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
import pandas as pd
import tf.transformations
from geometry_msgs.msg import Quaternion

# --- Helper Functions --- (No changes here)
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
    xyz: np.ndarray,  # Takes xyz directly
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    rvec = tvec = (0, 0, 0)
    xyz_cv = np.stack([xyz[:, 1], -xyz[:, 2], xyz[:, 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv, rvec, tvec, camera_matrix, dist_coeffs
    )
    uv = uv.reshape(xyz.shape[0], 2)
    return uv

def get_pos_pixels(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    clip: bool = True,
):
    pixels = project_points(
        points, camera_matrix, dist_coeffs
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
    """Converts pixel coordinates to world coordinates."""
    K_inv = np.linalg.inv(camera_matrix)
    homogeneous_pixel_coords = np.hstack((pixel_coords, np.ones((pixel_coords.shape[0], 1))))
    normalized_coords = (K_inv @ homogeneous_pixel_coords.T).T
    Z_c = camera_height / normalized_coords[:, 1]
    X_c = normalized_coords[:, 0] * Z_c
    Y_c = -normalized_coords[:, 1] * Z_c
    X_w = Z_c + camera_x_offset
    Y_w = X_c
    Z_w = np.zeros_like(X_w)
    world_coords = np.stack((X_w, Y_w, Z_w), axis=-1)
    return world_coords

def plot_footprints(ax, traj_pixels, robot_width_meters, robot_length_meters, camera_matrix, camera_height, camera_x_offset, width_scale, length_scale):
    """Plots robot footprints with perspective."""
    for i in range(len(traj_pixels) - 1):
        world_coords_current = get_world_coordinates(traj_pixels[i:i+1], camera_matrix, camera_height, camera_x_offset)[0]
        world_coords_next = get_world_coordinates(traj_pixels[i+1:i+2], camera_matrix, camera_height, camera_x_offset)[0]
        dist_current = world_coords_current[0]
        dist_next = world_coords_next[0]
        width_current = (robot_width_meters * width_scale * camera_matrix[0,0] / dist_current)
        length_current = (robot_length_meters * length_scale * camera_matrix[0,0] / dist_current)
        width_next = (robot_width_meters * width_scale * camera_matrix[0,0] / dist_next)
        length_next = (robot_length_meters * length_scale * camera_matrix[0,0]/ dist_next)
        p1 = traj_pixels[i] + np.array([-width_current / 2, 0])
        p2 = traj_pixels[i] + np.array([width_current / 2, 0])
        p3 = traj_pixels[i+1] + np.array([width_next / 2, 0])
        p4 = traj_pixels[i+1] + np.array([-width_next / 2, 0])
        polygon = Polygon([p1, p2, p3, p4], closed=True, edgecolor='white', facecolor='white', alpha=0.3)
        ax.add_patch(polygon)

def farthest_point_sampling(points, n_samples):
    """Performs farthest point sampling."""
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
    plot_footprints(ax, traj_pixels, robot_width_meters, robot_length_meters, camera_matrix, camera_height, camera_x_offset, width_scale, length_scale)
    sampled_points = traj_pixels[sampled_indices]
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], color='red', s=50, zorder=5)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
    ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))

# --- 6DoF Transformation Functions ---
def angle_to_quaternion(angle):
    """Convert an angle in radians into a quaternion."""
    if not isinstance(angle, np.ndarray):
        angle = np.array(angle, dtype=np.float32)
    if angle.shape[-1] != 3:
        raise ValueError(f"Input must have last dim equal to 3 got {angle.shape[-1]}")
    if len(angle.shape) > 2:
        raise ValueError(f"Input tensor must be 1D or 2D got {len(angle.shape)}")
    if len(angle.shape) == 1:
        return Quaternion(*tf.transformations.quaternion_from_euler(angle[0], angle[1], angle[2], axes='sxyz'))
    if len(angle.shape) == 2:
        return [Quaternion(*tf.transformations.quaternion_from_euler(ang_[0], ang_[1], ang_[2], axes='sxyz')) for ang_ in angle]

def quaternion_to_angle(q):
    """Convert a quaternion into an angle in radians."""
    if not isinstance(q, Quaternion) and not isinstance(q, list):
        raise ValueError(f"Input must be of type Quaternion or list of Quaternions")
    if isinstance(q, list):
        return np.array([tf.transformations.euler_from_quaternion((quat.x, quat.y, quat.z, quat.w), axes='sxyz') for quat in q], dtype=np.float32).tolist()
    if isinstance(q, Quaternion):
        return list(tf.transformations.euler_from_quaternion((q.x,q.y,q.z,q.w), axes='sxyz'))

def euler_to_rotation_matrix(euler_angles):
    """ Convert Euler angles to a rotation matrix (NumPy). """
    cos = np.cos(euler_angles)
    sin = np.sin(euler_angles)
    zero = np.zeros_like(euler_angles[:, 0])
    one = np.ones_like(euler_angles[:, 0])
    R_x = np.stack([one, zero, zero, zero, cos[:, 0], -sin[:, 0], zero, sin[:, 0], cos[:, 0]], axis=1).reshape(-1, 3, 3)
    R_y = np.stack([cos[:, 1], zero, sin[:, 1], zero, one, zero, -sin[:, 1], zero, cos[:, 1]], axis=1).reshape(-1, 3, 3)
    R_z = np.stack([cos[:, 2], -sin[:, 2], zero, sin[:, 2], cos[:, 2], zero, zero, zero, one], axis=1).reshape(-1, 3, 3)
    return np.matmul(np.matmul(R_z, R_y), R_x)

def extract_euler_angles_from_se3_batch(tf3_matx):
    """Extract Euler angles from SE(3) matrices (NumPy)."""
    if tf3_matx.shape[1:] != (4, 4):
        raise ValueError("Input tensor must have shape (batch, 4, 4)")
    rotation_matrices = tf3_matx[:, :3, :3]
    batch_size = tf3_matx.shape[0]
    euler_angles = np.zeros((batch_size, 3), dtype=tf3_matx.dtype)
    euler_angles[:, 0] = np.arctan2(rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2])  # Roll
    euler_angles[:, 1] = np.arctan2(-rotation_matrices[:, 2, 0], np.sqrt(rotation_matrices[:, 2, 1] ** 2 + rotation_matrices[:, 2, 2] ** 2))  # Pitch
    euler_angles[:, 2] = np.arctan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])  # Yaw
    return euler_angles

def to_robot_numpy(Robot_frame, P_relative):
    """Transforms points to robot frame using NumPy (6DoF)."""
    if not isinstance(Robot_frame, np.ndarray):
        Robot_frame = np.array(Robot_frame, dtype=np.float32)
    if not isinstance(P_relative, np.ndarray):
        P_relative = np.array(P_relative, dtype=np.float32)
    if len(Robot_frame.shape) == 1:
        Robot_frame = Robot_frame.reshape(1, -1)
    if len(P_relative.shape) == 1:
        P_relative = P_relative.reshape(1, -1)
    if len(Robot_frame.shape) > 2 or len(P_relative.shape) > 2:
        raise ValueError(f"Input must be 1D or 2D, got {Robot_frame.shape} and {P_relative.shape}")
    if Robot_frame.shape[-1] != 6 or P_relative.shape[-1] != 6:
        raise ValueError("Input Robot_frame/P_relative must have last dimension equal to 6")
    if Robot_frame.shape[0] != P_relative.shape[0]:
        raise ValueError("Input tensors must have the same batch size")

    batch_size = Robot_frame.shape[0]
    ones = np.ones_like(P_relative[:, 0])
    T1 = np.zeros((batch_size, 4, 4), dtype=Robot_frame.dtype)
    T2 = np.zeros((batch_size, 4, 4), dtype=P_relative.dtype)
    T1[:, :3, :3] = euler_to_rotation_matrix(Robot_frame[:, 3:])
    T2[:, :3, :3] = euler_to_rotation_matrix(P_relative[:, 3:])
    T1[:, :3, 3] = Robot_frame[:, :3]
    T2[:, :3, 3] = P_relative[:, :3]
    T1[:, 3, 3] = 1.0
    T2[:, 3, 3] = 1.0
    T1_inv = np.linalg.inv(T1)
    tf3_mat = np.matmul(T2, T1_inv)
    transform = np.zeros_like(Robot_frame)
    transform[:, :3] = np.matmul(T1_inv, np.concatenate((P_relative[:, :3], ones.reshape(-1, 1)), axis=1).reshape(batch_size, 4, 1)).squeeze(axis=2)[:, :3]
    transform[:, 3:] = extract_euler_angles_from_se3_batch(tf3_mat)
    return transform

def yaw_from_quaternion(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

# --- Main Processing Loop ---

# Load data and parameters
file_path ='/home/harshr/NV_cahsor/data/traversability/WC1_2024-08-27_19-57-34_chunk0000.pkl'
poses_file = '/home/harshr/NV_cahsor/data/west_campus_data/1/pose_WC1_2024-08-27_19-57-34_chunk0000.csv'
with open(file_path, "rb") as f:
    data = pickle.load(f)

camera_height = 0.409 + 0.1
camera_x_offset = 0.451
fx, fy, cx, cy = 935.2355857804463, 935.7905325732659, 656.1572332633887, 513.7144019593092
camera_matrix = gen_camera_matrix(fx, fy, cx, cy)
dist_coeffs = np.array([-0.08194476107782814, -0.06592640858415261, -0.0007043163003212235, 0.002577256982584405])

output_dir = '/home/harshr/NV_cahsor/data/traversability/trav_masks/WC1_2024-08-27_19-57-34_chunk0000_footprintT39'
os.makedirs(output_dir, exist_ok=True)

def load_poses(file_path):
    df = pd.read_csv(file_path)
    poses = []
    for index, row in df.iterrows():
        poses.append({
            'timestamp': row['timestamp'],
            'x': row['x'],
            'y': row['y'],
            'z': row['z'],
            'qx': row['qx'],
            'qy': row['qy'],
            'qz': row['qz'],
            'qw': row['qw']
        })
    return poses

thermal_images = data['thermal_npaths']
thermal_ts_list = data['thermal_timestamps']
all_poses = load_poses(poses_file)

max_future_poses = 35
n_samples = 9
robot_width_meters = 0.6
robot_length_meters = 1.0
width_scale_factor = 3.0
length_scale_factor = 0.3
NEGATIVE_THRESHOLD = -10.0
POSITIVE_THRESHOLD = 10.0
RATE_OF_CHANGE_THRESHOLD = 5.0
LOOKBACK_FRAMES = 5

def find_closest_pose(timestamp, poses):
    closest_pose = min(poses, key=lambda p: abs(p['timestamp'] - timestamp))
    return closest_pose

def calculate_pitch(qx, qy, qz, qw):
    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    pitch_rad = np.arctan2(t0, t1)
    pitch_deg = np.degrees(pitch_rad)
    return pitch_deg

previous_pitch = 0.0

for idx in range(len(thermal_ts_list)):
    thermal_ts = thermal_ts_list[idx]
    closest_pose = find_closest_pose(thermal_ts, all_poses)
    current_pitch = calculate_pitch(closest_pose['qx'], closest_pose['qy'], closest_pose['qz'], closest_pose['qw'])
    pitch_change = abs(current_pitch - previous_pitch)
    previous_pitch = current_pitch
    skip_image = False

    negative_tilt_detected = False
    for i in range(max(0, idx - LOOKBACK_FRAMES), idx + 1):
        if i < len(thermal_ts_list):
            past_pose = find_closest_pose(thermal_ts_list[i], all_poses)
            past_pitch = calculate_pitch(past_pose['qx'], past_pose['qy'], past_pose['qz'], past_pose['qw'])
            if past_pitch < NEGATIVE_THRESHOLD:
                negative_tilt_detected = True
                break
    if negative_tilt_detected:
        if current_pitch > POSITIVE_THRESHOLD or pitch_change > RATE_OF_CHANGE_THRESHOLD:
            skip_image = True
    if skip_image:
        print(f"Skipping image {idx} due to sky-pointing.")
        continue

    future_poses = [p for p in all_poses if p['timestamp'] >= thermal_ts]
    if not future_poses:
        print(f"No poses found at or after time {thermal_ts}")
        continue
    future_poses = future_poses[:max_future_poses]

    # --- CORRECTED 6DoF Transformation ---
    # Get the *first* future pose (this is our reference)
    first_future_pose = future_poses[0]
    first_future_euler = quaternion_to_angle(Quaternion(first_future_pose['qx'], first_future_pose['qy'], first_future_pose['qz'], first_future_pose['qw']))
    first_future_6dof = np.array([first_future_pose['x'], first_future_pose['y'], first_future_pose['z'], first_future_euler[0], first_future_euler[1], first_future_euler[2]])
    

    # Get *all* future poses as 6DoF
    future_6dof_poses = []
    for p in future_poses:
        p_euler = quaternion_to_angle(Quaternion(p['qx'], p['qy'], p['qz'], p['qw']))
        future_6dof_poses.append([p['x'], p['y'], p['z'], p_euler[0], p_euler[1], p_euler[2]])
    future_6dof_poses = np.array(future_6dof_poses)
    first_future_6dof = np.tile(first_future_6dof, (len(future_6dof_poses), 1))
    print(first_future_6dof.shape, future_6dof_poses.shape)
    # Calculate the relative positions.  ALL future poses relative to the FIRST future pose.
    positions_local = to_robot_numpy(first_future_6dof, future_6dof_poses)
    positions_local = positions_local[:, :3]  # Keep only x, y, z

    
    # Remove consecutive duplicates
    positions_local = np.array([positions_local[i] for i in range(len(positions_local)) if i==0 or not np.allclose(positions_local[i-1], positions_local[i])])

    # --- Camera Offset and Height ---
    positions_local[:, 0] += camera_x_offset
    positions_local[:, 2] -= camera_height

    # --- Image Loading and Processing --- (No changes here)
    img_path = thermal_images[idx]
    img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_array is None:
        print(f"Error loading image {img_path}")
        continue
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
    else:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    traj_pixels = get_pos_pixels(
        positions_local, camera_matrix, dist_coeffs, clip=True
    )
    sampled_indices = farthest_point_sampling(traj_pixels, n_samples)

    # --- Visualization --- (No changes here)
    fig, ax = plt.subplots(figsize=(12, 9))
    plot_trajs_and_points_on_image(ax, img_array, traj_pixels, sampled_indices, robot_width_meters, robot_length_meters, camera_matrix, camera_height, camera_x_offset, width_scale_factor, length_scale_factor, traj_color=RED)
    output_path = os.path.join(output_dir, f'traj_footprint_{idx:04d}.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Processed image {idx}, Pitch: {current_pitch:.2f} degrees")

print(f"Saved images to {output_dir}!!")