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
import logging


# --- Helper Functions ---

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

# --- Constants ---
VIZ_IMAGE_SIZE = (1280, 1024) # Original size, will be resized later if needed based on thermal input
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])

# --- Robot & Camera Parameters ---
# Camera intrinsics (Ensure these match the thermal camera used)
# FX, FY, CX, CY = 935.2355857804463, 935.7905325732659, 656.1572332633887, 513.7144019593092 # Example values
# DIST_COEFFS = np.array([-0.08194476107782814, -0.06592640858415261, -0.0007043163003212235, 0.002577256982584405]) # Example values
CAMERA_HEIGHT = 0.409 + 0.1 # Z offset from robot base_link to camera_link
FX, FY, CX, CY = 935.2355857804463, 935.7905325732659, 656.1572332633887, 513.7144019593092
CAMERA_MATRIX = gen_camera_matrix(FX, FY, CX, CY)
DIST_COEFFS = np.array([-0.08194476107782814, -0.06592640858415261, -0.0007043163003212235, 0.002577256982584405])

CAMERA_X_OFFSET = 0.451    # X offset from robot base_link to camera_link

# --- Trajectory Processing Parameters ---
MAX_FUTURE_POSES = 30        # How many poses into the future to consider for trajectory
N_SAMPLES = 9                # Number of points to sample using Farthest Point Sampling for visualization
ROBOT_WIDTH_METERS = 0.6     # Robot width
ROBOT_LENGTH_METERS = 1.0    # Robot length (less relevant for footprint projection width)
WIDTH_SCALE_FACTOR = 3.0     # Visual scaling for footprint width in projection
LENGTH_SCALE_FACTOR = 0.3    # Visual scaling for footprint length (less impactful for width-based footprint)

# --- Filtering Parameters ---
NEGATIVE_PITCH_THRESHOLD = -10.0 # Pitch degrees below which indicates steep downward tilt
POSITIVE_PITCH_THRESHOLD = 10.0  # Pitch degrees above which might indicate sky pointing after NEGATIVE tilt
RATE_OF_CHANGE_THRESHOLD = 5.0   # Pitch change degrees between frames threshold
LOOKBACK_FRAMES = 5              # How many frames to look back for negative tilt check
MIN_TRAJECTORY_POINTS = 7        # Minimum number of valid projected points to generate mask/footprint
BORDER_PERCENTAGE = 0.05         # Percentage of image dimension considered 'border'
BORDER_CLUSTER_THRESHOLD = 0.7   # Percentage of points needed on border to trigger filtering



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

def create_footprint_mask(traj_pixels, robot_width_meters, robot_length_meters, camera_matrix, camera_height, camera_x_offset, width_scale, length_scale):
    """Creates a binary mask of the robot footprints."""
    mask = np.zeros((VIZ_IMAGE_SIZE[1], VIZ_IMAGE_SIZE[0]), dtype=np.uint8)  # Initialize black mask

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
        polygon_coords = np.array([p1, p2, p3, p4], dtype=np.int32)
        cv2.fillPoly(mask, [polygon_coords], color=255)

    return mask

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

def find_closest_pose_in_list(timestamp, poses_list):
    """Finds the pose with the timestamp closest to the target timestamp."""
    if not poses_list:
        return None
    closest_pose = min(poses_list, key=lambda p: abs(p['timestamp'] - timestamp))
    return closest_pose

def extract_poses_from_path(dlio_path_msg):
    """Extracts pose data from a nav_msgs/Path message."""
    poses_list = []
    if not dlio_path_msg or not dlio_path_msg.poses:
        return poses_list
    for pose_stamped in dlio_path_msg.poses:
        ts = pose_stamped.header.stamp.to_sec()
        pose = pose_stamped.pose
        poses_list.append({
            'timestamp': ts,
            'x': pose.position.x,
            'y': pose.position.y,
            'z': pose.position.z,
            'qx': pose.orientation.x,
            'qy': pose.orientation.y,
            'qz': pose.orientation.z,
            'qw': pose.orientation.w,
        })
    return poses_list

def calculate_pitch(qx, qy, qz, qw):
    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    pitch_rad = np.arctan2(t0, t1)
    pitch_deg = np.degrees(pitch_rad)
    return pitch_deg

def is_clustered_on_border(traj_pixels, border_percentage=0.05):
    """
    Checks if a significant percentage of trajectory pixels are clustered near the image borders.

    Args:
        traj_pixels (np.ndarray): Array of trajectory pixel coordinates (Nx2).
        border_percentage (float): Percentage of the image width/height to consider as the border.

    Returns:
        bool: True if the trajectory is considered clustered on the border, False otherwise.
    """
    border_width = VIZ_IMAGE_SIZE[0] * border_percentage
    border_height = VIZ_IMAGE_SIZE[1] * border_percentage

    # Check if points are within the border regions
    left_border = traj_pixels[:, 0] < border_width
    right_border = traj_pixels[:, 0] > (VIZ_IMAGE_SIZE[0] - border_width)
    top_border = traj_pixels[:, 1] < border_height
    bottom_border = traj_pixels[:, 1] > (VIZ_IMAGE_SIZE[1] - border_height)

    # Combine border conditions
    near_border = left_border | right_border | top_border | bottom_border

    # Calculate the percentage of points near the border
    percentage_on_border = np.sum(near_border) / len(traj_pixels)

    # Define a threshold for considering the trajectory as clustered on the border
    border_cluster_threshold = 0.7  # e.g., 70% of points on the border

    return percentage_on_border > border_cluster_threshold

def process_traversability_single(
        thermal_ts: float,                 # Timestamp of the current thermal image
        thermal_image_path: str,           # Path to the corresponding thermal image
        all_past_thermal_ts: list,         # List of all thermal timestamps up to current index (for lookback)
        all_poses_from_path: list,         # Camera distortion coefficients D
        output_footprints_dir: str,        # Directory to save footprint visualization images
        output_masks_dir: str,             # Directory to save binary mask images
        index: int,                        # Current index (for filename generation)
        logger=None                        # Optional logger (e.g., rospy)
        ):
    """
    Processes traversability (footprints, masks) for a single thermal image context.
    Uses cached poses list directly.
    Returns: (path_to_mask_image, path_to_footprint_image) or (None, None)
    """
    log_prefix = f"[Trav Index {index}]"
    def log_info(msg):
        if logger: logger.loginfo(f"{log_prefix} {msg}")
        else: print(f"INFO: {log_prefix} {msg}")
    def log_warn(msg):
        if logger: logger.logwarn(f"{log_prefix} {msg}")
        else: print(f"WARN: {log_prefix} {msg}")
    def log_err(msg):
        if logger: logger.logerr(f"{log_prefix} {msg}")
        else: print(f"ERROR: {log_prefix} {msg}")

    mask_output_path = None
    footprint_output_path = None

    try:
        # --- Basic Input Checks ---
        if not all_poses_from_path:
            log_warn("Poses list is empty. Cannot generate.")
            return None, None
        if thermal_image_path is None or not os.path.exists(thermal_image_path):
             log_warn(f"Thermal image path invalid or not found: {thermal_image_path}. Cannot generate.")
             return None, None
        if CAMERA_MATRIX is None or DIST_COEFFS is None:
            log_warn("Camera intrinsics (matrix/coeffs) not available. Cannot generate.")
            return None, None

        # --- Pitch Filtering (Sky Pointing Check) ---
        closest_pose_current = find_closest_pose_in_list(thermal_ts, all_poses_from_path)
        if closest_pose_current is None:
            log_warn(f"Could not find a close pose for thermal_ts {thermal_ts}. Skipping.")
            return None, None

        current_pitch = calculate_pitch(closest_pose_current['qx'], closest_pose_current['qy'], closest_pose_current['qz'], closest_pose_current['qw'])

        # --- Find Future Poses ---
        # Find poses at or after the *current* thermal timestamp from the *provided* list
        future_poses_struct = [p for p in all_poses_from_path if p['timestamp'] >= thermal_ts]
        if not future_poses_struct:
            log_warn(f"No future poses found at or after time {thermal_ts}. Skipping.")
            return None, None
        future_poses_struct = sorted(future_poses_struct, key=lambda p: p['timestamp']) # Ensure sorted
        future_poses_struct = future_poses_struct[:MAX_FUTURE_POSES]
        # --- 6DoF Transformation ---
        if not future_poses_struct: # Double check after slicing
             log_warn("No future poses left after slicing. Skipping.")
             return None, None

        first_future_pose = future_poses_struct[0]
        first_future_euler = quaternion_to_angle(Quaternion(first_future_pose['qx'], first_future_pose['qy'], first_future_pose['qz'], first_future_pose['qw']))
        first_future_6dof = np.array([first_future_pose['x'], first_future_pose['y'], first_future_pose['z'], first_future_euler[0], first_future_euler[1], first_future_euler[2]])
        future_6dof_poses = []

        for p in future_poses_struct:
            p_euler = quaternion_to_angle(Quaternion(p['qx'], p['qy'], p['qz'], p['qw']))
            future_6dof_poses.append([p['x'], p['y'], p['z'], p_euler[0], p_euler[1], p_euler[2]])
        future_6dof_poses = np.array(future_6dof_poses)
        first_future_6dof = np.tile(first_future_6dof, (len(future_6dof_poses), 1))
        positions_local = to_robot_numpy(first_future_6dof, future_6dof_poses)
        positions_local = positions_local[:, :3] # Extract only X, Y, Z

        positions_local = np.array([positions_local[i] for i in range(len(positions_local)) if i==0 or not np.allclose(positions_local[i-1], positions_local[i])])

        positions_local[:, 0] += CAMERA_X_OFFSET
        positions_local[:, 2] -= CAMERA_HEIGHT

        # --- Project to Image Plane ---
        try:
            img_array_bgr = cv2.imread(thermal_image_path) # Read as BGR by default
            if img_array_bgr is None:
                raise IOError(f"cv2.imread failed for {thermal_image_path}")

            # Resize image to the expected visualization size for plotting consistency
            img_display_rgb = cv2.cvtColor(cv2.resize(img_array_bgr, VIZ_IMAGE_SIZE), cv2.COLOR_BGR2RGB)

        except Exception as e:
            log_err(f"Error loading/processing thermal image {thermal_image_path}: {e}")
            return None, None
        
        traj_pixels = get_pos_pixels(
            positions_local, CAMERA_MATRIX, DIST_COEFFS , clip=True
        )

        # --- CHECK FOR VALID TRAJECTORY PIXELS HERE ---
        if len(traj_pixels) < MIN_TRAJECTORY_POINTS:
            return None, None

        # --- CHECK FOR CLUSTERED POINTS ON BORDER ---
        if is_clustered_on_border(traj_pixels):
            return None, None
        
        sampled_indices = farthest_point_sampling(traj_pixels, N_SAMPLES)

        # --- Create the footprint mask directly using cv2 ---
        mask = create_footprint_mask(traj_pixels, ROBOT_WIDTH_METERS, ROBOT_LENGTH_METERS, CAMERA_MATRIX, CAMERA_HEIGHT, CAMERA_X_OFFSET, WIDTH_SCALE_FACTOR, LENGTH_SCALE_FACTOR)
        mask_output_path = os.path.join(output_masks_dir, f'mask_{index:06d}.png')
        cv2.imwrite(mask_output_path, mask)

        # --- Generate Footprint Visualization ---
        fig, ax = plt.subplots(figsize=(12, 9))
        plot_trajs_and_points_on_image(ax, img_display_rgb, traj_pixels, sampled_indices, ROBOT_WIDTH_METERS, ROBOT_LENGTH_METERS, 
                                       CAMERA_MATRIX, CAMERA_HEIGHT, CAMERA_X_OFFSET, WIDTH_SCALE_FACTOR, LENGTH_SCALE_FACTOR,
                                         traj_color=RED)
        footprint_output_path = os.path.join(output_footprints_dir, f'traj_footprint_{index:04d}.png')
        plt.savefig(footprint_output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return mask_output_path, footprint_output_path
    except Exception as e:
        log_err(f"Unexpected error during traversability processing: {e}")
        import traceback
        log_err(traceback.format_exc())
        plt.close('all') # Ensure plots are closed on error
        return None, None





