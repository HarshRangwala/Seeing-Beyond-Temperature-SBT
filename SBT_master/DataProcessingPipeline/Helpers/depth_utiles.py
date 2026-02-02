import cv2
import numpy as np
import torch

# --- Kernels for 'fill_in_fast' Algorithm ---
# These are required by the fill_in_fast function.

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# --- Original Helper Functions (from your scripts) ---

def preprocess_thermal(img):
    img = (img - img.mean()) / (img.std() + 1e-6)
    img = torch.clip(img, min=-3, max=3)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img

def undistort_image(camera_info, image):
    K = np.array(camera_info.K).reshape(3, 3)
    dist_coeffs = np.array(camera_info.D)
    undistorted = cv2.undistort(image, K, dist_coeffs)
    if len(undistorted.shape) == 2 or undistorted.shape[2] == 1:
        undistorted = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR)
    return undistorted, K, dist_coeffs

def project_points(points, K, dist_coeffs):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = (fx * points[:, 0] / points[:, 2]) + cx
    v = (fy * points[:, 1] / points[:, 2]) + cy
    return np.vstack((u, v)).T

def generate_depth_map(projections, points, image_shape):
    height, width = image_shape[:2]
    depth_map = np.zeros((height, width), dtype=np.float32)
    for (u, v), point in zip(projections, points):
        u_int, v_int = int(round(u)), int(round(v))
        if 0 <= u_int < width and 0 <= v_int < height:
            if depth_map[v_int, u_int] == 0 or point[2] < depth_map[v_int, u_int]:
                depth_map[v_int, u_int] = point[2]
    return depth_map

# --- Depth Completion Algorithm (from DenseLidar repo) ---

def fill_in_fast(depth_map, max_depth=30.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)
    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)
    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]
    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]
        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]
    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)
    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]
    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
    return depth_map

# --- Custom Hole Filling Logic (based on your original code) ---

def _fill_ground_holes_custom(depth_map, blind_spot_mask):
    """
    Fills holes in the lower portion of the depth map using a more aggressive
    method inspired by the user's original `process_depth_map` function,
    utilizing a distance transform for smooth, weighted filling.
    """
    processed_depth = depth_map.copy()
    valid_depths = processed_depth[processed_depth > 0]
    if len(valid_depths) == 0:
        return processed_depth
    min_d = np.percentile(valid_depths, 5)
    max_d = np.percentile(valid_depths, 95)
    robust_avg_depth = np.mean(valid_depths[(valid_depths >= min_d) & (valid_depths <= max_d)])
    dist_transform = cv2.distanceTransform(blind_spot_mask, cv2.DIST_L2, 3)
    max_dist = np.max(dist_transform)
    dist_transform_normalized = dist_transform / max_dist if max_dist > 0 else dist_transform
    height, width = processed_depth.shape
    for y in range(height):
        if not np.any(blind_spot_mask[y]):
            continue
        row_hole_mask = (blind_spot_mask[y] == 1)
        valid_pixels_in_row = processed_depth[y, ~row_hole_mask & (processed_depth[y] > 0)]
        fill_base_value = np.mean(valid_pixels_in_row) if len(valid_pixels_in_row) > 0 else robust_avg_depth
        dist_weights = dist_transform_normalized[y, row_hole_mask]
        final_fill_values = fill_base_value * ((1 - dist_weights) ** 2)
        processed_depth[y, row_hole_mask] = final_fill_values
    return processed_depth

# --- Main Processing Function ---

def create_dense_depth_map(sparse_depth_map):
    """
    Wrapper function that takes the sparse depth map, makes it dense,
    and intelligently fills holes and cleans up artifacts.
    """
    if sparse_depth_map is None or np.count_nonzero(sparse_depth_map) == 0:
        return np.full((185, 256), 30.0, dtype=np.float32)

    # Step 1: Run the main densification algorithm
    dense_depth = fill_in_fast(
        sparse_depth_map,
        max_depth=30.0,
        extrapolate=False,
        blur_type='bilateral'
    )

    # Step 2: Differentiated filling for Sky vs. Ground blind spot
    hole_mask = (dense_depth == 0).astype(np.uint8)
    horizon_line = int(dense_depth.shape[0] * 0.65)
    sky_mask = hole_mask.copy(); sky_mask[horizon_line:, :] = 0
    blind_spot_mask = hole_mask.copy(); blind_spot_mask[:horizon_line, :] = 0
    dense_depth[sky_mask == 1] = 30.0
    if np.any(blind_spot_mask):
        dense_depth = _fill_ground_holes_custom(dense_depth, blind_spot_mask)

    # --- Step 3: Final Cleanup and Refinement Stage ---

    # A) Fill any remaining small speckle holes that were missed.
    # We create a new mask of ALL leftover holes and use inpaint.
    final_hole_mask = (dense_depth == 0).astype(np.uint8)
    if np.any(final_hole_mask):
        dense_depth = cv2.inpaint(dense_depth, final_hole_mask, 3, cv2.INPAINT_NS)

    # B) Smooth out dark edge artifacts and blend filled regions.
    # A Median Blur is excellent for removing speckle noise and smoothing
    # artifacts while preserving the main sharp edges of objects.
    dense_depth = cv2.bilateralFilter(dense_depth, d=5, sigmaColor=2.0, sigmaSpace=2.0)

    return dense_depth

