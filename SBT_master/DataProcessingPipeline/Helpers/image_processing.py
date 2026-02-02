import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import torch

PATCH_SIZE = 256
PATCH_EPSILON = 0.7 * PATCH_SIZE * PATCH_SIZE
ACTUATION_LATENCY = 0.15
HOMOGRAPHY_MATRIX = np.array([[ 2.10928225e-01, -1.13894127e+00,  4.44216119e+02],
       [-1.26548983e-16, -1.84371290e+00,  1.29715449e+03],
       [-9.85515129e-20, -1.76433885e-03,  1.00000000e+00]]) 
GRID_SIZE = 4

def preprocess_thermal(img):
    img = (img - img.mean()) / (img.std() + 1e-6)
    img = torch.clip(img, min=-3, max = 3)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img


def undistort_image(camera_info, image):
    #print('undistorting the image')
    K = np.array(camera_info.K).reshape(3, 3)
    dist_coeffs = np.array(camera_info.D)
    undistorted = cv2.undistort(image, K, dist_coeffs)
    # Ensure the image has three channels
    if len(undistorted.shape) == 2 or undistorted.shape[2] == 1:
        undistorted = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR)
    return undistorted, K, dist_coeffs

def project_points(points, K, dist_coeffs):
    """Projects 3D points onto the 2D image plane."""
    #print('projecting the 3D points onto the 2D image')
    # Extract intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Project points by taking the focal lengths and principal point coordinates
    u = (fx * points[:, 0] / points[:, 2]) + cx
    v = (fy * points[:, 1] / points[:, 2]) + cy
    projections = np.vstack((u, v)).T
    return projections

def generate_depth_map(projections, points, image_shape):
    #print('Generating the depth map')
    height, width = image_shape[:2]
    depth_map = np.zeros((height, width), dtype=np.float32)
    for (u, v), point in zip(projections, points):
        u_int, v_int = int(round(u)), int(round(v))
        if 0 <= u_int < width and 0 <= v_int < height:
            if depth_map[v_int, u_int] == 0 or point[2] < depth_map[v_int, u_int]:
                depth_map[v_int, u_int] = point[2]

    # depth_map[depth_map>15]= np.inf  
    return depth_map


def convert_depth_to_color(depth_map):
    #print('Converting depth to color')
    depth_map_visual = np.copy(depth_map)
    depth_map_visual[depth_map_visual == 0] = np.max(depth_map_visual)
    depth_normalized = cv2.normalize(depth_map_visual, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    return depth_colored

def create_dense_depth_map(depth_map):
    m, n = depth_map.shape
    
    # Extract non-zero points from the sparse depth map
    y_indices, x_indices = np.nonzero(depth_map)
    depths = depth_map[y_indices, x_indices]
    Pts = np.vstack((x_indices, y_indices, depths)).T  # Shape: (num_points, 3)

    #print(f"Number of non-zero depth points: {Pts.shape[0]}")

    # Generate dense depth map using the dense_map function
    dense_depth = dense_map(Pts, n, m, GRID_SIZE, post_process=True)

    return dense_depth

def overlay_points(image, projections, color):
        """Overlays LIDAR points onto the image for visualization."""
        overlay_image = image.copy()
        for u, v in projections:
            u_int, v_int = int(round(u)), int(round(v))
            if 0 <= u_int < overlay_image.shape[1] and 0 <= v_int < overlay_image.shape[0]:
                cv2.circle(overlay_image, (u_int, v_int), 1, color, -1)
        return overlay_image
'''
def dense_map(Pts, n, m, grid, post_process=True):
    """
        Generates a dense depth map from sparse points using a weighted average within a grid,
        with optional post-processing.

        Args:
            Pts (numpy.ndarray): Array of points (x, y, depth).
            n (int): Width of the image.
            m (int): Height of the image.
            grid (int): Grid size for interpolation.
            post_process (bool): Whether to apply post-processing steps.

        Returns:
            numpy.ndarray: Dense depth map of shape (m, n).
        """
    ng = 2 * GRID_SIZE + 1

    # Initialize intermediate matrices
    mX = np.full((m, n), np.inf, dtype=np.float32)
    mY = np.full((m, n), np.inf, dtype=np.float32)
    mD = np.zeros((m, n), dtype=np.float32)

    # Populate mX, mY, mD with fractional parts and depth
    # Ensure that indices are within image boundaries
    valid_indices = (Pts[:,0] >=0) & (Pts[:,0] < n) & (Pts[:,1] >=0) & (Pts[:,1] < m)
    Pts = Pts[valid_indices]
    x_int = np.int32(Pts[:,0])
    y_int = np.int32(Pts[:,1])
    mX[y_int, x_int] = Pts[:,0] - np.round(Pts[:,0])
    mY[y_int, x_int] = Pts[:,1] - np.round(Pts[:,1])
    mD[y_int, x_int] = Pts[:,2]

    # Initialize KmX, KmY, KmD
    KmX = np.zeros((ng, ng, m - ng, n - ng), dtype=np.float32)
    KmY = np.zeros((ng, ng, m - ng, n - ng), dtype=np.float32)
    KmD = np.zeros((ng, ng, m - ng, n - ng), dtype=np.float32)

    # Populate KmX, KmY, KmD
    for i in range(ng):
        for j in range(ng):
            KmX[i, j] = mX[i:(m - ng + i), j:(n - ng + j)] - grid - 1 + i
            KmY[i, j] = mY[i:(m - ng + i), j:(n - ng + j)] - grid - 1 + j
            KmD[i, j] = mD[i:(m - ng + i), j:(n - ng + j)]

    # Initialize S and Y
    S = np.zeros_like(KmD[0, 0], dtype=np.float32)
    Y = np.zeros_like(KmD[0, 0], dtype=np.float32)

    # Weighted average based on distance
    for i in range(ng):
        for j in range(ng):
            distance = KmX[i, j] ** 2 + KmY[i, j] ** 2
            distance[distance == 0] = 1e-6  # Prevent division by zero
            s = 1 / np.sqrt(distance)
            Y += s * KmD[i, j]
            S += s

    # Prevent division by zero
    S[S == 0] = 1.0

    # Compute the dense depth values
    interpolated_depth = Y / S

    # Initialize the output depth map
    out = np.zeros((m, n), dtype=np.float32)

    # Assign interpolated depths to the valid region
    out[grid:m - grid -1, grid:n - grid -1] = interpolated_depth

    if post_process:
        # Apply post-processing steps
                # Step 1: Inpainting to fill holes
        mask = (out == 0).astype(np.uint8)
        depth_map_inpainted = cv2.inpaint(out, mask, 3, cv2.INPAINT_TELEA)

        # Step 2: Bilateral filtering to smooth while preserving edges
        depth_map_bilateral = cv2.bilateralFilter(depth_map_inpainted, d=3, sigmaColor=10, sigmaSpace=20)

        # Step 3: Median filtering to reduce noise and fill small gaps
        depth_map_median = cv2.medianBlur(depth_map_bilateral.astype(np.float32), 3)
        return depth_map_median
    
    return out
'''

def distance_weighting(distance):
    sigma = 1.0
    weights = np.exp(-distance / (2*sigma**2))
    weights[distance == 0] = 1.0
    return weights

def process_depth_map(depth_map):
    # Create masks for valid and empty regions
    valid_mask = (depth_map > 0).astype(np.uint8)
    
    # Get valid depth values and their range
    valid_depths = depth_map[depth_map > 0]
    if len(valid_depths) == 0:
        return depth_map
    
    # Use percentiles to avoid outliers
    min_depth = np.percentile(valid_depths, 5)
    max_depth = np.percentile(valid_depths, 95)
    
    # Normalize the valid depth values to reduce contrast
    processed_depth = depth_map.copy()
    processed_depth = np.clip(processed_depth, min_depth, max_depth)
    
    # Scale the depth values to use more of the available range
    # processed_depth = ((processed_depth - min_depth) / (max_depth - min_depth) * 255).astype(np.float32)
    
    # Fill empty areas with a less extreme value (e.g., 75% of max)
    empty_mask = (valid_mask == 0)
    # processed_depth[empty_mask] = 150  # Use a gray value instead of pure white
    valid_depths = processed_depth[valid_mask > 0]
    avg_depth = np.mean(valid_depths)
    
    # Create distance transform for empty regions
    dist_transform = cv2.distanceTransform(1 - valid_mask, cv2.DIST_L2, 5)
    
    # Normalize distance transform
    dist_transform = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
    height = depth_map.shape[0]
    # Define split point - only process below this point
    split_point = int(height * 0.7)  

    # Use distance transform to create smooth transitions
    for y in range(height):
        if y < split_point:
            # Keep sky untouched (255 for maximum depth)
            processed_depth[y][empty_mask[y]] = 200  # 200 for slight grey 
            
        else:
            empty_rows = empty_mask[y]
            # if np.any(empty_rows):
            #     # Adjust weights to make interpolation darker
            #     bottom_weight = (y - split_point) / (height - split_point)
            #     # Reduce this multiplier (e.g., 200 instead of 255) for darker values
            #     value = 50 * (1 - bottom_weight) + avg_depth * bottom_weight * 0.1 # 0.7 darkens the interpolation
            #     processed_depth[y][empty_rows] = value * (1 - dist_transform[y][empty_rows])
            if np.any(empty_rows):
                # Get local context
                row_valid = processed_depth[y][~empty_rows]
                if len(row_valid) > 0:
                    local_avg = np.mean(row_valid)
                else:
                    local_avg = avg_depth

                # Use local context for interpolation
                bottom_weight = (y - split_point) / (height - split_point)
                
                # Stronger weight for local context
                value = local_avg * 0.8  # Use 80% of local average
                
                # Apply distance transform with stronger effect
                dist_weight = dist_transform[y][empty_rows]
                processed_depth[y][empty_rows] = value * (1 - dist_weight * 1.5) 
    
    # Apply bilateral filter to smooth transitions while preserving edges
    # processed_depth = cv2.medianBlur(processed_depth.astype(np.uint8), 5)
    # smoothed = cv2.bilateralFilter(processed_depth, d=9, sigmaColor=50, sigmaSpace=50)
    
    # Fill small holes using morphological operations
    kernel = np.ones((7,7), np.uint8)
    dilated_mask = cv2.dilate(valid_mask, kernel, iterations=1)
    holes_mask = dilated_mask - valid_mask
    
    # Inpaint only the small holes
    final_depth = cv2.inpaint(
        processed_depth.astype(np.float32),
        holes_mask,
        inpaintRadius=9,
        flags=cv2.INPAINT_TELEA
    )
    
    # Final smoothing to blend transitions
    #final_depth = cv2.bilateralFilter(final_depth, d=7, sigmaColor=25, sigmaSpace=25)
    
    return final_depth

def fill_depth_holes(depth_map):
    # Create binary mask for holes
    hole_mask = (depth_map == 255).astype(np.uint8)
    
    # Get the average depth value of the valid regions near the bottom
    bottom_region = depth_map[int(0.4*depth_map.shape[0]):, :]
    valid_depths = bottom_region[bottom_region < 240]
    
    if len(valid_depths) > 0:
        fill_value = np.median(valid_depths)
    else:
        fill_value = 150  # Default gray value
    
    # Fill holes with gradual transition
    kernel = np.ones((5,5), np.uint8)
    dilated_mask = cv2.dilate(hole_mask, kernel, iterations=2)
    
    # Create gradient for smooth transition
    gradient_mask = cv2.GaussianBlur(dilated_mask.astype(float), (21, 21), 5)
    
    # Actually use the gradient mask for blending
    result = depth_map.copy().astype(float)
    blend = fill_value * gradient_mask + result * (1 - gradient_mask)
    result = cv2.GaussianBlur(blend, (7, 7), 1.5)
    
    return result.astype(np.uint8)




def dense_map(Pts, n, m, grid, post_process=True):
    """
    Generates a dense depth map from sparse points using a weighted average within a grid,
    with optional post-processing.

    Args:
        Pts (numpy.ndarray): Array of points (x, y, depth).
        n (int): Width of the image.
        m (int): Height of the image.
        grid (int): Grid size for interpolation.
        post_process (bool): Whether to apply post-processing steps.

    Returns:
        numpy.ndarray: Dense depth map of shape (m, n).
    """
    Pts = np.array(Pts, dtype=np.float32)

    ng = 2 * grid + 1  # Assuming grid size is passed as 'grid'

    # Initialize intermediate matrices
    mX = np.full((m, n), np.inf, dtype=np.float32)
    mY = np.full((m, n), np.inf, dtype=np.float32)
    mD = np.zeros((m, n), dtype=np.float32)

    # Populate mX, mY, mD with fractional parts and depth
    # Ensure that indices are within image boundaries
    valid_indices = (Pts[:,0] >=0) & (Pts[:,0] < n) & (Pts[:,1] >=0) & (Pts[:,1] < m)
    Pts = Pts[valid_indices]

    # Range limitation: only include points within 20 meters
    # max_distance = 10.0  # Maximum allowed distance in meters
    # valid_distance = Pts[:,2] <= max_distance
    # Pts = Pts[valid_distance]

    x_int = np.int32(Pts[:,0])
    y_int = np.int32(Pts[:,1])

    # Set fractional parts
    mX[y_int, x_int] = Pts[:,0] - np.round(Pts[:,0])
    mY[y_int, x_int] = Pts[:,1] - np.round(Pts[:,1])

    # Set depth, points beyond max_distance are set to infinity
    mD[y_int, x_int] = Pts[:,2]  # Already filtered, or alternatively:
    # mD[y_int, x_int] = np.where(Pts[:,2] <= max_distance, Pts[:,2], np.inf)

    # Initialize KmX, KmY, KmD
    KmX = np.zeros((ng, ng, m - ng, n - ng), dtype=np.float32)
    KmY = np.zeros((ng, ng, m - ng, n - ng), dtype=np.float32)
    KmD = np.zeros((ng, ng, m - ng, n - ng), dtype=np.float32)

    # Populate KmX, KmY, KmD
    for i in range(ng):
        for j in range(ng):
            KmX[i, j] = mX[i:(m - ng + i), j:(n - ng + j)] - grid - 1 + i
            KmY[i, j] = mY[i:(m - ng + i), j:(n - ng + j)] - grid - 1 + j
            KmD[i, j] = mD[i:(m - ng + i), j:(n - ng + j)]

    # Initialize S and Y
    S = np.zeros_like(KmD[0, 0], dtype=np.float32)
    Y = np.zeros_like(KmD[0, 0], dtype=np.float32)

    # Weighted average based on distance
    for i in range(ng):
        for j in range(ng):
            distance = KmX[i, j] ** 2 + KmY[i, j] ** 2
            # distance[distance == 0] = 1e-6  # Prevent division by zero
            # s = 1 / np.sqrt(distance)
            s = distance_weighting(distance)
            Y += s * KmD[i, j]
            S += s

    # Prevent division by zero
    S[S == 0] = 1.0

    # Compute the dense depth values
    interpolated_depth = Y / S

    # Initialize the output depth map
    out = np.zeros((m, n), dtype=np.float32)

    # Assign interpolated depths to the valid region
    out[grid:m - grid -1, grid:n - grid -1] = interpolated_depth

    # if post_process:
    #     # Apply post-processing steps
    #     # Step 1: Inpainting to fill holes
    #     mask = (out == 0).astype(np.uint8)
    #     depth_map_inpainted = cv2.inpaint(out, mask, 3, cv2.INPAINT_TELEA)         

    #     # Step 2: Bilateral filtering to smooth while preserving edges
    #     depth_map_bilateral = cv2.bilateralFilter(depth_map_inpainted, d=9, sigmaColor=75, sigmaSpace=75)

    #     # Step 3: Median filtering to reduce noise and fill small gaps
    #     depth_map_median = cv2.medianBlur(depth_map_bilateral.astype(np.float32), 1)
    #     return depth_map_median
    if post_process:
        # Enhanced post-processing pipeline
        # Fill holes
        mask = (out < 0.1).astype(np.uint8)
        depth_map_inpainted = cv2.inpaint(out, mask, 5, cv2.INPAINT_NS) # Use cv2.INPAINT_NS better and faster than cv2.INPAINT_TELEA
        inpaint_depth = process_depth_map(out)
        #filled_depth = fill_depth_holes(inpaint_depth)
        
        # Edge-preserving smoothing
        depth_map_bilateral = cv2.bilateralFilter(depth_map_inpainted, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Additional smoothing while preserving edges
        depth_map_bilateral = cv2.bilateralFilter(depth_map_bilateral, d=5, sigmaColor=0.5, sigmaSpace=5.0)
        
        # Final median blur for noise reduction
        # depth_map_median = cv2.medianBlur(depth_map_bilateral.astype(np.float32), 3) # Remove this
        
        # Normalize depth values
        # depth_normalized = cv2.normalize(depth_map_median, None, 0, 255, cv2.NORM_MINMAX) # Remove this
        
        return depth_map_bilateral # depth_normalized.astype(np.uint8) # return this depth_map_bilateral

    return out

def enhance_depth_image(depth_img):
    """Apply histogram equalization to depth image"""
    if depth_img.dtype != np.uint8:
        depth_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return cv2.equalizeHist(depth_img)

def refine_sky_mask(thermal_cv, depth_map):
    # 1. Temperature thresholding
    # Assuming 'thermal_cv' is your thermal image (grayscale)
    threshold = 150  # Adjust this threshold based on your data, in this case since sky is brighter we are using this
    thermal_sky_mask = (thermal_cv > threshold)

    # 2. Initial sky mask based on depth
    depth_sky_mask = (depth_map == 0)

    # 3. Combine masks
    initial_sky_mask = depth_sky_mask & thermal_sky_mask

    # 4. Region growing (using dilation)
    kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed
    refined_sky_mask = cv2.dilate(initial_sky_mask.astype(np.uint8), kernel, iterations=2)

    return refined_sky_mask.astype(bool)