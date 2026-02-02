#!/usr/bin/env python3
import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2, CompressedImage, CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import message_filters
import ros_numpy
from geometry_msgs.msg import TransformStamped

class LidarCameraAligner:
    def __init__(self):
        rospy.init_node('lidar_camera_aligner')

        # Parameters
        self.lidar_topic = '/sensor_suite/ouster/points'
        self.image_topic = '/sensor_suite/lwir/lwir/image_raw/compressed'
        self.camera_info_topic = '/sensor_suite/lwir/lwir/camera_info'
        self.depth_map_topic = '/aligned/depth_map'
        self.dense_depth_map_topic = '/aligned/depth_map_dense'  # New Topic for Dense Depth
        self.overlay_image_topic = '/aligned/overlay_image'
        self.publish_rate = 10  # Hz
        self.grid_size = 3  # Grid size for dense mapping

        # Subscribers
        self.lidar_sub = message_filters.Subscriber(self.lidar_topic, PointCloud2)
        self.image_sub = message_filters.Subscriber(self.image_topic, CompressedImage)
        self.info_sub = message_filters.Subscriber(self.camera_info_topic, CameraInfo)

        # Time Synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.lidar_sub, self.image_sub, self.info_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.callback)

        # Publishers
        self.depth_pub = rospy.Publisher(self.depth_map_topic, Image, queue_size=1)
        self.dense_depth_pub = rospy.Publisher(self.dense_depth_map_topic, Image, queue_size=1)  # New Publisher
        self.overlay_pub = rospy.Publisher(self.overlay_image_topic, Image, queue_size=1)

        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # CV Bridge
        self.bridge = CvBridge()

        rospy.loginfo("LidarCameraAligner node initialized.")

    def callback(self, lidar_msg, image_msg, info_msg):
        try:
            # Wait for the transform from os_lidar to ir_camera_optical
            transform = self.tf_buffer.lookup_transform(
                'ir_camera_optical',  # Target frame
                'os_sensor',          # Source frame
                lidar_msg.header.stamp,
                rospy.Duration(1.0)
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Transform not available: %s", e)
            return
        
        

        # Convert PointCloud2 to numpy array
        try:
            lidar_points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar_msg)
            print(len(lidar_points))
            rospy.loginfo(f"Received {len(lidar_points)} LIDAR points.")
        except Exception as e:
            rospy.logwarn("PointCloud2 conversion failed: %s", e)
            return

        # Transform LIDAR points to camera frame
        try:
            points_cam = self.transform_points(lidar_points, transform)
            rospy.loginfo("Transformed LIDAR points to camera frame.")
        except Exception as e:
            rospy.logwarn("Transforming points failed: %s", e)
            return

        points_cam = points_cam[points_cam[:, 2] > 0]

        # Decompress the image
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            rospy.loginfo(f"Decompressed image shape: {image.shape}")
        except CvBridgeError as e:
            rospy.logwarn("Image decompression failed: %s", e)
            return

        # Undistort the image
        undistorted_image, K, dist_coeffs = self.undistort_image(info_msg, image)
        rospy.loginfo(f"Undistorted image shape: {undistorted_image.shape}")

        # Project points onto image plane
        projections = self.project_points(points_cam, K, dist_coeffs)
        rospy.loginfo(f"Projected {len(projections)} LIDAR points onto image plane.")

        # Generate sparse depth map
        depth_map = self.generate_depth_map(projections, points_cam, undistorted_image.shape)
        rospy.loginfo("Generated sparse depth map.")

        # Convert depth map to a normalized color image for visualization
        depth_colored = self.convert_depth_to_color(depth_map)
        try:
            depth_msg = self.bridge.cv2_to_imgmsg(depth_colored, encoding="bgr8")
            depth_msg.header = image_msg.header
            self.depth_pub.publish(depth_msg)
            rospy.loginfo("Published colored depth map.")
        except CvBridgeError as e:
            rospy.logwarn("Depth map conversion failed: %s", e)

        # Convert sparse depth map to dense depth map using dense_map function
        dense_depth_map = self.create_dense_depth_map(depth_map)
        rospy.loginfo("Generated dense depth map.")

        try:
            # Convert to ROS Image message with 32FC1 encoding
            dense_depth_msg = self.bridge.cv2_to_imgmsg(dense_depth_map, encoding="32FC1")
            dense_depth_msg.header = image_msg.header
            self.dense_depth_pub.publish(dense_depth_msg)
            rospy.loginfo("Published dense depth map.")
        except CvBridgeError as e:
            rospy.logwarn("Dense depth map conversion failed: %s", e)

        # Overlay LIDAR points on the undistorted image
        color = (0, 255, 0)
        overlay_image = self.overlay_points(undistorted_image, projections, color)
        rospy.loginfo(f"Overlay image shape: {overlay_image.shape}")

        # Ensure the overlay_image has three channels
        if len(overlay_image.shape) == 2 or overlay_image.shape[2] == 1:
            overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2BGR)
            rospy.loginfo("Converted overlay image to BGR format.")

        # Convert overlay image to Image message
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay_image, encoding="bgr8")
            overlay_msg.header = image_msg.header
            self.overlay_pub.publish(overlay_msg)
            rospy.loginfo("Published overlay image.")
        except CvBridgeError as e:
            rospy.logwarn("Overlay image conversion failed: %s", e)

    def transform_points(self, points, transform: TransformStamped):
        """Transforms LIDAR points from os_lidar frame to ir_camera_optical frame."""
        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ])
        rotation = self.quaternion_to_rotation_matrix(transform.transform.rotation)

        # Apply rotation and translation
        points_transformed = (rotation @ points.T).T + translation
        return points_transformed

    def quaternion_to_rotation_matrix(self, q):
        """Converts a geometry_msgs/Quaternion to a rotation matrix."""
        x, y, z, w = q.x, q.y, q.z, q.w
        # Normalize the quaternion
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        x /= norm
        y /= norm
        z /= norm
        w /= norm

        # Compute rotation matrix elements
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),       2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w),   1 - 2*(x**2 + y**2)]
        ])
        return R

    def undistort_image(self, camera_info, image):
        """Undistorts the image using camera calibration parameters."""
        K = np.array(camera_info.K).reshape(3, 3)
        dist_coeffs = np.array(camera_info.D)
        undistorted = cv2.undistort(image, K, dist_coeffs)
        rospy.loginfo(f"Undistorted image shape after undistort: {undistorted.shape}")

        # Ensure the undistorted image has three channels
        if len(undistorted.shape) == 2 or undistorted.shape[2] == 1:
            undistorted = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR)
            rospy.loginfo("Converted undistorted image to BGR format.")

        return undistorted, K, dist_coeffs

    def project_points(self, points, K, dist_coeffs):
        """Projects 3D points onto the 2D image plane."""
        # Extract intrinsic parameters
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Project points
        u = (fx * points[:, 0] / points[:, 2]) + cx
        v = (fy * points[:, 1] / points[:, 2]) + cy
        projections = np.vstack((u, v)).T
        return projections

    def generate_depth_map(self, projections, points, image_shape):
        """Generates a sparse depth map from projected points."""
        height, width = image_shape[:2]
        depth_map = np.zeros((height, width), dtype=np.float32)

        for (u, v), point in zip(projections, points):
            u_int, v_int = int(round(u)), int(round(v))
            if 0 <= u_int < width and 0 <= v_int < height:
                # Update depth map with the closest point
                if depth_map[v_int, u_int] == 0 or point[2] < depth_map[v_int, u_int]:
                    depth_map[v_int, u_int] = point[2]

        return depth_map

    def convert_depth_to_color(self, depth_map):
        """Converts depth map to a color image using a colormap."""
        # Handle zero depth values by setting them to the maximum depth
        depth_map_visual = np.copy(depth_map)
        depth_map_visual[depth_map_visual == 0] = np.max(depth_map_visual)

        # Normalize to 0-255 for visualization
        depth_normalized = cv2.normalize(depth_map_visual, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)

        # Apply colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        return depth_colored

    def create_dense_depth_map(self, depth_map):
        """Generates a dense depth map from a sparse depth map using the dense_map function."""
        m, n = depth_map.shape
        grid = self.grid_size

        # Extract non-zero points from the sparse depth map
        y_indices, x_indices = np.nonzero(depth_map)
        depths = depth_map[y_indices, x_indices]
        Pts = np.vstack((x_indices, y_indices, depths)).T  # Shape: (num_points, 3)

        rospy.loginfo(f"Number of non-zero depth points: {Pts.shape[0]}")

        # Generate dense depth map using the dense_map function
        dense_depth = self.dense_map(Pts, n, m, grid, post_process=True)

        return dense_depth

    def overlay_points(self, image, projections, color):
        """Overlays LIDAR points onto the image for visualization."""
        overlay_image = image.copy()
        for u, v in projections:
            u_int, v_int = int(round(u)), int(round(v))
            if 0 <= u_int < overlay_image.shape[1] and 0 <= v_int < overlay_image.shape[0]:
                cv2.circle(overlay_image, (u_int, v_int), 1, color, -1)
        return overlay_image

    def run(self):
        rospy.spin()



    def dense_map(self, Pts, n, m, grid, post_process=True):
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
        ng = 2 * grid + 1

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
if __name__ == '__main__':
    try:
        aligner = LidarCameraAligner()
        aligner.run()
    except rospy.ROSInterruptException:
        pass
