import io
from collections import deque

from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
import rospy
from sensor_msgs.msg import CompressedImage, Image as RImage, CameraInfo, PointCloud2, PointField, RegionOfInterest
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from std_msgs.msg import Header
import open3d as o3d

from depth_costmap import load_models

FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8

class DepthPredictor:
    def __init__(self, encoder: str = 'vitb', input_size: int = 256, max_depth: int = 18) -> None:
        """Reads thermal images and predicts and publishes depth data"""
        self.input_size = input_size
        self.crop_top = 40
        self.crop_bottom = 225

        self.focal_length_x = 2715.79693 # Example value, should be updated by CameraInfo if used for point cloud
        self.focal_length_y = 2715.79693 # Example value

        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = DEVICE

        self.transform = transforms.Resize((256, 256), antialias=True)
        self.model = load_models(encoder_ckpt_path="/home/harshr/sbt_depth_experiments/DepthAnything/weights/crt_th_lidar/ssl-ptr-thermal_lidar_video_training_latest-2048-05-07-01-15_500.pth",
                decoder_ckpt_path="/home/harshr/sbt_depth_experiments/DepthAnything/weights/crt_decoder_thlr/depth_decoder_epoch_100.pth",
                latent_size=2048,
                num_layers= 50,
                device=DEVICE
                ).eval()

        self._cv_bridge = CvBridge()

        self.depth_pub = rospy.Publisher("/sensor_suite/lwir/lwir/depth/image_raw", RImage, queue_size=1)
        self.cam_info_pub = rospy.Publisher("/sensor_suite/lwir/lwir/depth/camera_info", CameraInfo, queue_size=1) # Publisher for CameraInfo
        # self.cam_info_pub = rospy.Publisher("/sensor_suite/lwir/lwir/camera_info", CameraInfo, queue_size=1)
        # Subscriber for thermal images
        self.thermal_sub = rospy.Subscriber("/sensor_suite/lwir/lwir/image_raw/compressed", CompressedImage, self.read_thermal)

        # Create the hardcoded CameraInfo message
        self.hardcoded_cam_info = CameraInfo()
        self.hardcoded_cam_info.header = Header()
        self.hardcoded_cam_info.header.frame_id = "ir_camera" # Hardcoded frame_id
        self.hardcoded_cam_info.height = 1024
        self.hardcoded_cam_info.width = 1280
        self.hardcoded_cam_info.distortion_model = "plumb_bob"
        self.hardcoded_cam_info.D = [-0.08194476107782814, -0.06592640858415261, -0.0007043163003212235, 0.002577256982584405]
        # Intrinsic camera matrix K
        self.hardcoded_cam_info.K = [935.2355857804463, 0.0, 656.1572332633887,
                                     0.0, 935.7905325732659, 513.7144019593092,
                                     0.0, 0.0, 1.0]
        # Rotation matrix R (identity for mono camera)
        self.hardcoded_cam_info.R = [1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0]
        # Projection matrix P
        self.hardcoded_cam_info.P = [935.2355857804463, 0.0, 656.1572332633887, 0.0,
                                     0.0, 935.7905325732659, 513.7144019593092, 0.0,
                                     0.0, 0.0, 1.0, 0.0]
        self.hardcoded_cam_info.binning_x = 0
        self.hardcoded_cam_info.binning_y = 0
        self.hardcoded_cam_info.roi = RegionOfInterest()
        self.hardcoded_cam_info.roi.x_offset = 0
        self.hardcoded_cam_info.roi.y_offset = 0
        self.hardcoded_cam_info.roi.height = 0
        self.hardcoded_cam_info.roi.width = 0
        self.hardcoded_cam_info.roi.do_rectify = False
        # Update focal lengths from hardcoded P matrix (index 0 and 5)
        self.focal_length_x = self.hardcoded_cam_info.P[0]
        self.focal_length_y = self.hardcoded_cam_info.P[5]


        # Deques for storing messages
        dummy_thermal_tensor = torch.ones((1, 1, self.input_size, self.input_size), dtype=torch.float, device=DEVICE) # Adjusted shape
        self.thermal_deque = deque([dummy_thermal_tensor], maxlen=100)

        dummy_depth_msg = RImage()
        dummy_depth_msg.header.frame_id = "lwir_depth" # Frame ID for depth image
        dummy_depth_msg.encoding = "32FC1" # Use float encoding for depth
        self.depth_deque = deque([dummy_depth_msg], maxlen=100)

        dummy_compressed_msg = CompressedImage()
        self.msg_deque = deque([dummy_compressed_msg], maxlen=100)

        self.lwir_img_counter = 0

    def undistort_image(self, camera_info, image):
        #print('undistorting the image')
        K = np.array(camera_info.K).reshape(3, 3)
        dist_coeffs = np.array(camera_info.D)
        undistorted = cv2.undistort(image, K, dist_coeffs)
        # # Ensure the image has three channels
        # if len(undistorted.shape) == 2 or undistorted.shape[2] == 1:
        #     undistorted = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR)
        return undistorted, K, dist_coeffs

    def preprocess_thermal(self, img):
        # img = (img - img.mean()) / (img.std() + 1e-6)
        # img = torch.clip(img, min=-3, max=2)
        # img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        mean = np.mean([0.485, 0.456, 0.406])
        std = np.mean([0.229, 0.224, 0.225])
        print("thermal min", img.min(), "thermal max:", img.max())
        img = torch.tensor(img, dtype=torch.float32) / 255
        print("thermal min", img.min(), "thermal max:", img.max())
        img = (img - mean) / std 
        return img

    def read_thermal(self, msg):
        """Processes incoming thermal image."""
        self.lwir_img_counter += 1
        self.msg_deque.append(msg) # Store original compressed message for header info

        img = self._cv_bridge.compressed_imgmsg_to_cv2(msg) # Decode mono8
        print("GOT IMAGE - thermal min", img.min(), "thermal max:", img.max())
        # img, _, _ = self.undistort_image(self.hardcoded_cam_info, img)
        # Optional cropping (adjust as needed)
        # height, width 
        img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = img_resized[self.crop_top:self.crop_bottom, :]
        
        #img = img[top_crop:bottom_crop, :]

        img_resized = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.tensor(img_resized, dtype=torch.float32)
        img_processed = self.preprocess_thermal(img_tensor).unsqueeze(0).unsqueeze(0).to(device=self.device) # Add batch and channel dim

        self.thermal_deque.append(img_processed)


    def infer(self):
        """Performs inference and prepares depth message."""
        if len(self.thermal_deque) < 2: # Ensure there's a processed image (skip the dummy one)
             return

        with torch.no_grad():
            depth = self.model(self.thermal_deque[-1]).cpu().squeeze(0).squeeze(0).numpy() # Remove batch and channel dim
        print("MOdel output - Depth min: ", depth.min(), "Depth max: ", depth.max())
        # depth = depth * 30. # Scale depth as needed
        print("Scaled output - Depth min: ", depth.min(), "Depth max: ", depth.max())

        # Resize depth back to a target output size (e.g., camera resolution or desired size)
        # Using the hardcoded camera info dimensions: 1280x1024
        depth_resized = cv2.resize(depth, (self.hardcoded_cam_info.width, self.hardcoded_cam_info.height), interpolation=cv2.INTER_NEAREST)

        # Create depth message
        depth_msg = self._cv_bridge.cv2_to_imgmsg(depth_resized.astype(np.float32), encoding="32FC1") # Use 32FC1 for float depth


        # Use header from the corresponding input message
        last_input_msg = self.msg_deque[-1]
        depth_msg.header.stamp = last_input_msg.header.stamp
        # Set the frame_id consistent with the CameraInfo we are publishing
        depth_msg.header.frame_id = self.hardcoded_cam_info.header.frame_id

        self.depth_deque.append(depth_msg)

if __name__ == '__main__':
    rospy.init_node("depth_pub", anonymous=True)
    r = rospy.Rate(10) # Publish rate

    depth_pred = DepthPredictor(encoder='vits', input_size=256) # Ensure input_size matches training/model expected size

    while not rospy.is_shutdown():
        if len(depth_pred.msg_deque) > 1: # Check if a new message has arrived
            depth_pred.infer() # Perform inference and create depth message

            # Publish the latest depth image
            if len(depth_pred.depth_deque) > 1: # Check if inference produced a message
                current_depth_msg = depth_pred.depth_deque[-1]
                depth_pred.depth_pub.publish(current_depth_msg)

                # Publish the hardcoded CameraInfo message with the same timestamp
                depth_pred.hardcoded_cam_info.header.stamp = current_depth_msg.header.stamp
                depth_pred.cam_info_pub.publish(depth_pred.hardcoded_cam_info)

        r.sleep()