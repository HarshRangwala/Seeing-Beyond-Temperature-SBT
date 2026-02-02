#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import torch
import sys

# Standard ROS Messages
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from std_msgs.msg import Header

# Import your model wrapper
from depth_costmap import load_deployment_model

class ThermalDepthNode:
    def __init__(self):
        rospy.init_node('thermal_depth_inference', anonymous=True)
        
        # --- CONFIGURATION (Matches your 256x256 Checkpoint) ---
        self.ENC_PATH = "/mnt/sbackup/Server_3/harshr/home/NV_cahsor/CAHSOR-master/TRON/checkpoint/tron/ssl-ptr-thermal_lidar_video_2/SBT_Plan2-2048-11-25-19-09/SBT_Plan2-2048-11-25-19-09_100.pth"
        self.DEC_PATH = "/mnt/sbackup/Server_3/harshr/home/NV_cahsor/CAHSOR-master/TRON/checkpoint/tron/decoder_depth_ckpts/SBT_move_base_exp/depth_estimation_move_base-2048-11-27-23-50/depth_decoder_epoch_050.pth"
        
        # Geometry
        self.ORIG_W = 1280
        self.ORIG_H = 1024
        self.CROP_TOP = 165
        self.CROP_BOTTOM = 74 
        self.MODEL_INPUT_SIZE = (256, 256) 
        
        # Normalization
        self.THERMAL_MEAN = 0.495356
        self.THERMAL_STD = 0.191781
        self.DEPTH_MEAN = 0.561041
        self.DEPTH_STD = 0.295559
        self.CLIP_DIST = 30.0
        self.REG_FACTOR = 3.7
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inference Device: {self.device}")

        # Load Model
        self.model = load_deployment_model(self.ENC_PATH, self.DEC_PATH, self.device)

        # ROS Setup
        self.sub = rospy.Subscriber("/sensor_suite/lwir/lwir/image_raw/compressed", CompressedImage, self.callback, queue_size=1)
        self.pub_depth = rospy.Publisher("/thermal/depth_pred", Image, queue_size=1)
        self.pub_info = rospy.Publisher("/thermal/camera_info", CameraInfo, queue_size=1)
        
        self.cam_info = self.create_original_camera_info()

    def create_original_camera_info(self):
        info = CameraInfo()
        info.header.frame_id = "ir_camera"
        info.height = self.ORIG_H
        info.width = self.ORIG_W
        info.distortion_model = "plumb_bob"
        info.D = [-0.08194476, -0.06592641, -0.00070432, 0.00257726]
        info.K = [935.23558, 0.0, 656.15723, 0.0, 935.79053, 513.71440, 0.0, 0.0, 1.0]
        info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.P = [935.23558, 0.0, 656.15723, 0.0, 0.0, 935.79053, 513.71440, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def preprocess(self, img_np):
        h, w = img_np.shape
        # 1. Crop
        cropped = img_np[self.CROP_TOP : h - self.CROP_BOTTOM, :]
        # 2. Resize
        resized = cv2.resize(cropped, self.MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # 3. Normalize
        if img_np.dtype == np.uint16:
            img_tensor = torch.from_numpy(resized).float() / 65535.0
        else:
            img_tensor = torch.from_numpy(resized).float() / 255.0
            
        img_tensor = (img_tensor - self.THERMAL_MEAN) / self.THERMAL_STD
        return img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

    def postprocess(self, z_score_pred):
        log_norm = (z_score_pred * self.DEPTH_STD) + self.DEPTH_MEAN
        log_depth = (log_norm - 1.0) * self.REG_FACTOR
        linear_norm = torch.exp(log_depth)
        metric_depth = linear_norm * self.CLIP_DIST
        metric_depth = torch.clamp(metric_depth, 0.1, self.CLIP_DIST)
        return metric_depth.cpu().squeeze().numpy()

    def callback(self, msg):
        try:
            # --- 1. MANUAL DECODING (Replaces cv_bridge) ---
            # Input is CompressedImage (jpg/png)
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            
            if cv_img is None:
                return

            # --- 2. INFERENCE ---
            input_tensor = self.preprocess(cv_img)
            with torch.no_grad():
                pred_raw = self.model(input_tensor)
            
            # --- 3. POST-PROCESS & RESTORE GEOMETRY ---
            pred_meters = self.postprocess(pred_raw) 
            
            # Calculate original crop dimensions
            crop_height = self.ORIG_H - self.CROP_TOP - self.CROP_BOTTOM 
            crop_width = self.ORIG_W 

            # Upscale 256x256 back to Crop Size
            pred_upscaled = cv2.resize(pred_meters, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR)
            print(pred_upscaled.shape, "Upscaled Depth Stats - min:", pred_upscaled.min(), "max:", pred_upscaled.max())            
            # Create Full Frame canvas (Invalid data = 0.0)
            full_frame_depth = np.zeros((self.ORIG_H, self.ORIG_W), dtype=np.float32)
            
            # Paste prediction in the middle
            full_frame_depth[self.CROP_TOP : self.ORIG_H - self.CROP_BOTTOM, :] = pred_upscaled
            print(full_frame_depth.shape, "Full Frame Depth Stats - min:", full_frame_depth[full_frame_depth>0].min(), "max:", full_frame_depth.max())
            # --- 4. MANUAL ENCODING (Replaces cv_bridge) ---
            # We are creating a raw sensor_msgs/Image
            depth_msg = Image()
            depth_msg.header = msg.header
            depth_msg.header.frame_id = "ir_camera"
            depth_msg.height = self.ORIG_H
            depth_msg.width = self.ORIG_W
            depth_msg.encoding = "32FC1" # Float32, 1 Channel
            depth_msg.is_bigendian = 0
            depth_msg.step = self.ORIG_W * 4 # 4 bytes per pixel (float32)
            
            # Convert numpy float32 array to bytes
            depth_msg.data = full_frame_depth.tobytes()
            
            # Sync Camera Info
            self.cam_info.header = msg.header
            self.cam_info.header.frame_id = "ir_camera" 
            
            # --- 5. PUBLISH ---
            self.pub_depth.publish(depth_msg)
            self.pub_info.publish(self.cam_info)
            
        except Exception as e:
            rospy.logerr(f"Inference Error: {e}")

if __name__ == '__main__':
    node = ThermalDepthNode()
    rospy.spin()