import io
from collections import deque

from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
import rospy
from sensor_msgs.msg import CompressedImage, Image as RImage, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import open3d as o3d

# from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from depth_costmap import load_models

FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8

class DepthPredictor:
    def __init__(self, encoder: str = 'vitb', input_size: int = 256, max_depth: int = 18) -> None:
        """Reads thermal images and predicts and publishes depth data

        Args:
            encoder: (str) model size, options: ['vits', 'vitb', 'vitl', 'vitg']
            input_size: tuple[int, int] image size to pass to the model
        """
        self.input_size = input_size
        # self.max_depth = max_depth
        self.focal_length_x = 2715.79693
        self.focal_length_y = 2715.79693
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = DEVICE

        # model_configs = {
        #     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        #     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        #     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        # }

        # model = DepthAnythingV2(**{**model_configs[encoder], "max_depth": max_depth})
        # model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        # checkpointkitti = 'checkpoints/depth_anything_v2_metric_vkitti_vits.pth'
        # checkpointhypersim = 'checkpoints/depth_anything_v2_metric_hypersim_vits.pth'
        # model.load_state_dict(torch.load(checkpointkitti, map_location='cpu'))
        # self.model = model.to(DEVICE).eval()
        self.transform = transforms.Resize((256, 256), antialias=True)
        self.model = load_models(encoder_ckpt_path="./weights/met_th_lidar/best_model.pth",
                decoder_ckpt_path="./weights/met_decoder_thldr/best_model.pth",
                latent_size=2048,
                num_layers= 50,
                device=DEVICE
                ).eval()
        self.image = np.array(Image.new('RGB', (input_size, input_size)))
        self._cv_bridge = CvBridge()

        # self.depth_pub = rospy.Publisher("/sensor_suite/right_camera_optical/depth", RImage, queue_size=1)
        self.depth_pub = rospy.Publisher("/sensor_suite/lwir/lwir/depth", RImage, queue_size=1)
        # self.cam_info_pub = rospy.Publisher("/sensor_suite/right_camera_optical/depth/camera_info", CameraInfo, queue_size=1)
        # self.cam_info_sub = rospy.Subscriber("/sensor_suite/right_camera_optical/camera_info", CameraInfo, self.read_cam_info)
        self.cam_info_pub = rospy.Publisher("/sensor_suite/lwir/lwir/depth/camera_info", CameraInfo, queue_size=1)
        self.cam_info_sub = rospy.Subscriber("/sensor_suite/lwir/lwir/camera_info", CameraInfo, self.read_cam_info)
        # self.thermal_sub = rospy.Subscriber("/sensor_suite/lwir/lwir/image_raw", RImage, self.read_thermal)
        self.thermal_sub = rospy.Subscriber("/sensor_suite/lwir/lwir/image_raw/compressed", CompressedImage, self.read_thermal)
        # self.thermal_sub = rospy.Subscriber("/sensor_suite/right_camera_optical/image_color", RImage, self.read_thermal)
        # self.point_cloud_pub = rospy.Publisher("/sensor_suite/estimated_pcd", PointCloud2, queue_size=1)
        depth_msg = RImage()
        image = torch.ones((1, self.input_size, self.input_size), dtype=torch.float, device=DEVICE)#RImage()
        self.thermal_deque = deque([image], maxlen=100)
        
        self.pc = PointCloud2()
        msg = RImage()
        depth_msg.header.frame_id = "lwir_depth"
        depth_msg.encoding = "mono8"
        self.depth_deque = deque([depth_msg], maxlen=100)
        self.msg_deque = deque([msg], maxlen=100)
        self.writing_img = np.array(Image.new('RGB', (1024, 720)))	
        self.lwir_img_counter = 0

    def preprocess_thermal(self, img):
        img = (img - img.mean()) / (img.std() + 1e-6)
        img = torch.clip(img, min=-3, max=2)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        return img

    def read_thermal(self, msg):
        """
        Process thermal image data from a topic that publishes sensor_msgs/CompressedImage to a PIL image
        """
        self.lwir_img_counter += 1 
        # convert sensor_msgs/CompressedImage to PIL image
        #Reding the img
        # self.msg = msg
        self.msg_deque.append(msg)
        # for Image
        # img = self._cv_bridge.imgmsg_to_cv2(msg, "mono8") #bgr8  # Changed HR
        img = self._cv_bridge.compressed_imgmsg_to_cv2(msg) 
        height, width = img.shape
        top_crop = int(height * 0.20)
        bottom_crop = height - int(height * 0.2)
        img = img[top_crop:bottom_crop, :]
        # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        # crop_top:  40 crop_bottom: 225 , resize: (256, 256)
        # img = img[40:225, :]
        # self.image = np.array(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), dtype=np.uint8)
        # img = np.array(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), dtype=np.uint8)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        img = torch.tensor(img, dtype=torch.float32)
        img = self.preprocess_thermal(img).unsqueeze(0).to(device=self.device)
        # print(f"{type(img) = }, {img.shape = }")
        self.thermal_deque.append(img)
        # self.image = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        # for CompressedImage
        # np_arr = np.fromstring(msg.data, np.uint8)
        # self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        #inference from the model
        # depth = self.infer(img)

    def read_cam_info(self, msg):
        self.cam_info_pub.publish(msg)
        

    def infer(self):
        # depth = self.model.infer_image(self.image)
        depth = self.model(self.thermal_deque[-1].unsqueeze(0)).cpu().squeeze(0).numpy()
        depth = depth * 30.
        # print(f"{depth.min() = }, {depth.max() = }")
        ###### Normalizations ###############

        # # max_depth = depth.max()
        # reversed_depth = np.ones_like(depth) * 255
        # depth = (reversed_depth - depth)/255
        # # depth = depth
        # # print(f"{depth.max() = }, {depth.min() = }, {depth.shape = }")
        # depth = (depth - depth.min()) / (depth.max() - depth.min())
        # depth = (depth * 10) + 2
        # # depth = depth.astype(np.uint8)
	    # # depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        
        # # self.writing_img = depth.copy()
        # # self.conv2pointcloud(depth)

        ########End of Normalization #########
        
        #publishing
        # self.depth_msg = self._cv_bridge.cv2_to_imgmsg(depth, encoding="passthrough")
        # self.depth_msg.header.stamp = self.msg.header.stamp
        # self.depth_msg.header.frame_id = self.msg.header.frame_id
        depth_resized = cv2.resize(depth[0],(1024, 1024))
        depth_msg = self._cv_bridge.cv2_to_imgmsg(depth_resized, encoding="passthrough")
	
        depth_msg.header.stamp = self.msg_deque[-1].header.stamp
        depth_msg.header.frame_id = self.msg_deque[-1].header.frame_id
        self.depth_deque.append(depth_msg)
        # return depth


    #TODO change frame to the camera you are using for the pointcloud
    def conv2pointcloud(self, depth):
        height, width = depth.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / self.focal_length_x
        y = (y - height / 2) / self.focal_length_y
        z = np.array(depth)
        # points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        # colors = np.array(self.image).reshape(-1, 3)
        # colors = np.floor(np.asarray(colors) * 255)
        # colors = colors[:, 0] * BIT_MOVE_16 + colors[:, 1] * BIT_MOVE_8 + colors[:, 2]

        self.pc = pc2.create_cloud(self.msg.header, FIELDS_XYZ, points)
        # self.point_cloud_pub.publish(pub_pc2)

if __name__ == '__main__':
    rospy.init_node("depth_pub", anonymous=True)
    r = rospy.Rate(10)

    # depth_pred = DepthPredictor()
    # depth_pub = rospy.Publisher("/depth_image", CompressedImage, queue_size=1)  # queue_size = 50
    # thermal_sub = rospy.Subscriber("/sensor_suite/lwir/lwir/image_raw/compressed", CompressedImage, depth_pred.read_thermal)
    depth_pred = DepthPredictor(encoder='vits')
    
    while not rospy.is_shutdown():
        depth_pred.infer()
        # depth_pred.depth_pub.publish(depth_pred.depth_msg)
        depth_pred.depth_pub.publish(depth_pred.depth_deque[-1])
        # depth_pred.point_cloud_pub.publish(depth_pred.pc)

        #img_here = depth_pred.writing_img#np.repeat(depth_pred.writing_img[..., np.newaxis], 3, axis=-1)
        # cv2.imwrite(f"dimg{depth_pred.depth_deque[-1].header.seq}.jpg", depth_pred.writing_img)
        r.sleep()
    # rospy.spin()
