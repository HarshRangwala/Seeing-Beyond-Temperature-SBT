import rospy
import torch
import cv2
import time
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from model.m2p2_model import VisionEncoder, DepthDecoder
from utils.helpers import get_conf, init_device

# Import for GPU utilization monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    rospy.logwarn("pynvml not available. GPU utilization monitoring disabled. Install with: pip install nvidia-ml-py")

class DepthInference:
    def __init__(self, cfg_dir: str):
        # Initialize all attributes first to prevent AttributeError
        self.cfg = None
        self.device = None
        self.bridge = CvBridge()
        
        # Performance tracking attributes - initialize early
        self.inference_times = []
        self.preprocessing_times = []
        self.model_inference_times = []
        self.encoder_times = []
        self.decoder_times = []
        self.postprocessing_times = []
        self.frame_count = 0
        self.log_interval = 30  # Initialize this early
        self.start_time = time.time()
        
        # Preprocessing parameters
        self.crop_top = 40
        self.crop_bottom = 225
        self.img_size = (256, 256)
        self.normalization_mean = np.mean([0.485, 0.456, 0.406])
        self.normalization_std = np.mean([0.229, 0.224, 0.225])
        
        # GPU monitoring attributes
        self.gpu_handle = None
        
        
        # Initialize ROS node
        rospy.init_node('depth_inference', anonymous=True)
        rospy.loginfo("ROS node initialized successfully")
        
        # Load configuration
        self.cfg = get_conf(cfg_dir)
        
        # Initialize device
        self.device = init_device(self.cfg)
        
        # Initialize GPU monitoring
        if PYNVML_AVAILABLE and torch.cuda.is_available():
            
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
        
        # Load models
        self.vision_encoder, self.depth_decoder = self.load_models()
        rospy.loginfo("Models loaded successfully")
        
        
        # Set up ROS topics
        input_topic = '/sensor_suite/lwir/lwir/image_raw/compressed'
        output_topic = '/predicted_depth/image_raw'
        
        self.depth_pub = rospy.Publisher(output_topic, Image, queue_size=1)
        self.image_sub = rospy.Subscriber(input_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        
        rospy.loginfo("SBT Depth Inference node initialized and waiting for compressed images...")
            
        

    def load_models(self):
        try:   
            print('LOade models...')
            vision_encoder = VisionEncoder(num_layers=50, pretrained=False, num_channel=1)
            print("LOaded vision encoder")
            depth_decoder = DepthDecoder(latent_size=2048, num_layers=50)

            # encoder_path = '/mnt/DATASETS/SERVER_BACKUP/Server_3/harshr/home/NV_cahsor/CAHSOR-master/TRON/checkpoint/tron/ssl-ptr_aug-thermal_lidar-2048-04-13-14-42/ssl-ptr_aug-thermal_lidar-2048-04-15-14-31/ssl-ptr_aug-thermal_lidar-2048-04-15-14-31_500.pth'
            # decoder_path = '/mnt/DATASETS/SERVER_BACKUP/Server_3/harshr/home/NV_cahsor/CAHSOR-master/TRON/checkpoint/tron/decoder_depth_thermal_lidar_ckpts/decoder_depth_2-2048-04-16-03-43/depth_decoder_epoch_100.pth'                    
            encoder_path = '/home/husky/SBT/checkpoints/ssl-ptr_aug-thermal_lidar-2048-04-15-14-31_500.pth'
            decoder_path = '/home/husky/SBT/checkpoints/depth_decoder_epoch_100.pth'                    
            ssl_checkpoint = torch.load(encoder_path, map_location=self.device)
            print("load SSL Ckpt")
            encoder_state_dict = {k.replace('vision_encoder.', '', 1): v for k, v in ssl_checkpoint.items() if k.startswith('vision_encoder.')}
            vision_encoder.load_state_dict(encoder_state_dict, strict=False)

            
            decoder_checkpoint = torch.load(decoder_path, map_location=self.device)
            depth_decoder.load_state_dict(decoder_checkpoint['depth_decoder'])
            print("load Depth Ckpt")
            vision_encoder.to(self.device).eval()
            depth_decoder.to(self.device).eval()
            
            return vision_encoder, depth_decoder
        except Exception as e:
            rospy.logerr(f"Failed to load models: {e}")
            raise
            
       
    
    def preprocess_thermal(self, img):
        img = (img - img.mean()) / (img.std() + 1e-6)
        img = torch.clip(img, min=-3, max=2)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        return img

    def preprocess_image(self, cv_image):
        try:
            # Ensure the image is grayscale
            if len(cv_image.shape) == 3:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # 1. Resize to 256x256
            resized_image = cv2.resize(cv_image, self.img_size, interpolation=cv2.INTER_AREA)
            
            # 2. Crop from top=40 to bottom=225
            cropped_image = resized_image[self.crop_top:225, :]
            
            # 3. Resize cropped image back to full size
            resized_image2 = cv2.resize(cropped_image, self.img_size, interpolation=cv2.INTER_AREA)
            
            # 4. Convert to tensor FIRST
            img_tensor = torch.tensor(resized_image2, dtype=torch.float32)
            
            # 5. Apply thermal preprocessing
            img_tensor = self.preprocess_thermal(img_tensor)
            
            # 6. Add channel and batch dimensions: [H, W] -> [1, 1, H, W]
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            
            return img_tensor.to(self.device)
            
        except Exception as e:
            rospy.logerr(f"Failed to preprocess image: {e}")
            raise

    def get_gpu_metrics(self):
        """Get comprehensive GPU metrics"""
        gpu_metrics = {}
        
        
        if torch.cuda.is_available():
            # GPU Memory (always available with PyTorch)
            gpu_metrics['memory_allocated_mb'] = torch.cuda.memory_allocated(0) / 1024**2
            gpu_metrics['memory_reserved_mb'] = torch.cuda.memory_reserved(0) / 1024**2
            gpu_metrics['memory_total_mb'] = torch.cuda.get_device_properties(0).total_memory / 1024**2
            
            # GPU Utilization (requires pynvml)
            if self.gpu_handle is not None:
                
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_metrics['gpu_utilization_percent'] = gpu_util.gpu
                gpu_metrics['memory_utilization_percent'] = gpu_util.memory
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_metrics['temperature_celsius'] = temp
                
                # Power usage
                power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Convert to watts
                gpu_metrics['power_usage_watts'] = power
                        
                    
        
        
        return gpu_metrics

    def image_callback(self, msg):
        total_start_time = time.time()
        
        # Preprocessing timing
        preprocess_start = time.time()
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        img_tensor = self.preprocess_image(image)
        preprocess_time = time.time() - preprocess_start
        
        # Model inference timing - SPLIT INTO ENCODER AND DECODER
        with torch.no_grad():
            # Time the encoder separately
            encoder_start = time.time()
            v_encoded_thermal, thermal_features = self.vision_encoder(img_tensor, return_features=True)
            encoder_time = time.time() - encoder_start
            
            # Time the decoder separately
            decoder_start = time.time()
            predicted_depth = self.depth_decoder(v_encoded_thermal, thermal_features)
            decoder_time = time.time() - decoder_start
        
        # Total inference time (encoder + decoder)
        inference_time = encoder_time + decoder_time
        
        # Postprocessing timing
        postprocess_start = time.time()
        depth_map = predicted_depth.squeeze().cpu().numpy() 
        depth_msg = self.bridge.cv2_to_imgmsg(depth_map * 255.0)
        depth_msg.header = msg.header
        self.depth_pub.publish(depth_msg)
        postprocess_time = time.time() - postprocess_start
        
        # Record all timing metrics - ADD NEW TIMING ARRAYS
        total_time = time.time() - total_start_time
        self.inference_times.append(total_time)
        self.preprocessing_times.append(preprocess_time)
        self.model_inference_times.append(inference_time)
        self.encoder_times.append(encoder_time)  # NEW
        self.decoder_times.append(decoder_time)  # NEW
        self.postprocessing_times.append(postprocess_time)
        
        self.frame_count += 1
        if self.frame_count % self.log_interval == 0:
            self.log_performance()

                
        

    def log_performance(self):
        
        if not self.inference_times:
            return
        
        # Calculate timing statistics
        avg_total_latency = np.mean(self.inference_times)
        avg_preprocess_time = np.mean(self.preprocessing_times)
        avg_encoder_time = np.mean(self.encoder_times)
        avg_decoder_time = np.mean(self.decoder_times)
        avg_inference_time = np.mean(self.model_inference_times)
        avg_postprocess_time = np.mean(self.postprocessing_times)
        
        fps = 1.0 / avg_total_latency
        
        # Get GPU metrics
        gpu_metrics = self.get_gpu_metrics()
        
        # Calculate overall statistics
        total_runtime = time.time() - self.start_time
        overall_fps = self.frame_count / total_runtime
        
        rospy.loginfo("=" * 80)
        rospy.loginfo("SBT DEPTH ESTIMATION - PERFORMANCE METRICS")
        num_params_enc = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        rospy.loginfo(f"Vision Encoder Parameters: {num_params_enc:,} (trainable)")
        rospy.loginfo("=" * 80)
        # Depth decoder number of parameters
        num_params = sum(p.numel() for p in self.depth_decoder.parameters() if p.requires_grad)
        rospy.loginfo(f"Depth Decoder Parameters: {num_params:,} (trainable)")
        rospy.loginfo("=" * 80)

        
        # End-to-end inference latency (as requested by Reviewer FqHr)
        rospy.loginfo(f"END-TO-END INFERENCE LATENCY (last {self.log_interval} frames):")
        rospy.loginfo(f"   • Total Latency: {avg_total_latency * 1000:.2f} ms")
        rospy.loginfo(f"   • Preprocessing: {avg_preprocess_time * 1000:.2f} ms ({avg_preprocess_time/avg_total_latency*100:.1f}%)")
        rospy.loginfo(f"   • Model Inference: {avg_inference_time * 1000:.2f} ms ({avg_inference_time/avg_total_latency*100:.1f}%)")
        rospy.loginfo(f"   • Postprocessing: {avg_postprocess_time * 1000:.2f} ms ({avg_postprocess_time/avg_total_latency*100:.1f}%)")
        rospy.loginfo(f"     ├─ Vision Encoder: {avg_encoder_time * 1000:.2f} ms ({avg_encoder_time/avg_total_latency*100:.1f}%)")  # NEW
        rospy.loginfo(f"     └─ Depth Decoder: {avg_decoder_time * 1000:.2f} ms ({avg_decoder_time/avg_total_latency*100:.1f}%)")
        # Frame rate (as requested by Reviewer RAuc)
        rospy.loginfo(f"FRAME RATE:")
        rospy.loginfo(f"   • Instantaneous FPS: {fps:.2f}")
        rospy.loginfo(f"   • Overall FPS: {overall_fps:.2f}")
        rospy.loginfo(f"   • Total Frames Processed: {self.frame_count}")
        
        # GPU memory usage (as requested by Reviewer FqHr)
        if gpu_metrics:
            rospy.loginfo(f"GPU MEMORY USAGE:")
            rospy.loginfo(f"   • Allocated: {gpu_metrics['memory_allocated_mb']:.2f} MB")
            rospy.loginfo(f"   • Reserved: {gpu_metrics['memory_reserved_mb']:.2f} MB")
            rospy.loginfo(f"   • Total Available: {gpu_metrics['memory_total_mb']:.2f} MB")
            rospy.loginfo(f"   • Memory Utilization: {gpu_metrics['memory_allocated_mb']/gpu_metrics['memory_total_mb']*100:.1f}%")
            
            # GPU usage (as requested by Reviewer RAuc)
            if 'gpu_utilization_percent' in gpu_metrics:
                rospy.loginfo(f"⚡ GPU UTILIZATION:")
                rospy.loginfo(f"   • GPU Usage: {gpu_metrics['gpu_utilization_percent']}%")
                rospy.loginfo(f"   • Memory Usage: {gpu_metrics['memory_utilization_percent']}%")
                rospy.loginfo(f"   • Temperature: {gpu_metrics['temperature_celsius']}°C")
                rospy.loginfo(f"   • Power Usage: {gpu_metrics['power_usage_watts']:.1f} W")
        
        rospy.loginfo("=" * 80)
        
        # Clear timing arrays for next interval
        self.inference_times = []
        self.preprocessing_times = []
        self.model_inference_times = []
        self.encoder_times = []
        self.decoder_times = []
        self.postprocessing_times = []
   

if __name__ == '__main__':
    try:
        depth_inference_node = DepthInference("/mnt/DATASETS/SERVER_BACKUP/Server_3/harshr/home/NV_cahsor/CAHSOR-master/TRON/conf/config_m2p2.yaml")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("SBT Depth Inference node terminated.")
    except Exception as e:
        rospy.logerr(f"SBT Depth Inference node failed: {e}")
        import traceback
        rospy.logerr(f"Full traceback: {traceback.format_exc()}")
