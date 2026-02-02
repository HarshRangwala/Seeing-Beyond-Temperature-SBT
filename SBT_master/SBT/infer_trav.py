import rospy
import torch
import cv2
import time
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from model.m2p2_model import VisionEncoder
from model.traversability_model import TraversabilityDecoder  # Import traversability decoder
from utils.helpers import get_conf, init_device

# Import for GPU utilization monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    rospy.logwarn("pynvml not available. GPU utilization monitoring disabled. Install with: pip install nvidia-ml-py")

class TraversabilityInference:
    def __init__(self, cfg_dir: str):
        # Initialize all attributes first
        self.cfg = None
        self.device = None
        self.bridge = CvBridge()
        
        # Performance tracking attributes
        self.inference_times = []
        self.preprocessing_times = []
        self.model_inference_times = []
        self.encoder_times = []
        self.decoder_times = []
        self.postprocessing_times = []
        self.frame_count = 0
        self.log_interval = 30
        self.start_time = time.time()
        
        # Preprocessing parameters
        self.crop_top = 40
        self.crop_bottom = 225
        self.img_size = (256, 256)
        
        # GPU monitoring attributes
        self.gpu_handle = None
        
        # Initialize ROS node
        rospy.init_node('traversability_inference', anonymous=True)
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
        self.vision_encoder, self.traversability_decoder = self.load_models()
        rospy.loginfo("Models loaded successfully")
        
        # Set up ROS topics
        input_topic = '/sensor_suite/lwir/lwir/image_raw/compressed'
        output_topic = '/predicted_traversability/image_raw'
        
        self.traversability_pub = rospy.Publisher(output_topic, Image, queue_size=1)
        self.image_sub = rospy.Subscriber(input_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        
        rospy.loginfo("SBT Traversability Inference node initialized and waiting for compressed images...")

    def load_models(self):
        """Load vision encoder and traversability decoder"""
        # Load vision encoder (same as before)
        vision_encoder = VisionEncoder(num_layers=50, pretrained=False, num_channel=1)
        encoder_path = '/home/husky/SBT/checkpoints/ssl-ptr_aug-thermal_lidar-2048-04-15-14-31_500.pth'
        ssl_checkpoint = torch.load(encoder_path, map_location=self.device)
        encoder_state_dict = {k.replace('vision_encoder.', '', 1): v for k, v in ssl_checkpoint.items() if k.startswith('vision_encoder.')}
        vision_encoder.load_state_dict(encoder_state_dict, strict=False)
        
        # Load traversability decoder (NEW)
        traversability_decoder = TraversabilityDecoder(
            latent_size=2048,  # Should match encoder output
            num_layers=50       # Should match your config
        )
        decoder_path = '/home/husky/SBT/checkpoints/traversability_decoder_epoch_100.pth'
        decoder_checkpoint = torch.load(decoder_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if "traversability_decoder" in decoder_checkpoint:
            traversability_decoder.load_state_dict(decoder_checkpoint["traversability_decoder"])
        else:
            traversability_decoder.load_state_dict(decoder_checkpoint)
        
        vision_encoder.to(self.device).eval()
        traversability_decoder.to(self.device).eval()
        
        return vision_encoder, traversability_decoder

    def preprocess_thermal(self, img):
        img = (img - img.mean()) / (img.std() + 1e-6)
        img = torch.clip(img, min=-3, max=2)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        return img

    def preprocess_image(self, cv_image):
        if len(cv_image.shape) == 3:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
        # Resize with LINEAR interpolation (matches albumentations)
        resized_image = cv2.resize(cv_image, self.img_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor and apply thermal preprocessing
        img_tensor = torch.tensor(resized_image, dtype=torch.float32)
        img_tensor = self.preprocess_thermal(img_tensor)
        
        # Add channel dimension: [H, W] -> [1, H, W] (matches training dataset)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  
        
        return img_tensor.to(self.device)

    def get_gpu_metrics(self):
        """Get comprehensive GPU metrics"""
        gpu_metrics = {}
        
        if torch.cuda.is_available():
            # GPU Memory
            gpu_metrics['memory_allocated_mb'] = torch.cuda.memory_allocated(0) / 1024**2
            gpu_metrics['memory_reserved_mb'] = torch.cuda.memory_reserved(0) / 1024**2
            gpu_metrics['memory_total_mb'] = torch.cuda.get_device_properties(0).total_memory / 1024**2
            
            # GPU Utilization
            if self.gpu_handle is not None:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_metrics['gpu_utilization_percent'] = gpu_util.gpu
                gpu_metrics['memory_utilization_percent'] = gpu_util.memory
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_metrics['temperature_celsius'] = temp
                
                # Power usage
                power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
                gpu_metrics['power_usage_watts'] = power
        
        return gpu_metrics

    def image_callback(self, msg):
        total_start_time = time.time()
        
        # Preprocessing timing
        preprocess_start = time.time()
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        img_tensor = self.preprocess_image(image)
        preprocess_time = time.time() - preprocess_start
        
        # Model inference timing
        with torch.no_grad():
            # Time the encoder
            encoder_start = time.time()
            v_encoded_thermal, thermal_features = self.vision_encoder(img_tensor, return_features=True)
            encoder_time = time.time() - encoder_start
            
            # Time the traversability decoder
            decoder_start = time.time()
            predicted_traversability = self.traversability_decoder(v_encoded_thermal, thermal_features)
            decoder_time = time.time() - decoder_start
        
        # Total inference time
        inference_time = encoder_time + decoder_time
        
        # Postprocessing timing
        postprocess_start = time.time()
        trav_map = predicted_traversability.squeeze().cpu().numpy() 
        trav_msg = self.bridge.cv2_to_imgmsg(trav_map * 255.0)  # Scale to 0-255
        trav_msg.header = msg.header
        self.traversability_pub.publish(trav_msg)
        postprocess_time = time.time() - postprocess_start
        
        # Record timing metrics
        total_time = time.time() - total_start_time
        self.inference_times.append(total_time)
        self.preprocessing_times.append(preprocess_time)
        self.model_inference_times.append(inference_time)
        self.encoder_times.append(encoder_time)
        self.decoder_times.append(decoder_time)
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
        rospy.loginfo("SBT TRAVERSABILITY ESTIMATION - PERFORMANCE METRICS")
        rospy.loginfo("=" * 80)
        
        # Log model parameters
        num_params_enc = sum(p.numel() for p in self.vision_encoder.parameters())
        num_params_dec = sum(p.numel() for p in self.traversability_decoder.parameters())
        rospy.loginfo(f"Vision Encoder Parameters: {num_params_enc:,}")
        rospy.loginfo(f"Traversability Decoder Parameters: {num_params_dec:,}")
        rospy.loginfo("=" * 80)
        
        # End-to-end latency
        rospy.loginfo(f"END-TO-END INFERENCE LATENCY (last {self.log_interval} frames):")
        rospy.loginfo(f"   • Total Latency: {avg_total_latency * 1000:.2f} ms")
        rospy.loginfo(f"   • Preprocessing: {avg_preprocess_time * 1000:.2f} ms ({avg_preprocess_time/avg_total_latency*100:.1f}%)")
        rospy.loginfo(f"   • Model Inference: {avg_inference_time * 1000:.2f} ms ({avg_inference_time/avg_total_latency*100:.1f}%)")
        rospy.loginfo(f"     ├─ Vision Encoder: {avg_encoder_time * 1000:.2f} ms ({avg_encoder_time/avg_total_latency*100:.1f}%)")
        rospy.loginfo(f"     └─ Traversability Decoder: {avg_decoder_time * 1000:.2f} ms ({avg_decoder_time/avg_total_latency*100:.1f}%)")
        rospy.loginfo(f"   • Postprocessing: {avg_postprocess_time * 1000:.2f} ms ({avg_postprocess_time/avg_total_latency*100:.1f}%)")
        
        # Frame rate
        rospy.loginfo(f"FRAME RATE:")
        rospy.loginfo(f"   • Instantaneous FPS: {fps:.2f}")
        rospy.loginfo(f"   • Overall FPS: {overall_fps:.2f}")
        rospy.loginfo(f"   • Total Frames Processed: {self.frame_count}")
        
        # GPU memory usage
        if gpu_metrics:
            rospy.loginfo(f"GPU MEMORY USAGE:")
            rospy.loginfo(f"   • Allocated: {gpu_metrics['memory_allocated_mb']:.2f} MB")
            rospy.loginfo(f"   • Reserved: {gpu_metrics['memory_reserved_mb']:.2f} MB")
            rospy.loginfo(f"   • Total Available: {gpu_metrics['memory_total_mb']:.2f} MB")
            rospy.loginfo(f"   • Memory Utilization: {gpu_metrics['memory_allocated_mb']/gpu_metrics['memory_total_mb']*100:.1f}%")
            
            if 'gpu_utilization_percent' in gpu_metrics:
                rospy.loginfo(f"GPU UTILIZATION:")
                rospy.loginfo(f"   • GPU Usage: {gpu_metrics['gpu_utilization_percent']}%")
                rospy.loginfo(f"   • Memory Usage: {gpu_metrics['memory_utilization_percent']}%")
                rospy.loginfo(f"   • Temperature: {gpu_metrics['temperature_celsius']}°C")
                rospy.loginfo(f"   • Power Usage: {gpu_metrics['power_usage_watts']:.1f} W")
        
        rospy.loginfo("=" * 80)
        
        # Clear timing arrays
        self.inference_times = []
        self.preprocessing_times = []
        self.model_inference_times = []
        self.encoder_times = []
        self.decoder_times = []
        self.postprocessing_times = []

if __name__ == '__main__':
    try:
        trav_inference_node = TraversabilityInference("./conf/config_m2p2")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("SBT Traversability Inference node terminated.")
    except Exception as e:
        rospy.logerr(f"SBT Traversability Inference node failed: {e}")
        import traceback
        rospy.logerr(f"Full traceback: {traceback.format_exc()}")
