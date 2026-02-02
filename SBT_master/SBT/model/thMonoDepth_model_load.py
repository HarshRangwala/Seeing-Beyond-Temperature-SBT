import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2

# Add ThermalMonoDepth to path
THERMAL_MONO_DEPTH_PATH = Path(__file__).resolve().parent / 'ThermalMonoDepth-main'
sys.path.append(str(THERMAL_MONO_DEPTH_PATH))
sys.path.append(str(THERMAL_MONO_DEPTH_PATH / 'common'))

from common import models

from tqdm import tqdm

class ThermalMonoDepthModel:
    def __init__(self, resnet_layers=18, scene_type='outdoor', max_depth=30):
        """Initialize ThermalMonoDepth model
        
        Args:
            resnet_layers (int): ResNet backbone (18 or 50)
            scene_type (str): 'indoor' or 'outdoor'
            max_depth (float): Maximum depth value
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet_layers = resnet_layers
        self.scene_type = scene_type
        self.max_depth = max_depth
        
        
        # Create model
        self.disp_pose_net = models.DispPoseResNet(self.resnet_layers, False, num_channel=1).to(self.device)
        self.disp_net = self.disp_pose_net.DispResNet
        
        # Model is initialized but not loaded with weights yet
        self.model_loaded = False

    def load_weights(self, weights_path):
        """Load pretrained weights
        
        Args:
            weights_path (str): Path to pretrained model weights
        """
        weights = torch.load(weights_path)
        self.disp_net.load_state_dict(weights['state_dict'])
        self.disp_net.eval()
        self.model_loaded = True
    
    def preprocess_thermal(self, thermal_img):
        """
        Adapt the ThermalMonoDepth preprocessing to work with 
        thermal images
        
        Args:
            thermal_img: Thermal image (grayscale)
        """
        # Convert RGB to grayscale if needed
        if len(thermal_img.shape) == 3 and thermal_img.shape[2] == 3:
            thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_RGB2GRAY)
        
        # Resize if needed
        # if thermal_img.shape[0] != 256 or thermal_img.shape[1] != 320:
        # thermal_img = cv2.resize(thermal_img, (256, 256))
        
        # Make sure we have a numpy array
        if not isinstance(thermal_img, np.ndarray):
            thermal_img = thermal_img.numpy()
        
        # Ensure we have a 3D tensor with shape [C, H, W]
        if len(thermal_img.shape) == 2:  # If image is just [H, W]
            thermal_img = np.expand_dims(thermal_img, axis=0)  # Make it [1, H, W]
        elif len(thermal_img.shape) == 3 and thermal_img.shape[2] == 1:  # If image is [H, W, 1]
            thermal_img = np.transpose(thermal_img, (2, 0, 1))  # Make it [1, H, W]
        
        # Convert to tensor
        thermal_tensor = torch.from_numpy(thermal_img).float()
        
        # Normalize according to model's expected input
        # thermal_tensor = thermal_tensor / 255.0  # Scale to [0,1]
        
        # Add batch dimension and normalize
        # This is the critical part - ensure we have a 4D tensor [B, C, H, W]
        if len(thermal_tensor.shape) == 3:  # If tensor is [C, H, W]
            thermal_tensor = thermal_tensor.unsqueeze(0)  # Make it [1, C, H, W]
        
        # Apply normalization
        thermal_tensor = ((thermal_tensor - 0.45) / 0.225).to(self.device)
        
        return thermal_tensor

    
    def infer_image(self, thermal_img):
        """Infer depth from thermal image
        
        Args:
            thermal_img (numpy.ndarray): Thermal image (grayscale or RGB)
            
        Returns:
            numpy.ndarray: Predicted depth map (metric)
        """
        if not self.model_loaded:
            raise RuntimeError("Model weights not loaded. Call load_weights() first.")
        
        # Preprocess the image
        thermal_tensor = self.preprocess_thermal(thermal_img)
        
        # Run inference
        with torch.no_grad():
            output = self.disp_net(thermal_tensor)
            
        # Convert disparity to depth
        pred_disp = output.squeeze().cpu().numpy()
        pred_depth = 1.0 / pred_disp
        
        return pred_depth

def load_ThermalMonoDepth_model(weights_path, resnet_layers = 18, scene_type='outdoor', max_depth=30):
    """Load ThermalMonoDepth model with pretrained weights
    
    Args:
        weights_path (str): Path to pretrained weights
        resnet_layers (int): ResNet backbone (18 or 50)
        scene_type (str): 'indoor' or 'outdoor'
        max_depth (float): Maximum depth value
        
    Returns:
        ThermalMonoDepthModel: Loaded model
    """
    model = ThermalMonoDepthModel(resnet_layers, scene_type, max_depth)
    model.load_weights(weights_path)
    return model

def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    Args:
        gt (N): ground truth depth (metric)
        pred (N): predicted depth (metric)
    """
    # Ensure non-zero GT values for division and log
    gt = gt[gt > 0]
    pred = pred[gt > 0]

    # Ensure non-zero Pred values after filtering based on GT
    gt = gt[pred > 0]
    pred = pred[pred > 0]

    if gt.shape[0] == 0:
        # Return default high errors or NaNs if no valid pixels
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    # Check for non-positive values before log
    valid_log_mask = (gt > 1e-6) & (pred > 1e-6)
    if np.sum(valid_log_mask) > 0:
        rmse_log = (np.log(gt[valid_log_mask]) - np.log(pred[valid_log_mask])) ** 2
        rmse_log = np.sqrt(rmse_log.mean())
        log10 = np.mean(np.abs((np.log10(gt[valid_log_mask]) - np.log10(pred[valid_log_mask]))))
    else:
        rmse_log = np.nan
        log10 = np.nan # Added log10 handling


    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


