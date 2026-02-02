import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
import os
from datetime import datetime
import cv2

# Import your project modules (adjust paths as needed)
from model.m2p2_model import VisionEncoder, DepthDecoder, TronModel, DepthLoss
from model.thMonoDepth_model_load import load_ThermalMonoDepth_model
from utils.io import load_checkpoint
from utils.helpers import get_conf, init_device

class SimplifiedDepthTester:
    def __init__(self, cfg_path, checkpoint_path, input_folder, output_folder):
        """Initialize the tester with configuration and checkpoint paths"""
        self.cfg = get_conf(cfg_path)
        self.device = init_device(self.cfg)
        self.checkpoint_path = checkpoint_path
        self.input_folder = Path(input_folder)
        self.depth_gt_folder = None
        
        # Create output directory for results
        self.output_dir = Path(output_folder)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models
        self.pretrained_model = self.load_pretrained_model()
        self.depth_decoder = self.load_depth_decoder()
        self.thermalMonoDepth = load_ThermalMonoDepth_model("/mnt/sbackup/Server_3/harshr/home/NV_cahsor/CAHSOR-master/TRON/checkpoint/thMonoDepth/dispnet_disp_model_best.pth.tar")
        
        # Initialize loss function for metrics calculation
        self.criterion = DepthLoss().to(self.device)
        
        # Get list of thermal images
        self.thermal_files = self.get_thermal_files()
        print(f"Found {len(self.thermal_files)} thermal images in {self.input_folder}")
        
        # Initialize metrics storage
        self.metrics = {
            'loss': [],
            'abs_rel': [],
            'rmse': [],
            'delta1': [],
            'delta2': [],
            'delta3': []
        }
    
    def get_thermal_files(self):
        """Get list of thermal images from input folder"""
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        files = []
        for ext in extensions:
            files.extend(list(self.input_folder.glob(f"*{ext}")))
        return sorted(files)
    
    def load_pretrained_model(self):
        """Load the pretrained vision encoder"""
        print(f"Loading pretrained base model from {self.cfg.directory.pretrained_path}")
        
        vision_encoder = VisionEncoder(
            latent_size=self.cfg.model.rep_size,
            num_layers=self.cfg.model.num_layers_enc
        ).to(self.device)
        
        model = TronModel(
            vision_encoder=vision_encoder,
            projector=None,
            latent_size=self.cfg.model.rep_size
        ).to(self.device)
        
        checkpoint = load_checkpoint(self.cfg.directory.pretrained_path, self.device)
        
        if "vision_encoder" in checkpoint:
            vision_encoder.load_state_dict(checkpoint["vision_encoder"])
            print("Loaded vision encoder weights directly")
        else:
            model.load_state_dict(checkpoint["model"], strict=False)
            print("Loaded vision encoder weights from full model")
        
        # Freeze parameters
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
            
        model.eval()
        return model
    
    def load_depth_decoder(self):
        """Load the trained depth decoder"""
        print(f"Loading depth decoder from {self.checkpoint_path}")
        
        decoder = DepthDecoder(
            latent_size=self.cfg.model.rep_size,
            num_layers=self.cfg.model.num_layers_enc
        ).to(self.device)
        
        checkpoint = load_checkpoint(self.checkpoint_path, self.device)
        
        if "depth_decoder" in checkpoint:
            decoder.load_state_dict(checkpoint["depth_decoder"])
            print("Successfully loaded depth decoder")
        else:
            raise ValueError("Checkpoint does not contain depth_decoder weights")
        
        decoder.eval()
        return decoder
    
    def preprocess_thermal_image(self, image_path):
        """Preprocess thermal image using the exact same steps as in the dataloader"""
        # Read image
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Failed to load image from {image_path}")
            img = np.zeros((256, 256), dtype=np.uint8)
        
        # Resize to the expected input size
        input_size = self.cfg.dataset.get('img_size', 256)
        img = cv2.resize(img, (input_size, input_size))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        
        # Apply the same preprocessing as in TraversabilityDataset.preprocess_thermal:
        # 1. Standardize
        img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-6)
        
        # 2. Clip values
        img_tensor = torch.clip(img_tensor, min=-3, max=2)
        
        # 3. Normalize to [0,1]
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-6)
        
        # 4. Add batch and channel dimensions
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
        
        return img_tensor, img
    
    def set_depth_gt_folder(self, depth_gt_folder):
        """Set ground truth depth folder for evaluation"""
        self.depth_gt_folder = Path(depth_gt_folder)
        if not self.depth_gt_folder.exists():
            print(f"Warning: Ground truth depth folder {depth_gt_folder} does not exist")
            self.depth_gt_folder = None
    
    def run_inference(self):
        """Run inference on thermal images in the input folder"""
        print(f"Starting inference on {len(self.thermal_files)} thermal images")
        
        with torch.no_grad():
            for i, thermal_path in enumerate(tqdm(self.thermal_files, desc="Processing")):
                # Preprocess thermal image
                thermal_tensor, thermal_np = self.preprocess_thermal_image(thermal_path)
                thermal_tensor = thermal_tensor.to(self.device)
                
                # Get encoder features
                v_encoded_thermal, thermal_features = self.pretrained_model.vision_encoder(thermal_tensor, return_features=True)
                
                # Get depth prediction
                depth_pred = self.depth_decoder(v_encoded_thermal, thermal_features)
                
                # Get thmono pred
                thermal_mono_pred = self.thermalMonoDepth.infer_image(thermal_np)

                # Convert prediction to numpy
                depth_pred_np = depth_pred[0, 0].cpu().numpy()
                
                # Load ground truth depth if available
                depth_gt_np = None
                if self.depth_gt_folder:
                    depth_gt_path = self.depth_gt_folder / f"{thermal_path.stem}_depth.png"
                    if depth_gt_path.exists():
                        depth_gt = cv2.imread(str(depth_gt_path), cv2.IMREAD_UNCHANGED)
                        depth_gt = cv2.resize(depth_gt, (thermal_np.shape[1], thermal_np.shape[0]))
                        depth_gt_np = depth_gt.astype(np.float32) / 255.0
                        
                        # Convert to tensor for metrics calculation
                        depth_gt_tensor = torch.from_numpy(depth_gt_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
                        
                        # Calculate metrics
                        _, batch_metrics = self.criterion(depth_pred, depth_gt_tensor, isTraining=False)
                        
                        # Store metrics
                        for k, v in batch_metrics.items():
                            if k in self.metrics:
                                self.metrics[k].append(v)
                
                # Save visualization
                self.save_visualization(thermal_np, depth_pred_np, thermal_mono_pred, depth_gt_np, thermal_path.stem, i)
        
        # Calculate and report average metrics if ground truth was available
        if self.depth_gt_folder and any(len(v) > 0 for v in self.metrics.values()):
            self.report_metrics()
    
    def save_visualization(self, thermal_np, depth_pred_np, thermal_mono_pred, depth_gt_np, image_name, idx):
        """Save visualization of input, ground truth (if available) and prediction"""
        # Normalize depth prediction for visualization
        depth_vis = depth_pred_np.copy()
        
        # Create figure for visualization
        if depth_gt_np is not None:
            # We have ground truth, so create 3-panel visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 4, 1)
            plt.title("Thermal Input")
            plt.imshow(thermal_np, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 4, 2)
            plt.title("Ground Truth Depth")
            plt.imshow(depth_gt_np, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 4, 3)
            plt.title("Predicted Depth")
            plt.imshow(depth_vis, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.title("ThMOnoDepth Depth")
            plt.imshow(thermal_mono_pred, cmap='gray')
            plt.axis('off')
        else:
            # No ground truth, so create 2-panel visualization
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 3, 1)
            plt.title("Thermal Input")
            plt.imshow(thermal_np, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.title("Predicted Depth")
            plt.imshow(depth_vis, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("ThMOnoDepth Depth")
            plt.imshow(thermal_mono_pred, cmap='gray')
            plt.axis('off')
        
        plt.tight_layout(pad=1.5)
        
        # Save figure
        save_path = self.output_dir / f"{image_name}_result.png"
        plt.savefig(save_path)
        plt.close()
        
        # Also save individual depth prediction as image
        depth_pred_img = (depth_vis * 255).astype(np.uint8)
        cv2.imwrite(str(self.output_dir / f"{image_name}_depth.png"), depth_pred_img)
    
    def report_metrics(self):
        """Calculate and report average metrics"""
        avg_metrics = {k: np.mean(v) for k, v in self.metrics.items() if len(v) > 0}
        
        print("\n" + "="*50)
        print("DEPTH ESTIMATION RESULTS SUMMARY")
        print("="*50)
        if 'abs_rel' in avg_metrics:
            print(f"Abs Rel Error:  {avg_metrics['abs_rel']:.4f}")
        if 'rmse' in avg_metrics:
            print(f"RMSE:           {avg_metrics['rmse']:.4f}")
        if 'delta1' in avg_metrics:
            print(f"Delta < 1.25:   {avg_metrics['delta1']:.4f}")
        if 'delta2' in avg_metrics:
            print(f"Delta < 1.25²:  {avg_metrics['delta2']:.4f}")
        if 'delta3' in avg_metrics:
            print(f"Delta < 1.25³:  {avg_metrics['delta3']:.4f}")
        print("="*50)
        
        # Save metrics to file
        metrics_file = self.output_dir / "depth_estimation_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(f"Test Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Test Samples: {len([v for v in self.metrics['abs_rel'] if v is not None])}\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n\n")
            f.write("="*50 + "\n")
            f.write("DEPTH ESTIMATION RESULTS SUMMARY\n")
            f.write("="*50 + "\n")
            if 'abs_rel' in avg_metrics:
                f.write(f"Abs Rel Error:  {avg_metrics['abs_rel']:.4f}\n")
            if 'rmse' in avg_metrics:
                f.write(f"RMSE:           {avg_metrics['rmse']:.4f}\n")
            if 'delta1' in avg_metrics:
                f.write(f"Delta < 1.25:   {avg_metrics['delta1']:.4f}\n")
            if 'delta2' in avg_metrics:
                f.write(f"Delta < 1.25²:  {avg_metrics['delta2']:.4f}\n")
            if 'delta3' in avg_metrics:
                f.write(f"Delta < 1.25³:  {avg_metrics['delta3']:.4f}\n")
            f.write("="*50 + "\n")
        
        print(f"Metrics saved to {metrics_file}")

def main():
    parser = argparse.ArgumentParser(description='Test Depth Estimation on Thermal Images from folder')
    parser.add_argument('--config', type=str, default='./conf/config_depth', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to depth decoder checkpoint')
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing thermal images')
    parser.add_argument('--depth_gt_folder', type=str, default=None, help='Optional folder with ground truth depth maps')
    parser.add_argument('--output_folder', type=str, default='depth_test_results', help='Directory to save test results')
    
    args = parser.parse_args()
    
    # Create tester and run inference
    tester = SimplifiedDepthTester(
        args.config, 
        args.checkpoint, 
        args.input_folder, 
        args.output_folder
    )
    
    # Set ground truth folder if provided
    if args.depth_gt_folder:
        tester.set_depth_gt_folder(args.depth_gt_folder)
    
    tester.run_inference()

if __name__ == "__main__":
    main()

