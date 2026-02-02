import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
from datetime import datetime

# Import your project modules (adjust paths as needed)
from model.traversability_model import TraversabilityDecoder, TraversabilityLoss
from model.m2p2_model import VisionEncoder, TronModel
from model.traverse_dataloader import TraversabilityDataset
from utils.io import load_checkpoint
from utils.helpers import get_conf, init_device

class TraversabilityTester:
    def __init__(self, cfg_path, checkpoint_path):
        """Initialize the tester with configuration and checkpoint paths"""
        self.cfg = get_conf(cfg_path)
        self.device = init_device(self.cfg)
        self.checkpoint_path = checkpoint_path
        
        # Create output directory for results
        self.output_dir = Path(self.cfg.directory.get('test_output', 'test_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize test dataloader
        self.test_data = self.init_dataloader(split='validation')
        print(f"Test dataset loaded with {len(self.test_data.dataset)} samples")
        
        # Load models
        self.pretrained_model = self.load_pretrained_model()
        self.traversability_decoder = self.load_traversability_decoder()
        
        # Initialize loss function for metrics calculation
        loss_weights = self.cfg.get('traversability_loss_weights', {'bce': 1.0, 'dice': 1.0})
        self.criterion = TraversabilityLoss(weights=loss_weights).to(self.device)
        
        # Initialize metrics storage
        self.metrics = {
            'loss': [],
            'iou': [],
            'f1': [],
            'precision': [],
            'recall': []
        }
    
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
    
    def load_traversability_decoder(self):
        """Load the trained traversability decoder"""
        print(f"Loading traversability decoder from {self.checkpoint_path}")
        
        decoder = TraversabilityDecoder(
            latent_size=self.cfg.model.rep_size,
            num_layers=self.cfg.model.num_layers_enc
        ).to(self.device)
        
        checkpoint = load_checkpoint(self.checkpoint_path, self.device)
        
        if "traversability_decoder" in checkpoint:
            decoder.load_state_dict(checkpoint["traversability_decoder"])
            print("Successfully loaded traversability decoder")
        else:
            raise ValueError("Checkpoint does not contain traversability_decoder weights")
        
        decoder.eval()
        return decoder
    
    def init_dataloader(self, split='test'):
        """Initialize test dataloader"""
        dataset = TraversabilityDataset(**self.cfg.dataset, split=split)
        dataloader = DataLoader(dataset, 
                                batch_size=self.cfg.test_params.get('batch_size', 8),
                                shuffle=False,
                                num_workers=self.cfg.dataloader.get('num_workers', 4))
        return dataloader
    
    def run_test(self):
        """Run inference on test data and calculate metrics"""
        print(f"Starting test inference on {len(self.test_data.dataset)} samples")
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_data, desc="Testing")):
                thermal, mask_gt = data
                thermal = thermal.to(self.device)
                mask_gt = mask_gt.to(self.device)
                
                # Get encoder features
                v_encoded_thermal, thermal_features = self.pretrained_model.vision_encoder(thermal, return_features=True)
                
                # Get traversability prediction
                predicted_mask = self.traversability_decoder(v_encoded_thermal, thermal_features)
                
                # Calculate loss and metrics
                loss, batch_metrics = self.criterion(predicted_mask, mask_gt, isTraining=False)
                
                # Store metrics
                self.metrics['loss'].append(loss.item())
                for k, v in batch_metrics.items():
                    if k in self.metrics:
                        self.metrics[k].append(v)
                
                # Save visualizations
                self.save_visualization(thermal, mask_gt, predicted_mask, i)
        
        # Calculate and report average metrics
        self.report_metrics()
    
    def save_visualization(self, thermal, mask_gt, predicted_mask, batch_idx):
        """Save visualization of input, ground truth and prediction"""
        # Process up to 4 samples per batch to avoid too many images
        num_samples = min(thermal.size(0), 4)
        
        for i in range(num_samples):
            # Convert tensors to numpy for visualization
            thermal_np = thermal[i, 0].cpu().numpy()
            mask_gt_np = mask_gt[i, 0].cpu().numpy()
            pred_mask_np = predicted_mask[i, 0].cpu().numpy()
            
            # Create figure
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.title("Thermal Input")
            plt.imshow(thermal_np, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth Mask")
            plt.imshow(mask_gt_np, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask")
            plt.imshow(pred_mask_np, cmap='gray')
            plt.axis('off')
            
            plt.tight_layout(pad=1.5)
            
            # Save figure
            save_path = self.output_dir / f"test_sample_{batch_idx}_{i}.png"
            plt.savefig(save_path)
            plt.close()
            
            # Optionally save individual images
            if self.cfg.test_params.get('save_individual_images', False):
                # Save thermal
                plt.figure(figsize=(5, 5))
                plt.imshow(thermal_np, cmap='gray')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(self.output_dir / f"thermal_{batch_idx}_{i}.png")
                plt.close()
                
                # Save ground truth
                plt.figure(figsize=(5, 5))
                plt.imshow(mask_gt_np, cmap='gray')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(self.output_dir / f"gt_{batch_idx}_{i}.png")
                plt.close()
                
                # Save prediction
                plt.figure(figsize=(5, 5))
                plt.imshow(pred_mask_np, cmap='gray')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(self.output_dir / f"pred_{batch_idx}_{i}.png")
                plt.close()
    
    def report_metrics(self):
        """Calculate and report average metrics"""
        avg_metrics = {k: np.mean(v) for k, v in self.metrics.items()}
        
        print("\n" + "="*50)
        print("TEST RESULTS SUMMARY")
        print("="*50)
        print(f"Average Loss:      {avg_metrics['loss']:.4f}")
        print(f"Average IoU:       {avg_metrics['iou']:.4f}")
        print(f"Average F1 Score:  {avg_metrics['f1']:.4f}")
        print(f"Average Precision: {avg_metrics['precision']:.4f}")
        print(f"Average Recall:    {avg_metrics['recall']:.4f}")
        print("="*50)
        
        # Save metrics to file
        metrics_file = self.output_dir / "test_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(f"Test Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Test Samples: {len(self.test_data.dataset)}\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n\n")
            f.write("="*50 + "\n")
            f.write("TEST RESULTS SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Average Loss:      {avg_metrics['loss']:.4f}\n")
            f.write(f"Average IoU:       {avg_metrics['iou']:.4f}\n")
            f.write(f"Average F1 Score:  {avg_metrics['f1']:.4f}\n")
            f.write(f"Average Precision: {avg_metrics['precision']:.4f}\n")
            f.write(f"Average Recall:    {avg_metrics['recall']:.4f}\n")
            f.write("="*50 + "\n")
        
        print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Traversability Estimation on Thermal Images')
    parser.add_argument('--config', type=str, default='./conf/config_trav', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to traversability decoder checkpoint')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save test results')
    
    args = parser.parse_args()
    
    # Create tester and run test
    tester = TraversabilityTester(args.config, args.checkpoint)
    tester.run_test()
