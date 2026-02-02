import gc
from pathlib import Path
from datetime import datetime
import sys
import os
import argparse

try:
    sys.path.append(str(Path(".").resolve()))
except:
    raise RuntimeError("Can't append root directory of the project to the path")

from rich import print
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from comet_ml.integration.pytorch import log_model, watch
import cv2

from model.m2p2_dataloader import TronDataset
from model.m2p2_model import VisionEncoder, DepthDecoder, TronModel, DepthLoss
from model.depthv2_model import load_DepthAnythingv2_model
from model.thMonoDepth_model_load import load_ThermalMonoDepth_model, compute_depth_errors
from utils.nn import check_grad_norm, init_weights
from utils.io import save_checkpoint, load_checkpoint
from utils.helpers import get_conf, init_logger, init_device, timeit

matplotlib.use('Agg')

class DepthDecoderTrainer:
    def __init__(self, cfg_dir: str):
        # Load config and initialize the logger and device
        self.cfg = get_conf(cfg_dir)
        self.cfg.directory.model_name = self.cfg.train_params.experiment_name
        self.cfg.directory.model_name += f"-{self.cfg.model.rep_size}-{datetime.now():%m-%d-%H-%M}"
        self.cfg.train_params.experiment_name = self.cfg.directory.model_name
        self.cfg.directory.save = str(
            Path(self.cfg.directory.save) / self.cfg.directory.model_name
        )
        self.logger = init_logger(self.cfg)
        self.device = init_device(self.cfg)
        
        # Create dataloaders
        self.train_data = self.init_dataloader(split='train')
        self.val_data = self.init_dataloader(split='validation')
        self.logger.log_parameters(
            {"train_len": len(self.train_data), "val_len": len(self.val_data)}
        )
        
        # Load pretrained vision encoder
        self.pretrained_model = self.load_pretrained_model()
        self.pretrained_model.eval()  # Set to evaluation mode
        
        # Initialize depth decoder
        self.depth_decoder = DepthDecoder(
            latent_size=self.cfg.model.rep_size, 
            num_layers=self.cfg.model.num_layers_enc
        ).to(self.device)
        self.depth_decoder.apply(init_weights(**self.cfg.init_weights))
        watch(self.depth_decoder)
        
        # Load Depth Anything v2 for comparison
        self.depthv2_model, _ = load_DepthAnythingv2_model(**self.cfg.depthAnythingV2)
        print("loaded Depth Anything v2 model")
        self.thermalMonoDepth = load_ThermalMonoDepth_model(**self.cfg.thMonoLoad)
        print("loaded ThermalMonoDepth model")

        # if self.cfg.baseline_eval.eval_model:
        #     print(f"\n{'='*20} Evaluating ThermalMonoDepth Baseline {'='*20}")
        #     self.evaluate_thermal_mono_depth()
        #     print(f"{'='*60}\n")
        #     exit()

        # Loss for depth reconstruction
        self.recon_criterion = DepthLoss()
        
        # Initialize optimizer for depth decoder
        self.optimizer, self.scheduler = self.init_optimizer()
        
        # Training variables
        self.best_recon_loss = float('inf')
        self.epoch = 1
        self.iteration = 0
        self.recon_losses = []
        self.recon_val_loss = []
        self.loss_interval = 5
        
        # Resume from checkpoint if needed
        self.if_resume()


    def train(self):
        """Trains the depth decoder using the pretrained vision encoder"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Starting Depth Decoder training!")
        
        for epoch in range(self.epoch, self.cfg.train_params.epochs + 1):
            self.epoch = epoch
            self.depth_decoder.train()
            
            running_recon_loss = []
            
            bar = tqdm(
                self.train_data,
                desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training Depth Decoder: ",
            )
            
            for data in bar:
                self.iteration += 1
                recon_loss = self.decode_batch(data)
                running_recon_loss.append(recon_loss)
                
                bar.set_postfix(recon_loss=recon_loss)
                
                # Visualize reconstruction periodically
                if self.iteration % 100 == 0:
                    self.visualize_reconstruction(data)
            
            bar.close()
            
            # Validate decoder
            val_recon_loss, val_metrics = self.validate()
            print(f"Validation Reconstruction Loss: {val_recon_loss:.4f}")
            
            # Update scheduler based on validation loss
            if self.scheduler is not None:
                self.scheduler.step(val_recon_loss)
            
            # Log metrics
            avg_recon_loss = np.mean(running_recon_loss)
            if self.epoch % self.loss_interval == 0:
                self.recon_losses.append(avg_recon_loss)
                self.recon_val_loss.append(val_recon_loss)
            
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                f"Iteration {self.iteration:05} summary: Recon Loss: {avg_recon_loss:.4f}"
            )
            
            self.logger.log_metric("train/decoder/epoch_recon_loss", avg_recon_loss, step=self.epoch)
            
            # Save best model  elif (self.e_loss[-1] < self.best * 0.95 and
            if val_recon_loss < self.best_recon_loss * 0.98:
                self.best_recon_loss = val_recon_loss
                self.save(save_best_only=True)
            
            # Regular save
            if self.epoch % self.cfg.train_params.save_every == 0:
                self.save(save_best_only=False)
            
            gc.collect()
        
        self.plot_losses()
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Depth Decoder training is DONE!")

    def decode_batch(self, data):
        """Forward pass through the depth decoder"""
        self.depth_decoder.train()
        
        # Move data to device
        # thermal, depth, _, _ = data
        thermal, depth = data
        thermal = thermal.to(self.device)
        depth = depth.to(self.device)
        
        # Get encoder output 
        with torch.no_grad():
            v_encoded_thermal, thermal_features = self.pretrained_model.vision_encoder(thermal, return_features=True)
        
        # Forward pass through decoder
        depth_recon = self.depth_decoder(v_encoded_thermal, thermal_features)
        
        # Compute loss
        loss = self.recon_criterion(depth_recon, depth, isTraining=True)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.cpu().item()

    def validate(self):
        """Validate the depth decoder"""
        self.depth_decoder.eval()
        running_recon_val_loss = []
        running_metrics = {'abs_rel': [], 'rmse': [], 'delta1': [], 'delta2': [], 'delta3': []}
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.val_data, desc=f"Epoch {self.epoch:03}, validating...")):
                # thermal, depth, _, _ = data
                thermal, depth = data
                thermal = thermal.to(self.device)
                depth = depth.to(self.device)
                
                # Get encoder output
                v_encoded_thermal, thermal_features = self.pretrained_model.vision_encoder(thermal, return_features=True)
                
                # Get depth reconstruction
                depth_recon = self.depth_decoder(v_encoded_thermal, thermal_features)
                
                # Compute loss and metrics
                recon_loss, metrics = self.recon_criterion(depth_recon, depth, isTraining=False)
                
                running_recon_val_loss.append(recon_loss.cpu().item())
                
                for k, v in metrics.items():
                    running_metrics[k].append(v)
                
                # Visualize validation samples periodically
                if i % 10 == 0:
                    self.visualize_validation(thermal, depth, depth_recon, i)
        
        # Calculate average metrics
        avg_metrics = {k: np.mean(v) for k, v in running_metrics.items()}
        
        # Log metrics
        print(f"Validation Metrics - Abs Rel: {avg_metrics['abs_rel']:.4f}, "
              f"RMSE: {avg_metrics['rmse']:.4f}, "
              f"Delta1: {avg_metrics['delta1']:.4f}, "
              f"Delta2: {avg_metrics['delta2']:.4f}, "
              f"Delta3: {avg_metrics['delta3']:.4f}")
        
        self.logger.log_metric("val/decoder/epoch_recon_loss", np.mean(running_recon_val_loss), step=self.epoch)
        self.logger.log_metric("val/metrics/abs_rel", avg_metrics['abs_rel'], step=self.epoch)
        self.logger.log_metric("val/metrics/rmse", avg_metrics['rmse'], step=self.epoch)
        self.logger.log_metric("val/metrics/delta1", avg_metrics['delta1'], step=self.epoch)
        self.logger.log_metric("val/metrics/delta2", avg_metrics['delta2'], step=self.epoch)
        self.logger.log_metric("val/metrics/delta3", avg_metrics['delta3'], step=self.epoch)
        
        return np.mean(running_recon_val_loss), avg_metrics

    def visualize_reconstruction(self, data):
        """Visualize training reconstructions"""
        with torch.no_grad():
            # thermal, depth, _, _ = data
            thermal, depth = data
            thermal = thermal.to(self.device)
            
            # Get encoder output
            v_encoded_thermal, thermal_features = self.pretrained_model.vision_encoder(thermal, return_features=True)
            
            # Get depth reconstruction
            depth_recon = self.depth_decoder(v_encoded_thermal, thermal_features)
            
            # Move to CPU and convert to numpy for visualization
            depth_recon_np = depth_recon.cpu().numpy()
            thermal_np = thermal.cpu().numpy()
            depth_np = depth.cpu().numpy()
            
            # Visualize first image in batch
            recon_img = depth_recon_np[0, 0, :, :]
            thermal_img = thermal_np[0, 0, :, :]
            depth_img = depth_np[0, 0, :, :]
            
            # Denormalize depths (assuming 0-1 normalization with max depth of 20m)
            depth_denorm = depth_img * 30.0
            recon_depth_denorm = recon_img * 30.0
            
            # Create visualization directory
            os.makedirs('depth_recon', exist_ok=True)
            save_dir = f'depth_recon/depth_reconstruction_epoch_{self.epoch:03}_iter_{self.iteration:05}.png'
            
            # Plot visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.title("Thermal Input")
            plt.imshow(thermal_img, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth Depth")
            plt.imshow(depth_denorm, cmap='magma')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.title("Predicted Depth")
            plt.imshow(recon_depth_denorm, cmap='magma')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_dir)
            plt.close()
            
            # Log to logger
            with self.logger.context_manager("train_images"):
                self.logger.log_image(
                    save_dir, 
                    name=f"train_depth_recon_epoch_{self.epoch:03}_iter_{self.iteration:05}", 
                    step=self.epoch
                )

    def visualize_validation(self, thermal, depth, depth_recon, batch_idx):
        """Visualize validation reconstructions with Depth Anything v2 comparison"""
        # Convert tensors to numpy arrays
        depth_recon_np = depth_recon.cpu().numpy()
        thermal_np = thermal.cpu().numpy()
        depth_np = depth.cpu().numpy()
        
        # Get first image in batch
        recon_img = depth_recon_np[0, 0, :, :]
        thermal_img = thermal_np[0, 0, :, :]
        depth_img = depth_np[0, 0, :, :]
        
        # Perform Depth Anything v2 inference
        thermal_rgb = cv2.cvtColor((thermal_img*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        depthv2_pred = self.depthv2_model.infer_image(thermal_rgb)

        thermal_mono_pred = self.thermalMonoDepth.infer_image(thermal_img)
        
        # Create visualization directory
        os.makedirs('validation_vis', exist_ok=True)
        save_dir = f'validation_vis/val_depth_comparison_epoch{self.epoch}_batch{batch_idx}.png'
        
        # Plot visualization
        plt.figure(figsize=(15, 8))
        
        plt.subplot(1, 5, 1)
        plt.title("Thermal Input")
        plt.imshow(thermal_img, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 5, 2)
        plt.title("Ground Truth Depth")
        plt.imshow(depth_img, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 5, 3)
        plt.title("Our Depth Prediction")
        plt.imshow(recon_img, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 5, 4)
        plt.title("Depth Anything v2 Prediction")
        plt.imshow(depthv2_pred, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 5, 5)
        plt.title("Thermal Mono Depth Prediction")
        plt.imshow(thermal_mono_pred, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir)
        plt.close()
        
        # Log to logger
        with self.logger.context_manager("validation_images"):
            self.logger.log_image(
                save_dir, 
                name=f"val_depth_recon_epoch_{self.epoch:03}_batch_{batch_idx}", 
                step=self.epoch
            )
            

    def load_pretrained_model(self):
        """Load the pretrained TRON model"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Loading pretrained model from {self.cfg.directory.pretrained_path}")
        
        # Initialize model architecture
        vision_encoder = VisionEncoder(
            latent_size=self.cfg.model.rep_size, 
            num_layers=self.cfg.model.num_layers_enc
        ).to(self.device)
        
        # Create model instance
        model = TronModel(
            vision_encoder=vision_encoder,
            # imu_encoder=None,
            projector=None,  # We don't need the projector for inference
            latent_size=self.cfg.model.rep_size
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = load_checkpoint(self.cfg.directory.pretrained_path, self.device)
        
        # Load vision encoder weights
        if "vision_encoder" in checkpoint:
            vision_encoder.load_state_dict(checkpoint["vision_encoder"])
            print("Loaded vision encoder weights directly")
        else:
            # If vision encoder weights aren't separately stored, load from full model
            model.load_state_dict(checkpoint["model"], strict=False)
            print("Loaded vision encoder weights from full model")
        
        # Freeze the encoder parameters
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
            
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Successfully loaded pretrained model")
        return model
    
    def evaluate_thermal_mono_depth(self):
        """Evaluates the pre-loaded ThermalMonoDepth model on the validation set."""
        # self.thermalMonoDepth.eval() # Model already loaded in eval mode
        errors = []
        ratios = []

        min_depth = 1e-3 # Standard min depth, adjust if needed
        max_depth = self.cfg.thMonoLoad.max_depth

        # <<< ADDED: Counter for logging images >>>
        logged_images_count = 0
        max_images_to_log = 5 # Log first image of first 5 batches

        print(f"Using Min Depth: {min_depth}, Max Depth: {max_depth} for evaluation")

        with torch.no_grad():
            pbar = tqdm(self.val_data, desc="Evaluating ThermalMonoDepth")
            for batch_idx, data in enumerate(pbar):
                thermal_batch, depth_batch = data

                thermal_batch_np = thermal_batch.cpu().numpy()
                depth_batch_np = depth_batch.cpu().numpy() * max_depth # Convert GT back to metric

                for i in range(thermal_batch_np.shape[0]):
                    thermal_img = thermal_batch_np[i, 0, :, :]
                    gt_depth = depth_batch_np[i, 0, :, :]
                    gt_height, gt_width = gt_depth.shape[:2]

                    pred_depth = self.thermalMonoDepth.infer_image(thermal_img)

                    pred_inv_depth = 1 / (pred_depth + 1e-6)
                    pred_inv_depth_resized = cv2.resize(pred_inv_depth, (gt_width, gt_height), interpolation=cv2.INTER_NEAREST)
                    pred_depth_resized = 1 / (pred_inv_depth_resized + 1e-6)

                    mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
                    val_pred_depth_masked = pred_depth_resized[mask]
                    val_gt_depth_masked = gt_depth[mask]

                    if val_gt_depth_masked.shape[0] == 0 or val_pred_depth_masked.shape[0] == 0:
                        # Don't print warning for every skipped sample, too verbose
                        # print(f"Warning: No valid pixels for evaluation in sample {batch_idx * self.cfg.dataloader.batch_size + i}. Skipping.")
                        continue

                    median_pred = np.median(val_pred_depth_masked)
                    median_gt = np.median(val_gt_depth_masked)

                    if median_pred <= 0 or median_gt <= 0:
                         # Don't print warning for every skipped sample, too verbose
                        # print(f"Warning: Invalid median found (pred={median_pred}, gt={median_gt}) in sample {batch_idx * self.cfg.dataloader.batch_size + i}. Skipping median scaling for this sample.")
                        ratio = 1.0
                    else:
                        ratio = median_gt / median_pred

                    ratios.append(ratio)
                    # <<< MODIFIED: Store scaled full prediction for potential logging >>>
                    scaled_pred_depth = pred_depth_resized * ratio

                    # Apply mask *after* scaling for error calculation
                    val_pred_depth_scaled_masked = scaled_pred_depth[mask]

                    val_pred_depth_scaled_masked[val_pred_depth_scaled_masked < min_depth] = min_depth
                    val_pred_depth_scaled_masked[val_pred_depth_scaled_masked > max_depth] = max_depth

                    img_errors = compute_depth_errors(val_gt_depth_masked, val_pred_depth_scaled_masked)
                    if not np.isnan(img_errors[0]):
                        errors.append(img_errors)

                    # <<< ADDED: Logging block >>>
                    if logged_images_count < max_images_to_log and i == 0: # Log first image of the batch
                        try:
                            fig, axes = plt.subplots(1, 3, figsize=(18, 5)) # Increased figure size

                            # Plot Thermal Input
                            axes[0].imshow(thermal_img, cmap='gray')
                            axes[0].set_title(f"Thermal Input (Batch {batch_idx})")
                            axes[0].axis('off')

                            # Plot Ground Truth Depth
                            im1 = axes[1].imshow(gt_depth, cmap='gray', vmin=min_depth, vmax=max_depth)
                            axes[1].set_title("Ground Truth Depth (Metric)")
                            axes[1].axis('off')
                            # fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04) # Add colorbar

                            # Plot Scaled ThermalMonoDepth Prediction
                            im2 = axes[2].imshow(scaled_pred_depth, cmap='gray', vmin=min_depth, vmax=max_depth)
                            axes[2].set_title("ThermalMonoDepth Pred (Scaled)")
                            axes[2].axis('off')
                            # fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04) # Add colorbar

                            plt.tight_layout()
                            temp_img_path = f"./thermal_mono_eval_batch_{batch_idx}.png"
                            plt.savefig(temp_img_path)
                            plt.close(fig) # Close the figure to free memory

                            # Log to Comet ML
                            with self.logger.context_manager("thermal_mono_depth_baseline_eval"):
                                self.logger.log_image(
                                    temp_img_path,
                                    name=f"thermal_mono_eval_batch_{batch_idx}",
                                    step=0 # Use step 0 for baseline eval
                                )
                            logged_images_count += 1
                            # Optional: remove temp file after logging
                            # os.remove(temp_img_path)

                        except Exception as e:
                            print(f"Error generating/logging validation image: {e}")
                            plt.close('all') # Ensure all figures are closed on error

        # --- Aggregate and print results ---
        if not errors:
            print("\n--- ThermalMonoDepth Evaluation Results ---")
            print("No valid samples found for evaluation.")
            print("-------------------------------------------")
            return

        errors = np.array(errors)
        mean_errors = np.nanmean(errors, axis=0)

        ratios = np.array(ratios)
        med_ratio = np.median(ratios)
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)

        print("\n--- ThermalMonoDepth Evaluation Results ---")
        print(f" Evaluated on {len(errors)} validation samples.")
        print(f" Scaling ratios | Median: {med_ratio:0.3f} | Mean: {mean_ratio:0.3f} | Std: {std_ratio:0.3f}")
        print("\n Metrics (after median scaling):")
        print(("-"*49))
        print(("| {:>8} | {:>8} | {:>8} | {:>8} |").format("abs_rel", "sq_rel", "rmse", "rmse_log"))
        print(("| {: 8.3f} | {: 8.3f} | {: 8.3f} | {: 8.3f} |").format(*mean_errors[0:4]))
        print(("-"*49))
        print(("| {:>8} | {:>8} | {:>8} |").format("a1", "a2", "a3"))
        print(("| {: 8.3f} | {: 8.3f} | {: 8.3f} |").format(*mean_errors[4:7]))
        print(("-"*36))
        print("-------------------------------------------")

    
    
    def init_optimizer(self):
        """Initialize optimizer and scheduler for depth decoder"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Initializing optimizer and scheduler for depth decoder")
        
        # Optimizer
        if self.cfg.train_params.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(self.depth_decoder.parameters(), **self.cfg.depth_decoder_adamw)
        elif self.cfg.train_params.optimizer.lower() == "rmsprop":
            optimizer = optim.RMSprop(self.depth_decoder.parameters(), **self.cfg.rmsprop)
        elif self.cfg.train_params.optimizer.lower() == "sgd":
            optimizer = optim.SGD(self.depth_decoder.parameters(), **self.cfg.sgd)
        else:
            raise ValueError(f"Unknown optimizer {self.cfg.train_params.optimizer}")
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            threshold=0.001,
            threshold_mode='rel',
            cooldown=2,
            min_lr=self.cfg.scheduler.final_lr,
            eps=1e-8
        )
        
        return optimizer, scheduler

    def init_dataloader(self, split='train'):
        """Initialize dataloaders"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Initializing {split} dataloader")
        
        dataset = TronDataset(**self.cfg.dataset, split=split)
        dataloader = DataLoader(dataset, **self.cfg.dataloader)
        
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {split.capitalize()} dataset has {len(dataset)} samples")
        
        return dataloader
    
    def if_resume(self):
        """Resume training from checkpoint if needed"""
        if self.cfg.train_params.resume_decoder:
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Loading depth decoder checkpoint")
            
            checkpoint = load_checkpoint(self.cfg.directory.load_decoder, self.device)
            
            self.depth_decoder.load_state_dict(checkpoint["depth_decoder"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epoch = checkpoint["epoch"] + 1
            self.iteration = checkpoint["iteration"] + 1
            self.best_recon_loss = checkpoint["best_recon_loss"]
            self.recon_losses = checkpoint.get("recon_losses", [])
            self.recon_val_loss = checkpoint.get("recon_val_loss", [])
            
            if self.scheduler is not None and "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
                
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Successfully resumed from epoch {self.epoch-1}")

    def save(self, save_best_only=False, name=None):
        """Save model checkpoint"""
        save_path = Path(self.cfg.directory.save)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "time": str(datetime.now()),
            "depth_decoder": self.depth_decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
            "epoch": self.epoch,
            "iteration": self.iteration,
            "best_recon_loss": self.best_recon_loss,
            "recon_losses": self.recon_losses,
            "recon_val_loss": self.recon_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        
        if name is None:
            save_name = f"depth_decoder_epoch_{self.epoch:03d}"
        else:
            save_name = name
        
        # Call save_checkpoint with all required arguments
        if save_best_only:
            save_checkpoint(checkpoint, True, str(save_path), "best_model")
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Saved BEST model with loss {self.best_recon_loss:.4f}")
        else:
            save_checkpoint(checkpoint, False, str(save_path), save_name)
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Saved checkpoint at epoch {self.epoch}")
        
        # Log model to comet
        log_model(self.logger, self.depth_decoder, model_name=f"depth_decoder_epoch_{self.epoch:03d}")



    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, self.epoch, self.loss_interval), self.recon_losses, label='Training Loss')
        plt.plot(range(0, self.epoch, self.loss_interval), self.recon_val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Depth Decoder Training Progress')
        plt.legend()
        plt.grid(True)
        
        save_path = Path(self.cfg.directory.save) / "loss_plot.png"
        plt.savefig(save_path)
        plt.close()
        
        # Log to logger
        self.logger.log_image(str(save_path), name="depth_decoder_loss_plot")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="./conf/config_depth", type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    learner = DepthDecoderTrainer(cfg_path)
    learner.train()