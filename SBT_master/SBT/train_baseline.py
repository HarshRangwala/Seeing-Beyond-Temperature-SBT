import gc
from pathlib import Path
from datetime import datetime
import sys
import os
import argparse
from comet_ml.integration.pytorch import log_model, watch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2

from model.m2p2_dataloader import TronDataset
from model.depth_model_baseline import UNet, DepthLoss
from model.depthv2_model import load_DepthAnythingv2_model
from utils.io import save_checkpoint, load_checkpoint
from utils.helpers import get_conf, init_logger, init_device, timeit, print_model_stats
from utils.nn import check_grad_norm, op_counter


import pickle

import matplotlib
matplotlib.use('Agg')  # Set backend first
import matplotlib.pyplot as plt


class DepthBaselineLearner:
    def __init__(self, cfg_dir: str):
        # Load config and initialize settings
        self.cfg = get_conf(cfg_dir)
        self.cfg.directory.model_name = f"depth_baseline_{datetime.now():%m-%d-%H-%M}"
        self.cfg.directory.save = str(Path(self.cfg.directory.save) / self.cfg.directory.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.logger = init_logger(self.cfg)
        self.device = init_device(self.cfg)

        # Initialize dataloaders
        self.train_loader = self.init_dataloader(split='train')
        self.val_loader = self.init_dataloader(split='validation')
        self.logger.log_parameters(
            {"train_len": len(self.train_loader), "val_len": len(self.val_loader)}
        )
        
        # Initialize model, loss and optimizer
        self.model = UNet(in_channels=1, out_channels=1).to(self.device)
        watch(self.model)
        self.logger.log_code(folder="./model")
        print_model_stats(self.model)
        self.depthv2_model, _ = load_DepthAnythingv2_model(depth_measurement='metric', encoder='vitl', dataset='vkitti', max_depth=20)

        self.criterion = DepthLoss()
        self.optimizer = self.init_optimizer()
        
        # Training variables
        self.epoch = 1
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Resume if specified
        if self.cfg.train_params.resume:
            self.load_checkpoint()

    def train(self):
        # print("\nEvaluating Depth Anything v2 baseline...")
        # depth_anything_metrics = self.validate_depth_anything()
        # # Log Depth Anything v2 metrics
        # self.logger.log_metrics({
        #     "depth_anything_abs_rel": depth_anything_metrics['abs_rel'],
        #     "depth_anything_rmse": depth_anything_metrics['rmse'],
        #     "depth_anything_delta1": depth_anything_metrics['delta1']
        # })
        print("Training UNet baseline on m2p2 dataset...")
        for epoch in range(self.epoch, self.cfg.train_params.epochs + 1):
            self.epoch = epoch                        
            train_loss = self.train_epoch()
            val_loss, epoch_abs_rel, epoch_rmse, epoch_delta1, epoch_delta2, epoch_delta3 = self.validate()
            
            # Save losses for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.logger.log_metric("train_epoch_loss", train_loss, step=epoch)  
            self.logger.log_metric("val_epoch_loss", val_loss, step=epoch)

            self.logger.log_metric("val_abs_rel", epoch_abs_rel, step=epoch)  # Log Abs Rel
            self.logger.log_metric("val_rmse", epoch_rmse, step=epoch)  # Log RMSE
            self.logger.log_metric("val_delta1", epoch_delta1, step=epoch)
            self.logger.log_metric("val_delta2", epoch_delta2, step=epoch)
            self.logger.log_metric("val_delta3", epoch_delta3, step=epoch)

            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)  # ReduceLROnPlateau needs `val_loss`
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.log_metric("learning_rate", current_lr, step=epoch)
                else:
                    self.lr_scheduler.step()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.log_metric("learning_rate", current_lr, step=epoch)
            
            # Save checkpoint if best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(is_best=True)
            
            # Regular checkpoint saving
            if epoch % self.cfg.train_params.save_every == 0:
                self.save_checkpoint(is_best=False)
            
            # Plot and save every N epochs
            if epoch % self.cfg.train_params.plot_every == 0:
                self.plot_losses()
            
            print(f"Epoch {epoch}/{self.cfg.train_params.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Epoch {epoch}/{self.cfg.train_params.epochs} - "
                f"Validation Metrics - Abs Rel: {epoch_abs_rel:.4f}, RMSE: {epoch_rmse:.4f}, "
                f"Delta1: {epoch_delta1:.4f}, Delta2: {epoch_delta2:.4f}, Delta3: {epoch_delta3:.4f}")
            
            self.logger.log_metric("train_epoch_loss", train_loss, step=epoch)
            self.logger.log_metric("val_epoch_loss", val_loss, step=epoch)
            
        self.plot_losses()

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_metrics = {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}
        running_loss = []
        # Calculate a global step (epoch * batches) for logging
        num_batches = len(self.train_loader) 

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch:03d} Training Baseline (UNet)...")
        for batch_idx, (thermal, depth, _, _) in enumerate(pbar):
            thermal = thermal.to(self.device)
            depth = depth.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(thermal)
            # output = output * 20.0
            loss = self.criterion(output, depth)

            with torch.no_grad():
                metrics = self.criterion.compute_depth_metrics(output, depth)
                running_metrics['abs_rel'] += metrics['abs_rel']
                running_metrics['rmse'] += metrics['rmse']
                running_metrics['delta1'] += metrics['delta1']
                running_metrics['delta2'] += metrics['delta2']
                running_metrics['delta3'] += metrics['delta3']

            
            loss.backward()
            self.optimizer.step()
            grad_norm = check_grad_norm(self.model)

            running_loss.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            global_step = batch_idx + (self.epoch - 1) * num_batches
            self.logger.log_metric("train_batch_loss", loss.item(), step=global_step)
            
            # Visualize predictions periodically
            if batch_idx % self.cfg.train_params.vis_every == 0:
                self.visualize_predictions(thermal, depth, output, 'train', batch = batch_idx)
        epoch_loss = np.mean(running_loss)
        epoch_abs_rel = running_metrics['abs_rel'] / num_batches
        epoch_rmse =running_metrics['rmse'] / num_batches
        epoch_delta1 = running_metrics['delta1'] / num_batches
        epoch_delta2 = running_metrics['delta2'] / num_batches
        epoch_delta3 = running_metrics['delta3'] / num_batches

        self.logger.log_metric("train_epoch_abs_rel", epoch_abs_rel, epoch=self.epoch)
        self.logger.log_metric("train_epoch_rmse", epoch_rmse, epoch=self.epoch)
        self.logger.log_metric("train_epoch_delta1", epoch_delta1, epoch=self.epoch)
        self.logger.log_metric("train_epoch_delta2", epoch_delta2, epoch=self.epoch)
        self.logger.log_metric("train_epoch_delta3", epoch_delta3, epoch=self.epoch)
                
        return epoch_loss # l1 loss

    def validate(self):
        self.model.eval()
        running_metrics = {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}
        
        running_loss = []
        num_batches = len(self.val_loader)

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch:03d} Validation")
            for batch_idx, (thermal, depth, _, _) in enumerate(pbar):
                thermal = thermal.to(self.device)
                depth = depth.to(self.device)
                
                output = self.model(thermal)
                # output = output * 20.0
                loss = self.criterion(output, depth)                
                running_loss.append(loss.item())
                metrics = self.criterion.compute_depth_metrics(output, depth)
                running_metrics['abs_rel'] += metrics['abs_rel']
                running_metrics['rmse'] += metrics['rmse']
                running_metrics['delta1'] += metrics['delta1']
                running_metrics['delta2'] += metrics['delta2']
                running_metrics['delta3'] += metrics['delta3']


                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                global_step = batch_idx + (self.epoch - 1) * num_batches
                self.logger.log_metric("val_batch_loss", loss.item(), step=global_step)
                
                # Visualize predictions periodically
                if batch_idx % self.cfg.train_params.vis_every == 0:
                    self.visualize_predictions(thermal, depth, output, 'validation', batch = batch_idx)
            
            epoch_loss = np.mean(running_loss)
            epoch_abs_rel = running_metrics['abs_rel'] / num_batches
            epoch_rmse =running_metrics['rmse'] / num_batches
            epoch_delta1 = running_metrics['delta1'] / num_batches
            epoch_delta2 = running_metrics['delta2'] / num_batches
            epoch_delta3 = running_metrics['delta3'] / num_batches

            self.logger.log_metric("val_epoch_abs_rel", epoch_abs_rel, epoch=self.epoch)
            self.logger.log_metric("val_epoch_rmse", epoch_rmse, epoch=self.epoch)
            self.logger.log_metric("val_epoch_delta1", epoch_delta1, epoch=self.epoch)
            self.logger.log_metric("val_epoch_delta2", epoch_delta2, epoch=self.epoch)
            self.logger.log_metric("val_epoch_delta3", epoch_delta3, epoch=self.epoch)
            
                    
        return epoch_loss, epoch_abs_rel, epoch_rmse, epoch_delta1, epoch_delta2, epoch_delta3

    def validate_depth_anything(self):
        """Separate validation function for Depth Anything v2"""
        print("Evaluating Depth Anything v2 on validation dataset...")
        running_metrics_depthv2 = {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}
        num_batches = len(self.val_loader)

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Depth Anything v2 Validation")
            for batch_idx, (thermal, depth, _, _) in enumerate(pbar):
                thermal = thermal.to(self.device)
                depth = depth.to(self.device)
                
                # Denormalize ground truth depth to meters (0-20)
                # depth_denorm = depth * 20.0  # Assuming max_depth=20 in dataset
                # print(f"Depth Image Min: {depth_denorm.min().item():.4f}, Max: {depth_denorm.max().item():.4f}")
                # Process entire batch
                batch_size = thermal.size(0)
                depthv2_preds = []
                
                for i in range(batch_size):
                    thermal_np = thermal[i, 0].detach().cpu().numpy()
                    thermal_rgb = cv2.cvtColor((thermal_np*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    depthv2_pred = self.depthv2_model.infer_image(thermal_rgb)  # Output in meters (0-20)
                    #print(f"Depth Image Min: {depthv2_pred.min().item():.4f}, Max: {depthv2_pred.max().item():.4f}")
                    depthv2_pred_tensor = torch.tensor(depthv2_pred, dtype=torch.float32)
                    depthv2_preds.append(depthv2_pred_tensor)
                
                # Stack predictions and move to device
                depthv2_batch = torch.stack(depthv2_preds, dim=0).unsqueeze(1).to(self.device)
                
                # Compute metrics on denormalized depth
                metrics = self.criterion.compute_depth_metrics(depthv2_batch, depth)
                running_metrics_depthv2['abs_rel'] += metrics['abs_rel']
                running_metrics_depthv2['rmse'] += metrics['rmse']
                running_metrics_depthv2['delta1'] += metrics['delta1']
                running_metrics_depthv2['delta2'] += metrics['delta2']
                running_metrics_depthv2['delta3'] += metrics['delta3']
                
                pbar.set_postfix({
                    'abs_rel': f"{metrics['abs_rel']:.4f}",
                    'rmse': f"{metrics['rmse']:.4f}",
                    'delta1': f"{metrics['delta1']:.4f}",
                    'delta2': f"{metrics['delta2']:.4f}",
                    'delta3': f"{metrics['delta3']:.4f}"
                })

        # Calculate final metrics (averaged across batches)
        final_metrics = {k: v / num_batches for k, v in running_metrics_depthv2.items()}

        # Save and print metrics
        save_path = Path(self.cfg.directory.save) / 'depth_anything_metrics.pkl'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(final_metrics, f)

        print("\nDepth Anything v2 Metrics:")
        print(f"Abs Rel: {final_metrics['abs_rel']:.4f}")
        print(f"RMSE: {final_metrics['rmse']:.4f}")
        print(f"Delta1: {final_metrics['delta1']:.4f}")
        print(f"Delta2: {final_metrics['delta2']:.4f}")
        print(f"Delta3: {final_metrics['delta3']:.4f}")

        return final_metrics

    # def visualize_predictions(self, thermal, depth, pred, phase, batch):
    #     """
    #     Visualize thermal input, ground truth depth and predicted depth
    #     Args:
    #         thermal: thermal input tensor
    #         depth: ground truth depth tensor 
    #         pred: predicted depth tensor
    #         phase: 'train' or 'val'
    #     """
    #     with torch.no_grad():
    #         # Convert tensors to numpy arrays with proper detachment
    #         thermal_np = thermal[0, 0].detach().cpu().numpy()
    #         depth_np = depth[0, 0].detach().cpu().numpy()
    #         pred_np = pred[0, 0].detach().cpu().numpy()

    #         thermal_rgb = cv2.cvtColor((thermal_np*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    #         depthv2_pred = self.depthv2_model.infer_image(thermal_rgb)

    #         # Create visualization directory
            
    #         save_dir = os.path.join("visualizations UNet", phase)
    #         os.makedirs(save_dir, exist_ok = True)
    #         save_path = os.path.join(save_dir, f"{phase}_depth_comparison_epoch{self.epoch}_batch{batch}.png")

    #         # Create figure with subplots
    #         fig = plt.figure(figsize=(15, 5))

    #         ax1 = fig.add_subplot(1, 4, 1)
    #         ax1.imshow(thermal_np, cmap='gray')
    #         ax1.set_title("Thermal")
    #         ax1.axis('off')

    #         ax2 = fig.add_subplot(1, 4, 2)
    #         ax2.imshow(depth_np, cmap='gray')
    #         ax2.set_title("Ground Truth Depth")
    #         ax2.axis('off')

    #         ax3 = fig.add_subplot(1, 4, 3)
    #         ax3.imshow(pred_np, cmap='gray')
    #         ax3.set_title("Predicted Depth")
    #         ax3.axis('off')
            
    #         ax4 = fig.add_subplot(1, 4, 4)
    #         ax4.imshow(depthv2_pred, cmap='gray')
    #         ax4.set_title("Depth Anything v2 Prediction")
    #         ax4.axis('off')

    #         plt.tight_layout()
    #         with self.logger.context_manager(phase):
    #             self.logger.log_figure(
    #             figure_name=f"{phase}_epoch_{self.epoch}_batch{batch}",
    #             figure=fig,
    #             step=self.epoch  
    #             )            
    #         plt.close()

    def visualize_predictions(self, thermal, depth, pred, phase, batch):
        """
        Visualize thermal input, ground truth depth and predicted depth
        Args:
            thermal: thermal input tensor
            depth: ground truth depth tensor 
            pred: predicted depth tensor
            phase: 'train' or 'val'
        """
        with torch.no_grad():
            # Convert tensors to numpy arrays with proper detachment
            thermal_np = thermal[0, 0].detach().cpu().numpy()
            depth_np = depth[0, 0].detach().cpu().numpy()
            pred_np = pred[0, 0].detach().cpu().numpy()

            thermal_rgb = cv2.cvtColor((thermal_np*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            depthv2_pred = self.depthv2_model.infer_image(thermal_rgb)

            # Create base visualization directory
            base_dir = "visualizations_UNet"
            
            # Create phase-specific directory (train or validation)
            phase_dir = os.path.join(base_dir, phase)
            os.makedirs(phase_dir, exist_ok=True)
            
            # Define save path with phase folder
            save_path = os.path.join(phase_dir, f"depth_comparison_epoch{self.epoch}_batch{batch}.png")

            # Create figure with subplots
            fig = plt.figure(figsize=(15, 5))

            ax1 = fig.add_subplot(1, 4, 1)
            ax1.imshow(thermal_np, cmap='gray')
            ax1.set_title("Thermal")
            ax1.axis('off')

            ax2 = fig.add_subplot(1, 4, 2)
            ax2.imshow(depth_np, cmap='gray')
            ax2.set_title("Ground Truth Depth")
            ax2.axis('off')

            ax3 = fig.add_subplot(1, 4, 3)
            ax3.imshow(pred_np, cmap='gray')
            ax3.set_title("Predicted Depth")
            ax3.axis('off')
            
            ax4 = fig.add_subplot(1, 4, 4)
            ax4.imshow(depthv2_pred, cmap='gray')
            ax4.set_title("Depth Anything v2 Prediction")
            ax4.axis('off')

            plt.tight_layout()
            
            # Save the figure to the phase-specific directory
            plt.savefig(save_path)
            
            # Log to comet.ml
            with self.logger.context_manager(phase):
                self.logger.log_figure(
                figure_name=f"{phase}_epoch_{self.epoch}_batch{batch}",
                figure=fig,
                step=self.epoch  
                )            
            plt.close()




    def init_dataloader(self, split = 'train'):
        """Initializes the dataloaders"""
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the train and val dataloaders!"
        )
        dataset = TronDataset(**self.cfg.dataset, split = split)
        data = DataLoader(dataset, **self.cfg.dataloader)
        if split == 'train':
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} - Training consists of {len(dataset)} samples."
            )
        else:
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} - Training consists of {len(dataset)} samples."
            )
        
        

        return data
    
    def init_optimizer(self):
        """Initializes the optimizer and learning rate scheduler"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the optimizer!")
        if self.cfg.train_params.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), **self.cfg.adamw)

        elif self.cfg.train_params.optimizer.lower() == "rmsprop":
            optimizer = optim.RMSprop(self.model.parameters(), **self.cfg.rmsprop)

        elif self.cfg.train_params.optimizer.lower() == "sgd":
            optimizer = optim.SGD(self.model.parameters(), **self.cfg.sgd)

        else:
            raise ValueError(
                f"Unknown optimizer {self.cfg.train_params.optimizer}"
                + "; valid optimizers are 'adam' and 'rmsprop'."
            )
        
        if self.cfg.train_params.scheduler == "StepLR":
            self.lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.train_params.scheduler_params.step_size, gamma=self.cfg.train_params.scheduler_params.gamma)
                                    
        elif self.cfg.train_params.scheduler == "ReduceLROnPlateau":
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.cfg.train_params.scheduler_params.patience, factor=self.cfg.train_params.scheduler_params.factor, verbose=True)

        elif self.cfg.train_params.scheduler == "CosineAnnealingLR":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.train_params.scheduler_params.T_max, eta_min=self.cfg.train_params.scheduler_params.eta_min)
                        
        elif self.cfg.train_params.scheduler == "ExponentialLR":
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.train_params.scheduler_params.gamma)
        
        else:
            self.lr_scheduler = None  # No scheduler if not specified

        return optimizer

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        save_dir = Path(self.cfg.directory.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            torch.save(checkpoint, save_dir / 'best_model.pth')
        torch.save(checkpoint, save_dir / f'checkpoint_epoch_{self.epoch}.pth')

    def load_checkpoint(self):
        checkpoint = torch.load(self.cfg.directory.load)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(checkpoint['epoch'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', color='blue', linestyle='-', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', color='red', linestyle='--', linewidth=2)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend(loc='upper right')

        plt.tight_layout()
        save_dir = Path(self.cfg.directory.save_loss_plot)
        plt.savefig(save_dir / 'loss_plot.png')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="./conf/config_tron.yaml", type=str)
    args = parser.parse_args()
    
    learner = DepthBaselineLearner(args.conf)
    learner.train()