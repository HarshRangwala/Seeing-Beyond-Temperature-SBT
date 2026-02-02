import gc
from pathlib import Path
from datetime import datetime
import sys
import os
import argparse

from rich import print
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from comet_ml.integration.pytorch import log_model, watch

from model.m2p2_dataloader import BCDataset
from model.BCmodel import BCModel
from model.m2p2_model import VisionEncoder, TronModel
from utils.nn import check_grad_norm, init_weights
from utils.io import save_checkpoint, load_checkpoint
from utils.helpers import get_conf, init_logger, init_device, timeit


matplotlib.use('Agg')

class BCModelTrainer:
    def __init__(self, cfg_dir: str):
        # Load config and initialize the logger and device
        self.cfg = get_conf(cfg_dir)
        self.cfg.directory.model_name = self.cfg.train_params.experiment_name
        self.cfg.directory.model_name += f"-BC-{datetime.now():%m-%d-%H-%M}"
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
        
        # Initialize BC model
        self.bc_model = BCModel(
            latent_size=self.cfg.model.rep_size,
            hidden_layers=[512, 128, 64],
            output_dim=2,
            dropout_rate=0.4,
        ).to(self.device)
        
        # Initialize weights for the MLP part
        self.bc_model.mlp.apply(init_weights(**self.cfg.init_weights))
        watch(self.bc_model)
        self.logger.log_code(folder = "./model")
        
        # Loss for command velocity prediction (MSE loss)
        self.criterion = nn.MSELoss()
        
        # Initialize optimizer for BC model
        self.optimizer, self.scheduler = self.init_optimizer()
        
        # Training variables
        self.best_val_loss = float('inf')
        self.epoch = 1
        self.iteration = 0
        self.train_losses = []
        self.val_losses = []
        self.loss_interval = 5
        
        # Resume from checkpoint if needed
        self.if_resume()

    def train(self):
        """Trains the BC model using the pretrained vision encoder"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Starting BC model training!")
        
        for epoch in range(self.epoch, self.cfg.train_params.epochs + 1):
            self.epoch = epoch
            self.bc_model.train()
            
            running_loss = []
            
            bar = tqdm(
                self.train_data,
                desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training BC model: ",
            )
            
            for data in bar:
                self.iteration += 1
                loss = self.train_batch(data)
                running_loss.append(loss)
                
                bar.set_postfix(loss=loss)
                
                # Visualize predictions periodically
                if self.iteration % 100 == 0:
                    self.visualize_predictions(data)
            
            bar.close()
            
            # Validate model
            val_loss = self.validate()
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Update scheduler based on validation loss
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            avg_train_loss = np.mean(running_loss)
            if self.epoch % self.loss_interval == 0:
                self.train_losses.append(avg_train_loss)
                self.val_losses.append(val_loss)
            
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                f"Iteration {self.iteration:05} summary: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            self.logger.log_metric("train/bc/epoch_loss", avg_train_loss, step=self.epoch)
            
            # Save best model
            if val_loss < self.best_val_loss * 0.98:
                self.best_val_loss = val_loss
                self.save(save_best_only=True)
            
            # Regular save
            if self.epoch % self.cfg.train_params.save_every == 0:
                self.save(save_best_only=False)
            
            gc.collect()
        
        self.plot_losses()
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - BC model training is DONE!")

    def train_batch(self, data):
        """Forward pass and optimization for BC model"""
        self.bc_model.train()
        
        # Move data to device
        thermal, cmd_vel_gt = data
        thermal = thermal.to(self.device)
        cmd_vel_gt = cmd_vel_gt.to(self.device)
        
        # Get encoder output (no gradient computation needed)
        with torch.no_grad():
            v_encoded_thermal, thermal_features = self.pretrained_model.vision_encoder(thermal, return_features=True)
        # Forward pass through BC model
        cmd_vel_pred = self.bc_model(v_encoded_thermal)
        
        # Compute loss
        loss = self.criterion(cmd_vel_pred, cmd_vel_gt)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.cpu().item()

    def validate(self):
        """Validate the BC model"""
        self.bc_model.eval()
        running_val_loss = []
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.val_data, desc=f"Epoch {self.epoch:03}, validating...")):
                thermal, cmd_vel_gt = data
                thermal = thermal.to(self.device)
                cmd_vel_gt = cmd_vel_gt.to(self.device)

                with torch.no_grad():
                    v_encoded_thermal, thermal_features = self.pretrained_model.vision_encoder(thermal, return_features=True)
                # Forward pass
                cmd_vel_pred = self.bc_model(v_encoded_thermal)
                
                # Compute loss
                loss = self.criterion(cmd_vel_pred, cmd_vel_gt)
                
                running_val_loss.append(loss.cpu().item())
                
                # Visualize validation samples periodically
                if i % 10 == 0:
                    self.visualize_validation(thermal, cmd_vel_gt, cmd_vel_pred, i)
        
        avg_val_loss = np.mean(running_val_loss)
        self.logger.log_metric("val/bc/epoch_loss", avg_val_loss, step=self.epoch)
        
        return avg_val_loss

    def visualize_predictions(self, data):
        """Visualize training predictions"""
        with torch.no_grad():
            thermal, cmd_vel_gt = data
            thermal = thermal.to(self.device)
            cmd_vel_gt = cmd_vel_gt.to(self.device)
            with torch.no_grad():
                v_encoded_thermal, thermal_features = self.pretrained_model.vision_encoder(thermal, return_features=True)
            # Get predictions
            cmd_vel_pred = self.bc_model(v_encoded_thermal)
            
            # Move to CPU and convert to numpy
            cmd_vel_pred_np = cmd_vel_pred.cpu().numpy()
            cmd_vel_gt_np = cmd_vel_gt.cpu().numpy()
            thermal_np = thermal.cpu().numpy()
            
            # Create visualization directory
            os.makedirs('bc_predictions', exist_ok=True)
            save_dir = f'bc_predictions/pred_epoch_{self.epoch:03}_iter_{self.iteration:05}.png'
            
            # Get sample from batch
            sample_idx = 0
            thermal_img = thermal_np[sample_idx, 0]
            gt_linear = cmd_vel_gt_np[sample_idx, 0]
            gt_angular = cmd_vel_gt_np[sample_idx, 1]
            pred_linear = cmd_vel_pred_np[sample_idx, 0]
            pred_angular = cmd_vel_pred_np[sample_idx, 1]
            
            # Plot visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Show thermal image
            ax1.imshow(thermal_img, cmap='gray')
            ax1.set_title("Thermal Input")
            ax1.axis('off')
            
            # Show command velocity comparison
            bar_width = 0.35
            x = np.arange(2)
            ax2.bar(x - bar_width/2, [gt_linear, gt_angular], bar_width, label='Ground Truth')
            ax2.bar(x + bar_width/2, [pred_linear, pred_angular], bar_width, label='Predicted')
            ax2.set_xticks(x)
            ax2.set_xticklabels(['Linear Velocity', 'Angular Velocity'])
            ax2.set_title("Command Velocity Comparison")
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(save_dir)
            plt.close(fig)
            
            # Log to logger
            with self.logger.context_manager("train_predictions"):
                self.logger.log_image(
                    save_dir, 
                    name=f"train_pred_epoch_{self.epoch:03}_iter_{self.iteration:05}", 
                    step=self.epoch
                )

    def visualize_validation(self, thermal, cmd_vel_gt, cmd_vel_pred, batch_idx):
        """Visualize validation predictions"""
        # Convert tensors to numpy arrays
        cmd_vel_pred_np = cmd_vel_pred.cpu().numpy()
        cmd_vel_gt_np = cmd_vel_gt.cpu().numpy()
        thermal_np = thermal.cpu().numpy()
        
        # Create visualization directory
        os.makedirs('validation_bc', exist_ok=True)
        save_dir = f'validation_bc/val_pred_epoch{self.epoch}_batch{batch_idx}.png'
        
        # Get sample from batch
        sample_idx = 0
        thermal_img = thermal_np[sample_idx, 0]
        gt_linear = cmd_vel_gt_np[sample_idx, 0]
        gt_angular = cmd_vel_gt_np[sample_idx, 1]
        pred_linear = cmd_vel_pred_np[sample_idx, 0]
        pred_angular = cmd_vel_pred_np[sample_idx, 1]
        # print("---GROUND TRUTH LINEAR AND ANGULAR----")
        # print([gt_linear, gt_angular])
        # print("---PREDICTED LINEAR AND ANGULAR----")
        # print([pred_linear, pred_angular])

        # Plot visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Show thermal image
        ax1.imshow(thermal_img, cmap='gray')
        ax1.set_title("Thermal Input")
        ax1.axis('off')
        
        # Show command velocity comparison
        bar_width = 0.35
        x = np.arange(2)
        ax2.bar(x - bar_width/2, [gt_linear, gt_angular], bar_width, label='Ground Truth')
        ax2.bar(x + bar_width/2, [pred_linear, pred_angular], bar_width, label='Predicted')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Linear Velocity', 'Angular Velocity'])
        ax2.set_title("Command Velocity Comparison")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir)
        plt.close(fig)
        
        # Log to logger
        with self.logger.context_manager("validation_predictions"):
            self.logger.log_image(
                save_dir, 
                name=f"val_pred_epoch_{self.epoch:03}_batch_{batch_idx}", 
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
            imu_encoder=None,
            projector=None,
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
    
    def init_optimizer(self):
        """Initialize optimizer and scheduler"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the optimizer and scheduler!")
        
        # Optimizer
        if self.cfg.train_params.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(self.bc_model.parameters(), **self.cfg.bc_model_adamw)
        elif self.cfg.train_params.optimizer.lower() == "rmsprop":
            optimizer = optim.RMSprop(self.bc_model.parameters(), **self.cfg.rmsprop)
        elif self.cfg.train_params.optimizer.lower() == "sgd":
            optimizer = optim.SGD(self.bc_model.parameters(), **self.cfg.sgd)
        else:
            raise ValueError(f"Unknown optimizer {self.cfg.train_params.optimizer}")
        
        # Scheduler
        if self.cfg.scheduler_params.scheduler == "ReduceLROnPlateau":
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Using ReduceLROnPlateau scheduler.")
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                threshold=0.001,
                threshold_mode='rel',
                cooldown=2,
                min_lr=1e-6,
                eps=1e-8
            )
        elif self.cfg.scheduler_params.scheduler == "CosineAnnealingLR":
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Using CosineAnnealingLR scheduler.")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                
                T_max=self.cfg.scheduler_params.T_max,
                eta_min=self.cfg.scheduler_params.eta_min
            )
        else:
            scheduler = None
        
        return optimizer, scheduler

    def init_dataloader(self, split='train'):
        """Initialize dataloaders"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Initializing {split} dataloader")
        
        # Use BCDataset instead of TronDataset
        dataset = BCDataset(
            root=self.cfg.dataset.root,
            stats=self.cfg.dataset.stats,
            resize=self.cfg.dataset.resize,
            seed=self.cfg.dataset.seed,
            split=split
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=self.cfg.dataloader.batch_size,
            shuffle=(split=='train'),
            num_workers=self.cfg.dataloader.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {split.capitalize()} dataset has {len(dataset)} samples")
        
        return dataloader
    
    def if_resume(self):
        """Resume training from checkpoint if needed"""
        if self.cfg.train_params.resume_bc:
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Loading BC model checkpoint")
            
            checkpoint = load_checkpoint(self.cfg.directory.load_bc, self.device)
            
            self.bc_model.load_state_dict(checkpoint["bc_model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epoch = checkpoint["epoch"] + 1
            self.iteration = checkpoint["iteration"] + 1
            self.best_val_loss = checkpoint["best_val_loss"]
            self.train_losses = checkpoint.get("train_losses", [])
            self.val_losses = checkpoint.get("val_losses", [])
            
            if self.scheduler is not None and "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
                
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Successfully resumed from epoch {self.epoch-1}")

    def save(self, save_best_only=False, name=None):
        """Save model checkpoint"""
        save_path = Path(self.cfg.directory.save)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "time": str(datetime.now()),
            "bc_model": self.bc_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
            "epoch": self.epoch,
            "iteration": self.iteration,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        
        if name is None:
            save_name = f"bc_model_epoch_{self.epoch:03d}"
        else:
            save_name = name
        
        # Call save_checkpoint with all required arguments
        if save_best_only:
            save_checkpoint(checkpoint, True, str(save_path), "best_model")
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Saved BEST model with loss {self.best_val_loss:.4f}")
        else:
            save_checkpoint(checkpoint, False, str(save_path), save_name)
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Saved checkpoint at epoch {self.epoch}")
        
        # Log model to comet
        log_model(self.logger, self.bc_model, model_name=f"bc_model_epoch_{self.epoch:03d}")

    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, self.epoch, self.loss_interval), self.train_losses, label='Training Loss')
        plt.plot(range(0, self.epoch, self.loss_interval), self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('BC Model Training Progress')
        plt.legend()
        plt.grid(True)
        
        save_path = Path(self.cfg.directory.save) / "loss_plot.png"
        plt.savefig(save_path)
        plt.close()
        
        # Log to logger
        self.logger.log_image(str(save_path), name="bc_model_loss_plot")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="./conf/config_bc", type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    trainer = BCModelTrainer(cfg_path)
    trainer.train()