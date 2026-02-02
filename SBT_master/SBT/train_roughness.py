import gc
from pathlib import Path
from datetime import datetime
import sys
import os
import argparse

# Ensure the project root is in the Python path
try:
    sys.path.append(str(Path(".").resolve()))
except Exception as e:
    print(f"Warning: Could not append project root to sys.path. Error: {e}")
    # raise RuntimeError("Can't append root directory of the project to the path") # Optional: make it fatal

from rich import print
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn # Import nn for standard loss functions
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from comet_ml.integration.pytorch import log_model, watch
import cv2

# --- Project Specific Imports ---
# Dataloader for roughness
from model.roughness_dataloader import RoughnessDataset
# Pretrained model base (VisionEncoder, TronModel)
from model.m2p2_model import VisionEncoder, TronModel
# Roughness prediction model
from model.roughness_model import RoughnessModel

from utils.nn import check_grad_norm, init_weights, EarlyStopping
from utils.io import save_checkpoint, load_checkpoint
from utils.helpers import get_conf, init_logger, init_device, timeit

matplotlib.use('Agg')

class RoughnessTrainer: # Renamed class
    def __init__(self, cfg_dir: str):
        # Load config and initialize the logger and device
        self.cfg = get_conf(cfg_dir)
        # Define experiment name and save directory based on roughness task
        self.cfg.directory.model_name = self.cfg.train_params.experiment_name
        # Updated suffix for roughness
        self.cfg.directory.model_name += f"-rg-{self.cfg.model.rep_size}-{datetime.now():%m-%d-%H-%M}"
        self.cfg.train_params.experiment_name = self.cfg.directory.model_name
        self.cfg.directory.save = str(
            Path(self.cfg.directory.save) / self.cfg.directory.model_name
        )
        self.logger = init_logger(self.cfg)
        self.device = init_device(self.cfg)

        # Create dataloaders for Roughness
        self.train_data = self.init_dataloader(split='train')
        self.val_data = self.init_dataloader(split='validation')
        self.logger.log_parameters(
            {"train_len": len(self.train_data.dataset) if hasattr(self.train_data, 'dataset') else len(self.train_data),
             "val_len": len(self.val_data.dataset) if hasattr(self.val_data, 'dataset') else len(self.val_data)}
        )

        # Load pretrained vision encoder (part of TronModel)
        # self.pretrained_model = self.load_pretrained_model()
        # self.pretrained_model.eval()  # Set base encoder to evaluation mode
        self.vision_encoder = self.load_pretrained_model()

        # Initialize Roughness prediction model
        self.rg_model = RoughnessModel(
            input_dim=self.cfg.model.rep_size,
            hidden_dims=self.cfg.model.hidden_dims # Assuming hidden_dims is in config
        ).to(self.device)
        self.rg_model.apply(init_weights(**self.cfg.init_weights))
        watch(self.rg_model) # Watch the roughness model with Comet ML

        # Initialize Loss function for Roughness (Regression Task)
        loss_type = self.cfg.train_params.get('loss', 'mse').lower()
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
            print("Using MSE Loss for roughness prediction.")
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss()
            print("Using L1 Loss (MAE) for roughness prediction.")
        else:
            print(f"Warning: Unknown loss type '{loss_type}' specified. Defaulting to MSE Loss.")
            self.criterion = nn.MSELoss()
        self.criterion = self.criterion.to(self.device)

        self.optimizer, self.scheduler = self.init_optimizer()

        self.early_stopper = EarlyStopping(
            patience=self.cfg.train_params.early_stop_patience,  # Recommended: 10-20
            delta=self.cfg.train_params.early_stop_delta,        # Recommended: 0.001
            verbose=True
        )

        # Training variables
        self.best_val_loss = float('inf') # Use validation loss to track best model
        self.epoch = 1
        self.iteration = 0
        self.roughness_losses = []       
        self.roughness_val_losses = []   
        self.loss_interval = 5

        self.if_resume()

    def train(self):
        """Trains the roughness prediction model using the pretrained vision encoder features"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Starting Roughness Model training!")

        for epoch in range(self.epoch, self.cfg.train_params.epochs + 1):
            self.epoch = epoch
            self.rg_model.train() # Set roughness model to training mode

            running_batch_loss = []

            bar = tqdm(
                self.train_data,
                desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training Roughness: ", # Updated description
            )

            for data in bar:
                self.iteration += 1
                batch_loss = self.train_batch(data)
                running_batch_loss.append(batch_loss)

                bar.set_postfix(batch_loss=f"{batch_loss:.4f}")

                # No visualization needed per batch for regression, unlike segmentation

            bar.close()

            # Validate model
            val_loss = self.validate() # Validate returns only loss for regression
            print(f"Validation Results - Loss: {val_loss:.4f}")

            self.early_stopper(val_loss, self.rg_model)
        
            if self.early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                self.save()  
                break

            # Update scheduler based on validation loss
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Log average training loss for the epoch
            avg_epoch_loss = np.mean(running_batch_loss)
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                f"Iteration {self.iteration:05} summary: Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            self.logger.log_metric("train/roughness/epoch_loss", avg_epoch_loss, step=self.epoch) # Updated log key

            # Store losses for plotting
            if self.epoch % self.loss_interval == 0:
                self.roughness_losses.append(avg_epoch_loss)
                self.roughness_val_losses.append(val_loss)

            # Save best model based on validation loss
            save_best = False
            # Added a small relative improvement threshold to avoid saving too often
            if val_loss < self.best_val_loss * 0.995:
                 print(f"New best validation loss: {val_loss:.4f} (previous: {self.best_val_loss:.4f})")
                 self.best_val_loss = val_loss
                 save_best = True

            if save_best:
                self.save(save_best_only=True)

            # Regular save
            if self.epoch % self.cfg.train_params.save_every == 0:
                self.save(save_best_only=False)

            gc.collect() # Garbage collection

        self.plot_metrics() # Plot final losses
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Roughness Model training is DONE!")

    def train_batch(self, data):
        """Processes a single training batch for roughness prediction"""
        self.rg_model.train() # Ensure model is in train mode

        # Move data to device
        # Assuming RoughnessDataset yields (thermal_image, roughness_score)
        thermal, roughness_gt = data
        thermal = thermal.to(self.device)
        roughness_gt = roughness_gt.to(self.device).float() # Ensure GT is float and on device
        # Reshape GT if necessary (e.g., if loss expects [batch_size, 1])
        if len(roughness_gt.shape) == 1:
            roughness_gt = roughness_gt.unsqueeze(1)

        # Get encoder output (no gradient computation needed for pretrained part)
        # with torch.no_grad():
            # Use the thermal input for the encoder
            # We likely only need the final latent representation, not intermediate features
        v_encoded_thermal, _ = self.vision_encoder(thermal, return_features=True)

        # Forward pass through roughness model
        predicted_roughness = self.rg_model(v_encoded_thermal)

        # Compute loss
        loss = self.criterion(predicted_roughness, roughness_gt)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.rg_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Log batch loss immediately
        self.logger.log_metric("train/roughness/batch_loss", loss.item(), step=self.iteration) # Updated log key

        return loss.cpu().item()

    def validate(self):
        """Validate the roughness prediction model"""
        self.rg_model.eval() 
        running_val_loss = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.val_data, desc=f"Epoch {self.epoch:03}, validating...")):
                thermal, roughness_gt = data
                thermal = thermal.to(self.device)
                roughness_gt = roughness_gt.to(self.device).float() # Ensure GT is float and on device
                if len(roughness_gt.shape) == 1:
                    roughness_gt = roughness_gt.unsqueeze(1)

                v_encoded_thermal, _ = self.vision_encoder(thermal, return_features=True)

                predicted_roughness = self.rg_model(v_encoded_thermal)

                val_loss = self.criterion(predicted_roughness, roughness_gt)

                running_val_loss.append(val_loss.item())

        # Calculate average validation loss
        avg_val_loss = np.mean(running_val_loss)

        # Log validation loss
        self.logger.log_metric("val/roughness/epoch_loss", avg_val_loss, step=self.epoch) # Updated log key

        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.log_metric("train/learning_rate", current_lr, step=self.epoch)

        return avg_val_loss 

    def load_pretrained_model(self):
        """Load the pretrained TRON model (or just the VisionEncoder part)"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Loading pretrained base model from {self.cfg.directory.pretrained_path}")

        # Initialize the base model architecture (VisionEncoder + potentially others if TronModel is complex)
        # Ensure these parameters match the *pretraining* configuration
        vision_encoder = VisionEncoder(
            latent_size=self.cfg.model.rep_size,
            num_layers=self.cfg.model.num_layers_enc
        ).to(self.device)

        model = TronModel(
            vision_encoder=vision_encoder,
            # imu_encoder=None, 
            projector=None,   
            latent_size=self.cfg.model.rep_size
        ).to(self.device)

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
            param.requires_grad = True # Changed to True
            
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Successfully loaded pretrained model")
        return vision_encoder # model

    def init_optimizer(self):
        """Initialize optimizer and scheduler for depth decoder"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Initializing optimizer and scheduler for depth decoder")
        params_to_optimize = list(self.vision_encoder.parameters())+list(self.rg_model.parameters())
        # Optimizer
        if self.cfg.train_params.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(params_to_optimize, **self.cfg.rg_dt_adamw)
        elif self.cfg.train_params.optimizer.lower() == "rmsprop":
            optimizer = optim.RMSprop(params_to_optimize, **self.cfg.rmsprop)
        elif self.cfg.train_params.optimizer.lower() == "sgd":
            optimizer = optim.SGD(params_to_optimize, **self.cfg.sgd)
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
        """Initialize Roughness dataloaders"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Initializing {split} Roughness dataloader")

        # Use RoughnessDataset
        dataset = RoughnessDataset(**self.cfg.dataset, split=split)
        dataloader = DataLoader(dataset, **self.cfg.dataloader)

        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {split.capitalize()} Roughness dataset has {len(dataset)} samples")

        return dataloader

    def if_resume(self):
        """Resume training from checkpoint if specified in config"""
        # Check for a specific resume path for the roughness model
        resume_path = self.cfg.directory.get('resume_roughness_model_path', None) # Use a specific key
        if resume_path and Path(resume_path).exists():
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Resuming Roughness model training from {resume_path}")

            checkpoint = load_checkpoint(resume_path, self.device)
            # Load state dicts
            if "vision_encoder" in checkpoint:
                self.vision_encoder.load_state_dict(checkpoint["vision_encoder"])            
            if "roughness_model" in checkpoint: 
                self.rg_model.load_state_dict(checkpoint["roughness_model"])
            else:
                print("Warning: 'roughness_model' key not found in checkpoint. Model weights not loaded.")

            if "optimizer" in checkpoint:
                 try:
                      self.optimizer.load_state_dict(checkpoint["optimizer"])
                 except ValueError as e:
                      print(f"Warning: Could not load optimizer state, possibly due to parameter mismatch: {e}. Optimizer state reset.")
                 except Exception as e:
                     print(f"Warning: Could not load optimizer state: {e}. Optimizer state reset.")
            else:
                 print("Warning: 'optimizer' key not found in checkpoint. Optimizer state not loaded.")

            # Load training progress
            self.epoch = checkpoint.get("epoch", 0) + 1 
            self.iteration = checkpoint.get("iteration", 0)
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            # No best_iou for roughness

            # Load loss history for plotting continuity
            self.roughness_losses = checkpoint.get("roughness_losses", [])
            self.roughness_val_losses = checkpoint.get("roughness_val_losses", []) 

            if self.scheduler is not None and "scheduler" in checkpoint:
                 try:
                      self.scheduler.load_state_dict(checkpoint["scheduler"])
                 except Exception as e:
                      print(f"Warning: Could not load scheduler state: {e}. Scheduler state reset.")

            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Successfully resumed from epoch {self.epoch-1}")
        elif resume_path:
            print(f"Warning: Resume path specified but not found: {resume_path}. Starting training from scratch.")
        else:
             print("No resume path specified for roughness model. Starting training from scratch.")

    def save(self, save_best_only=False, name=None):
        """Save model checkpoint"""
        save_path = Path(self.cfg.directory.save)
        save_path.mkdir(parents=True, exist_ok=True)

        # --- Save only the roughness model and its optimizer state ---
        checkpoint = {
            "time": str(datetime.now()),
            "epoch": self.epoch,
            "iteration": self.iteration,
            "vision_encoder": self.vision_encoder.state_dict(),
            "roughness_model": self.rg_model.state_dict(), # Save roughness model state
            "optimizer": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
            "best_val_loss": self.best_val_loss,
            # Store history for resuming plots
            "roughness_losses": self.roughness_losses, 
            "roughness_val_losses": self.roughness_val_losses,
        }

        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        if name is None:
            save_name = f"roughness_model_epoch_{self.epoch:03d}" # Updated name
        else:
            save_name = name
        if save_best_only:
             # Use a consistent name for the best model
            best_filename = "best_roughness_model" # Updated name
            save_checkpoint(checkpoint, True, str(save_path), best_filename)
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Saved BEST roughness model with val_loss {self.best_val_loss:.4f}")
             # Log the best model file to Comet
            try:
                 self.logger.log_model("best_roughness_model", str(save_path / f"{best_filename}.pth")) # Updated name
            except Exception as e:
                 print(f"Warning: Could not log best model to Comet: {e}")
        else:
            save_checkpoint(checkpoint, False, str(save_path), save_name)
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Saved checkpoint at epoch {self.epoch}")
            # Log regular checkpoints to Comet
            log_model(self.logger, self.rg_model, model_name=f"roughness_model_epoch_{self.epoch:03d}") # Updated name

    def plot_metrics(self):
        """Plot training and validation losses"""
        save_path_dir = Path(self.cfg.directory.save)
        save_path_dir.mkdir(parents=True, exist_ok=True)

        # Generate epoch numbers based on the logging interval
        epochs_recorded = list(range(self.loss_interval, (len(self.roughness_losses) * self.loss_interval) + 1, self.loss_interval))

        if not epochs_recorded: # Handle case where training ends before first interval
            print("Not enough epochs completed to plot metrics.")
            return

        # Ensure lists have the same length as epochs_recorded (handle resuming correctly)
        num_points = len(epochs_recorded)
        train_losses = self.roughness_losses[:num_points] # Use correct list names
        val_losses = self.roughness_val_losses[:num_points] # Use correct list names

        # --- Plot Losses ---
        plt.figure(figsize=(10, 6)) # Adjusted size for single plot
        plt.plot(epochs_recorded, train_losses, label='Avg Training Loss', marker='o')
        plt.plot(epochs_recorded, val_losses, label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (' + self.cfg.train_params.get('loss', 'mse').upper() + ')') # Indicate loss type
        plt.title('Roughness Model Training Progress - Loss') # Updated title
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        loss_plot_path = save_path_dir / "roughness_loss_plot.png" # Updated filename
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Saved loss plot to {loss_plot_path}")
        # Log loss plot to logger
        try:
            self.logger.log_image(str(loss_plot_path), name="roughness_loss_plot", step=self.epoch) # Updated log name
        except Exception as e:
             print(f"Warning: Could not log loss plot to Comet: {e}")

        # --- Log final metrics ---
        if train_losses and val_losses:
             print(f"Final Recorded Training Loss: {train_losses[-1]:.4f}")
             print(f"Final Recorded Validation Loss: {val_losses[-1]:.4f}")
             self.logger.log_metric("final/train_loss", train_losses[-1], step=self.epoch)
             self.logger.log_metric("final/val_loss", val_losses[-1], step=self.epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="./conf/config_roughness", type=str,
                        help="Path to the configuration file for roughness training.")
    args = parser.parse_args()
    cfg_path = args.conf
    learner = RoughnessTrainer(cfg_path)
    learner.train()