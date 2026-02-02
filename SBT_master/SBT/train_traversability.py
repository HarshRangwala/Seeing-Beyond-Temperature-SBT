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
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from comet_ml.integration.pytorch import log_model, watch
import cv2

# --- Project Specific Imports ---
#  dataloader is in traverse_dataloader.py
from model.traverse_dataloader import TraversabilityDataset
#  pretrained model base (VisionEncoder, TronModel) is in model.m2p2_model
from model.m2p2_model import VisionEncoder, TronModel
#  traversability model parts are in traversability_model.py
from model.traversability_model import TraversabilityDecoder, TraversabilityLoss

from utils.nn import check_grad_norm, init_weights, EarlyStopping
from utils.io import save_checkpoint, load_checkpoint
from utils.helpers import get_conf, init_logger, init_device, timeit

matplotlib.use('Agg') 

class TraversabilityTrainer:
    def __init__(self, cfg_dir: str):
        # Load config and initialize the logger and device
        self.cfg = get_conf(cfg_dir)
        # Define experiment name and save directory based on traversability task
        self.cfg.directory.model_name = self.cfg.train_params.experiment_name
        self.cfg.directory.model_name += f"-trav-{self.cfg.model.rep_size}-{datetime.now():%m-%d-%H-%M}"
        self.cfg.train_params.experiment_name = self.cfg.directory.model_name
        self.cfg.directory.save = str(
            Path(self.cfg.directory.save) / self.cfg.directory.model_name
        )
        self.logger = init_logger(self.cfg)
        self.device = init_device(self.cfg)

        # Create dataloaders for Traversability
        self.train_data = self.init_dataloader(split='train')
        self.val_data = self.init_dataloader(split='validation')
        self.logger.log_parameters(
            {"train_len": len(self.train_data.dataset) if hasattr(self.train_data, 'dataset') else len(self.train_data),
             "val_len": len(self.val_data.dataset) if hasattr(self.val_data, 'dataset') else len(self.val_data)}
        )

        # Load pretrained vision encoder (part of TronModel)
        self.pretrained_model = self.load_pretrained_model()
        self.pretrained_model.eval()  # Set base encoder to evaluation mode

        # Initialize Traversability decoder
        self.traversability_decoder = TraversabilityDecoder(
            latent_size=self.cfg.model.rep_size,
            num_layers=self.cfg.model.num_layers_enc # Ensure this matches the encoder used in pretraining
        ).to(self.device)
        self.traversability_decoder.apply(init_weights(**self.cfg.init_weights))
        watch(self.traversability_decoder)

        loss_weights = self.cfg.get('traversability_loss_weights', {'bce': 1.0, 'dice': 1.0})
        self.criterion = TraversabilityLoss(weights=loss_weights).to(self.device)

        # Initialize optimizer for traversability decoder
        self.optimizer, self.scheduler = self.init_optimizer()

        # Training variables
        self.best_val_loss = float('inf') # Using validation loss to track best model
        self.best_iou = 0.0 
        self.epoch = 1
        self.iteration = 0
        self.traversability_losses = []
        self.traversability_val_loss = []
        self.val_iou_scores = []
        self.val_f1_scores = []
        self.loss_interval = 5

        # Resume from checkpoint if needed
        self.if_resume()

    

    def train(self):
        """Trains the traversability decoder using the pretrained vision encoder features"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Starting Traversability Decoder training!")

        for epoch in range(self.epoch, self.cfg.train_params.epochs + 1):
            self.epoch = epoch
            self.traversability_decoder.train() # Set decoder to training mode

            running_batch_loss = []

            bar = tqdm(
                self.train_data,
                desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training Traversability: ",
            )

            for data in bar:
                self.iteration += 1
                batch_loss = self.train_batch(data)
                running_batch_loss.append(batch_loss)

                bar.set_postfix(batch_loss=f"{batch_loss:.4f}")

                # Visualize prediction periodically
                if self.iteration % 100 == 0:
                    self.visualize_prediction(data, "train")

            bar.close()

            # Validate decoder
            val_loss, val_metrics = self.validate()
            print(f"Validation Results - Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}")
            # Update scheduler based on validation loss
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Log average training loss for the epoch
            avg_epoch_loss = np.mean(running_batch_loss)
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                f"Iteration {self.iteration:05} summary: Train Loss: {avg_epoch_loss:.4f}"
            )
            self.logger.log_metric("train/traversability/epoch_loss", avg_epoch_loss, step=self.epoch)

            # Store losses for plotting
            if self.epoch % self.loss_interval == 0:
                self.traversability_losses.append(avg_epoch_loss)
                self.traversability_val_loss.append(val_loss)
                self.val_iou_scores.append(val_metrics['iou'])
                self.val_f1_scores.append(val_metrics['f1'])

            # Save best model based on validation loss (or IoU)
            save_best = False
            if val_loss < self.best_val_loss * 0.98:
                 print(f"New best validation loss: {val_loss:.4f} (previous: {self.best_val_loss:.4f})")
                 self.best_val_loss = val_loss
                 save_best = True
            # Alternatively, save based on IoU
            # if val_metrics['iou'] > self.best_iou:
            #     print(f"New best validation IoU: {val_metrics['iou']:.4f} (previous: {self.best_iou:.4f})")
            #     self.best_iou = val_metrics['iou']
            #     self.best_val_loss = val_loss # Still store loss for reference
            #     save_best = True

            if save_best:
                self.save(save_best_only=True)

            # Regular save
            if self.epoch % self.cfg.train_params.save_every == 0:
                self.save(save_best_only=False)

            gc.collect() # Garbage collection

        self.plot_metrics() # Plot final losses and metrics
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Traversability Decoder training is DONE!")

    def train_batch(self, data):
        """Processes a single training batch"""
        self.traversability_decoder.train() # Ensure decoder is in train mode

        # Move data to device
        thermal, mask = data
        thermal = thermal.to(self.device)
        mask = mask.to(self.device) # Ground truth mask

        # Get encoder output (no gradient computation needed for pretrained part)
        with torch.no_grad():
            # Use the same thermal input for the encoder
            v_encoded_thermal, thermal_features = self.pretrained_model.vision_encoder(thermal, return_features=True)

        # Forward pass through traversability decoder
        predicted_mask = self.traversability_decoder(v_encoded_thermal, thermal_features)

        # Compute loss (using isTraining=True in the loss function)
        loss = self.criterion(predicted_mask, mask, isTraining=True)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log batch loss immediately
        self.logger.log_metric("train/traversability/batch_loss", loss.item(), step=self.iteration)

        return loss.cpu().item()

    def validate(self):
        """Validate the traversability decoder"""
        self.traversability_decoder.eval() 
        running_val_loss = []
        running_metrics = {'iou': [], 'f1': [], 'precision': [], 'recall': []}

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.val_data, desc=f"Epoch {self.epoch:03}, validating...")):
                thermal, mask = data
                thermal = thermal.to(self.device)
                mask = mask.to(self.device)

                # Get encoder output
                v_encoded_thermal, thermal_features = self.pretrained_model.vision_encoder(thermal, return_features=True)

                # Get traversability prediction
                predicted_mask = self.traversability_decoder(v_encoded_thermal, thermal_features)

                # Compute loss and metrics (using isTraining=False)
                val_loss, metrics = self.criterion(predicted_mask, mask, isTraining=False)

                running_val_loss.append(val_loss.item())

                # Store metrics for averaging
                for k, v in metrics.items():
                    if k in running_metrics: # Ensure key exists
                        running_metrics[k].append(v)

                # Visualize validation samples periodically
                if i % 10 == 0:
                    self.visualize_prediction(data, "val", i, predicted_mask_tensor=predicted_mask)

        # Calculate average metrics
        avg_val_loss = np.mean(running_val_loss)
        avg_metrics = {k: np.mean(v) if v else 0 for k, v in running_metrics.items()} # Handle empty lists if validation set is small

        print(f"Validation Metrics - IoU: {avg_metrics['iou']:.4f}, "
              f"F1: {avg_metrics['f1']:.4f}, "
              f"Precision: {avg_metrics['precision']:.4f}, "
              f"Recall: {avg_metrics['recall']:.4f}")

        self.logger.log_metric("val/traversability/epoch_loss", avg_val_loss, step=self.epoch)
        self.logger.log_metric("val/metrics/iou", avg_metrics['iou'], step=self.epoch)
        self.logger.log_metric("val/metrics/f1", avg_metrics['f1'], step=self.epoch)
        self.logger.log_metric("val/metrics/precision", avg_metrics['precision'], step=self.epoch)
        self.logger.log_metric("val/metrics/recall", avg_metrics['recall'], step=self.epoch)

        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.log_metric("train/learning_rate", current_lr, step=self.epoch)

        return avg_val_loss, avg_metrics
    
    def visualize_prediction(self, data, split, batch_idx=0, predicted_mask_tensor=None):
        """Visualize training or validation predictions"""
        vis_folder = Path(self.cfg.directory.save) / f"{split}_visualizations"
        vis_folder.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            thermal, mask_gt = data
            thermal = thermal.to(self.device) # Ensure thermal is on device for potential re-inference

            
            v_encoded_thermal, thermal_features = self.pretrained_model.vision_encoder(thermal, return_features=True)

            predicted_mask_tensor = self.traversability_decoder(v_encoded_thermal, thermal_features)

            # Move tensors to CPU and convert to numpy for visualization
            pred_mask_np = predicted_mask_tensor.cpu().numpy()
            thermal_np = thermal.cpu().numpy()
            mask_gt_np = mask_gt.cpu().numpy() # Ground truth mask

            # Visualize first image in batch
            pred_img = pred_mask_np[0, 0, :, :] # Prediction (values 0-1)
            thermal_img = thermal_np[0, 0, :, :] # Input thermal
            gt_img = mask_gt_np[0, 0, :, :] # Ground truth mask


            # Define save path
            if split == 'train':
                 save_name = f"train_trav_pred_epoch_{self.epoch:03}_iter_{self.iteration:05}.png"
            else: # validation
                 save_name = f"val_trav_pred_epoch_{self.epoch:03}_batch_{batch_idx:03}.png"
            save_dir = vis_folder / save_name

            # Plot visualization (Thermal, Ground Truth Mask, Predicted Mask)
            plt.figure(figsize=(15, 5)) # Adjusted size for 3 plots

            plt.subplot(1, 3, 1)
            plt.title("Thermal")
            plt.imshow(thermal_img, cmap='gray') 
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Ground Truth Mask")
            plt.imshow(gt_img, cmap='gray') 
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask")
            plt.imshow(pred_img, cmap='gray') 
            # Or show probabilities: plt.imshow(pred_img, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')

            plt.tight_layout(pad=1.5)
            plt.savefig(save_dir)
            plt.close()

            # Log image to Comet ML logger
            log_context = f"{split}_images"
            log_name = save_name.replace(".png", "") # Use filename without extension as log name
            with self.logger.context_manager(log_context):
                self.logger.log_image(
                    str(save_dir),
                    name=log_name,
                    step=self.epoch if split == 'val' else self.iteration # Log val per epoch, train per iteration
                )

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
            param.requires_grad = False
            
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Successfully loaded pretrained model")
        return model
    
    def init_optimizer(self):
        """Initialize optimizer and scheduler for depth decoder"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Initializing optimizer and scheduler for depth decoder")
        
        # Optimizer
        if self.cfg.train_params.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(self.traversability_decoder.parameters(), **self.cfg.trav_decoder_adamw)
        elif self.cfg.train_params.optimizer.lower() == "rmsprop":
            optimizer = optim.RMSprop(self.traversability_decoder.parameters(), **self.cfg.rmsprop)
        elif self.cfg.train_params.optimizer.lower() == "sgd":
            optimizer = optim.SGD(self.traversability_decoder.parameters(), **self.cfg.sgd)
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
        """Initialize Traversability dataloaders"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Initializing {split} Traversability dataloader")

        dataset = TraversabilityDataset(**self.cfg.dataset, split=split)
        dataloader = DataLoader(dataset, **self.cfg.dataloader)
        
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {split.capitalize()} dataset has {len(dataset)} samples")
        
        return dataloader
    
    def if_resume(self):
        """Resume training from checkpoint if specified in config"""
        # Check for a specific resume path for the traversability decoder
        resume_path = self.cfg.directory.get('resume_traversability_decoder_path', None)
        if resume_path and Path(resume_path).exists():
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Resuming Traversability decoder training from {resume_path}")

            checkpoint = load_checkpoint(resume_path, self.device)
            # Load state dicts
            if "traversability_decoder" in checkpoint:
                self.traversability_decoder.load_state_dict(checkpoint["traversability_decoder"])
            else:
                print("Warning: 'traversability_decoder' key not found in checkpoint. Decoder weights not loaded.")

            if "optimizer" in checkpoint:
                 try:
                      self.optimizer.load_state_dict(checkpoint["optimizer"])
                 except ValueError as e:
                      print(f"Warning: Could not load optimizer state, possibly due to parameter mismatch: {e}. Optimizer state reset.")
            else:
                 print("Warning: 'optimizer' key not found in checkpoint. Optimizer state not loaded.")

            # Load training progress
            self.epoch = checkpoint.get("epoch", 0) + 1 # Start from next epoch
            self.iteration = checkpoint.get("iteration", 0) # Resume iteration count
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            self.best_iou = checkpoint.get("best_iou", 0.0)

            # Load loss/metric history for plotting continuity
            self.traversability_losses = checkpoint.get("traversability_losses", [])
            self.traversability_val_loss = checkpoint.get("traversability_val_loss", [])
            self.val_iou_scores = checkpoint.get("val_iou_scores", [])
            self.val_f1_scores = checkpoint.get("val_f1_scores", [])


            if self.scheduler is not None and "scheduler" in checkpoint:
                 try:
                      self.scheduler.load_state_dict(checkpoint["scheduler"])
                 except Exception as e:
                      print(f"Warning: Could not load scheduler state: {e}. Scheduler state reset.")

            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Successfully resumed from epoch {self.epoch-1}")
        elif resume_path:
            print(f"Warning: Resume path specified but not found: {resume_path}. Starting training from scratch.")
        else:
             print("No resume path specified for traversability decoder. Starting training from scratch.")

    def save(self, save_best_only=False, name=None):
        """Save model checkpoint"""
        save_path = Path(self.cfg.directory.save)
        save_path.mkdir(parents=True, exist_ok=True)

        # --- Save only the decoder and its optimizer state ---
        checkpoint = {
            "time": str(datetime.now()),
            "epoch": self.epoch,
            "iteration": self.iteration,
            "traversability_decoder": self.traversability_decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
            "best_val_loss": self.best_val_loss,
            "best_iou": self.best_iou,
            # Store history for resuming plots
            "traversability_losses": self.traversability_losses,
            "traversability_val_loss": self.traversability_val_loss,
            "val_iou_scores": self.val_iou_scores,
            "val_f1_scores": self.val_f1_scores,
        }

        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        if name is None:
            save_name = f"traversability_decoder_epoch_{self.epoch:03d}"
        else:
            save_name = name
        if save_best_only:
             # Use a consistent name for the best model
            best_filename = "best_traversability_model"
            save_checkpoint(checkpoint, True, str(save_path), best_filename)
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Saved BEST traversability model with val_loss {self.best_val_loss:.4f}")
             # Log the best model file to Comet
            try:
                 self.logger.log_model("best_traversability_decoder", str(save_path / f"{best_filename}.pth"))
            except Exception as e:
                 print(f"Warning: Could not log best model to Comet: {e}")
        else:
            save_checkpoint(checkpoint, False, str(save_path), save_name)
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Saved checkpoint at epoch {self.epoch}")
            log_model(self.logger, self.traversability_decoder, model_name=f"traversability_decoder_epoch_{self.epoch:03d}")

    def plot_metrics(self):
        """Plot training and validation losses and key metrics"""
        save_path_dir = Path(self.cfg.directory.save)
        save_path_dir.mkdir(parents=True, exist_ok=True)

        epochs_recorded = list(range(self.loss_interval, self.epoch + 1, self.loss_interval))
        if not epochs_recorded: # Handle case where training ends before first interval
            print("Not enough epochs completed to plot metrics.")
            return

        # Ensure lists have the same length as epochs_recorded
        num_points = len(epochs_recorded)
        train_losses = self.traversability_losses[-num_points:]
        val_losses = self.traversability_val_loss[-num_points:]
        val_ious = self.val_iou_scores[-num_points:]
        val_f1s = self.val_f1_scores[-num_points:]


        # --- Plot Losses ---
        plt.figure(figsize=(12, 5))
        plt.plot(epochs_recorded, train_losses, label='Avg Training Loss', marker='o')
        plt.plot(epochs_recorded, val_losses, label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Traversability Decoder Training Progress - Loss')
        plt.legend()
        plt.grid(True)
        loss_plot_path = save_path_dir / "traversability_loss_plot.png"
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Saved loss plot to {loss_plot_path}")
        # Log loss plot to logger
        self.logger.log_image(str(loss_plot_path), name="traversability_loss_plot", step=self.epoch)

        # --- Plot Metrics (IoU and F1) ---
        plt.figure(figsize=(12, 5))
        plt.plot(epochs_recorded, val_ious, label='Validation IoU', marker='s')
        plt.plot(epochs_recorded, val_f1s, label='Validation F1-Score', marker='^')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Traversability Decoder Training Progress - Metrics')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1) # Metrics are typically between 0 and 1
        metrics_plot_path = save_path_dir / "traversability_metrics_plot.png"
        plt.savefig(metrics_plot_path)
        plt.close()
        print(f"Saved metrics plot to {metrics_plot_path}")
        # Log metrics plot to logger
        self.logger.log_image(str(metrics_plot_path), name="traversability_metrics_plot", step=self.epoch)
        # --- Log final metrics ---
        print(f"Final Training Loss: {train_losses[-1]:.4f}")
        print(f"Final Validation Loss: {val_losses[-1]:.4f}")
        print(f"Final Validation IoU: {val_ious[-1]:.4f}")
        print(f"Final Validation F1-Score: {val_f1s[-1]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="./conf/config_trav", type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    learner = TraversabilityTrainer(cfg_path)
    learner.train()