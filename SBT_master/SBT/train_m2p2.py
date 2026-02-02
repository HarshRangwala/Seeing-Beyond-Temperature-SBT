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
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from comet_ml.integration.pytorch import log_model, watch
import matplotlib
from model.m2p2_dataloader import TronDataset
from model.m2p2_model import VisionEncoder, Projector, TronModel, IMUEncoder
from utils.nn import check_grad_norm, init_weights
from utils.io import save_checkpoint, load_checkpoint
from utils.helpers import get_conf, init_logger, init_device, timeit, print_model_stats, compute_and_plot_correlation
matplotlib.use('Agg') 
class TRONPretrainer:
    def __init__(self, cfg_dir: str):
        # Load config and initialize logger and device
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
        
        # Initialize model
        self.model = self.init_tron_model()
        self.model.vision_encoder.apply(init_weights(**self.cfg.init_weights))
        watch(self.model)
        self.logger.log_code(folder="./model")
        
        # Initialize optimizer
        self.optimizer, self.scheduler = self.init_optimizer()
        
        # Initialize training variables
        self.best_loss = float('inf')
        self.epoch = 1
        self.iteration = 0
        self.e_loss = []
        
        # Resume from checkpoint if needed
        self.if_resume()
        
        # Logging variables
        self.barlow_losses = []
        self.val_losses = []
        self.loss_interval = 5

    def train(self):
        """Trains the self-supervised TRON model"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Starting TRON pretraining!")
        
        for epoch in range(self.epoch, self.cfg.train_params.epochs + 1):
            self.epoch = epoch
            self.model.train()
            
            running_loss = []
            running_barlow_loss_v = []
            
            bar = tqdm(
                self.train_data,
                desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training TRON: ",
            )
            
            for data in bar:
                self.iteration += 1
                (loss, barlow_loss_v, grad_norm), t_train = self.forward_batch(data)
                t_train /= self.train_data.batch_size
                
                running_loss.append(loss)
                running_barlow_loss_v.append(barlow_loss_v)
                
                with self.logger.context_manager("trainer_info"):
                    self.logger.log_metric("train/grad_norm", grad_norm, step=self.iteration)
                    self.logger.log_metric("train/learning_rate", self.optimizer.param_groups[0]['lr'], step=self.iteration)
                
                bar.set_postfix(
                    loss=loss, 
                    barlow_loss_v=barlow_loss_v, 
                    grad=grad_norm, 
                    time=t_train
                )
            
            bar.close()
            
            # Log metrics
            if self.epoch % self.loss_interval == 0:
                self.barlow_losses.append(np.mean(running_loss))
                
                # if self.epoch < self.cfg.train_params.bt_start:
                #     self.e_loss.append(np.mean(running_loss))
            else:
                self.e_loss.append(np.mean(running_loss))
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Average loss for epoch
            epoch_loss = np.mean(running_loss)
            
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                + f"Iteration {self.iteration:05} summary: train Loss: {epoch_loss:.4f} || "
                + f"Barlow Vision Loss: {np.mean(running_barlow_loss_v):.4f} || "
            )
            
            self.logger.log_metric("train/epoch_loss", epoch_loss, step=self.epoch)
            self.logger.log_metric("train/Barlow Vision Loss", np.mean(running_barlow_loss_v), step=self.epoch)
            print("Best Loss --", self.best_loss)
            print("EPOCH LOSS --", epoch_loss)
            # Save checkpoints
            if self.epoch % self.cfg.train_params.save_every == 0:
                self.save(save_best_only=False)  # Regular interval save
            
            # Save if we have a new best model
            if epoch_loss < self.best_loss * 0.95 and self.epoch >= self.cfg.train_params.start_saving_best:
                self.best_loss = epoch_loss
                self.save(save_best_only=True)  # Save as best model
            

            
        self.save() # Final Save
        self.plot_losses()
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - TRON pretraining is DONE!")

    @timeit
    def forward_batch(self, data):
        """Forward pass of a batch"""
        self.model.train()
        
        # Move data to device
        # Move data to device
        thermal, depth, accel, gyro = data
        thermal = thermal.to(device=self.device)
        depth = depth.to(device=self.device)
        accel = accel.to(device=self.device)
        gyro = gyro.to(device=self.device)
        
        # Forward pass
        zv1, zv2, zi, _, _, _, _, _ = self.model(thermal, depth, accel, gyro)
        
        # Compute Barlow Twins loss
        barlow_loss_v = self.model.barlow_loss(zv1, zv2)  # Vision-Vision (Thermal-Depth) invariance
        
        if zi is not None:
             # Vision-IMU invariance (Thermal-IMU)
             barlow_loss_vi = self.model.barlow_loss(zv1, zi)
             
             # Combined loss
             # Using l1_coeff as the weighting factor (default 0.5 in config)
             # If l1_coeff is 1.0, we ignore IMU. If 0.5, equal weight.
             loss = self.cfg.model.l1_coeff * barlow_loss_v + (1.0 - self.cfg.model.l1_coeff) * barlow_loss_vi
        else:
             loss = barlow_loss_v
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Check grad norm for debugging
        grad_norm = check_grad_norm(self.model)
        
        # Gradient clipping
        if self.cfg.train_params.grad_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train_params.grad_clipping
            )
        
        # Update weights
        self.optimizer.step()
        
        # Plot correlation periodically
        if self.epoch % 10 == 0 and self.iteration % 100 == 0:
            correlation_plot_path = compute_and_plot_correlation(zv1, zv2, self.epoch, self.iteration)
            with self.logger.context_manager("train_correlation"):
                self.logger.log_image(
                    correlation_plot_path, 
                    name=f"train_correlation_epoch_{self.epoch:03}_iter_{self.iteration:05}", 
                    step=self.epoch
                )
        
        return loss.detach().cpu().item(), barlow_loss_v.detach().cpu().item(), grad_norm


    def init_tron_model(self):
        """Initialize the TRON model"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the model with {self.cfg.model.rep_size} representation dim!")
        
        # Vision encoder
        vision_encoder = VisionEncoder(latent_size=self.cfg.model.rep_size, num_layers=self.cfg.model.num_layers_enc, pretrained=self.cfg.model.pretrained).to(self.device)
        
        # IMU encoder
        imu_encoder = IMUEncoder(latent_size=self.cfg.model.rep_size).to(self.device)
        
        # Projector
        projector = Projector(
            in_dim=self.cfg.model.rep_size,
            hidden_dim=4*self.cfg.model.rep_size,
            out_dim=self.cfg.model.rep_size
        ).to(self.device)
        
        # TRON model
        model = TronModel(vision_encoder, imu_encoder, projector, latent_size=self.cfg.model.rep_size)
        
        return model.to(self.device)

    def init_optimizer(self):
        """Initialize optimizer and scheduler"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the optimizer and scheduler!")
        
        # Optimizer
        if self.cfg.train_params.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), **self.cfg.adamw)
        elif self.cfg.train_params.optimizer.lower() == "rmsprop":
            optimizer = optim.RMSprop(self.model.parameters(), **self.cfg.rmsprop)
        elif self.cfg.train_params.optimizer.lower() == "sgd":
            optimizer = optim.SGD(self.model.parameters(), **self.cfg.sgd)
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
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the {split} dataloader!")
        
        dataset = TronDataset(**self.cfg.dataset, split=split)
        data = DataLoader(dataset, **self.cfg.dataloader)
        
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {split.capitalize()} dataset has {len(dataset)} samples.")
        
        return data

    def if_resume(self):
        """Resume training from checkpoint if needed"""
        if self.cfg.train_params.resume:
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - LOADING checkpoint!")
            save_dir = self.cfg.directory.load
            checkpoint = load_checkpoint(save_dir, self.device)
            
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epoch = checkpoint["epoch"] + 1
            self.e_loss = checkpoint["e_loss"]
            self.iteration = checkpoint["iteration"] + 1
            self.best_loss = checkpoint["best_loss"]
            
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} - "
                f"LOADING checkpoint successful, starting from epoch {self.epoch} "
                f"with best loss {self.best_loss}"
            )

    def save(self, save_best_only=False, name=None):
        """Save checkpoint"""
        checkpoint = {
            "time": str(datetime.now()),
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model": self.model.state_dict(),
            "vision_encoder": self.model.vision_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
            "best_loss": self.best_loss,
            "e_loss": self.e_loss,
            "barlow_losses": self.barlow_losses,
            "cfg": self.cfg  # Save configuration for downstream tasks
        }
        
        if name is None:
            save_name = f"{self.cfg.directory.model_name}_{self.epoch}"
        else:
            save_name = name
        
        # Create directory if it doesn't exist
        os.makedirs(self.cfg.directory.save, exist_ok=True)
        # Save checkpoint without re-checking the loss condition
        if save_best_only:
            save_checkpoint(checkpoint, True, self.cfg.directory.save, "best_model")
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Saved BEST model with loss {self.best_loss:.4f}")
        else:
            save_checkpoint(checkpoint, False, self.cfg.directory.save, save_name)
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Saved checkpoint at epoch {self.epoch}")
        
        # # Save full checkpoint
        # if self.e_loss[-1] < self.best_loss:
        #     self.best_loss = self.e_loss[-1]
        #     checkpoint["best_loss"] = self.best_loss
            
        #     # Save best model
        #     if save_best_only:
        #         save_checkpoint(checkpoint, True, self.cfg.directory.save, "best_model")
        #         print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Saved BEST model with loss {self.best_loss:.4f}")
        #     else:
        #         # Save both best and regular checkpoint
        #         # save_checkpoint(checkpoint, True, self.cfg.directory.save, "best_model")
        #         save_checkpoint(checkpoint, False, self.cfg.directory.save, save_name)
        #         print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Saved checkpoint at epoch {self.epoch}")

    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(12, 6))
        
        # Plot training losses
        epochs = list(range(1, len(self.e_loss) + 1))
        plt.plot(epochs, self.e_loss, 'b-', label='Training Loss')
        
        # Plot component losses if available
        if self.barlow_losses:
            loss_epochs = list(range(self.loss_interval, len(self.barlow_losses) * self.loss_interval + 1, self.loss_interval))
            plt.plot(loss_epochs, self.barlow_losses, 'g-', label='Barlow Loss')
        # Plot validation loss if available
        if self.val_losses:
            val_epochs = list(range(self.loss_interval, len(self.val_losses) * self.loss_interval + 1, self.loss_interval))
            plt.plot(val_epochs, self.val_losses, 'm-', label='Validation Loss')
        
        plt.title('TRON Pretraining Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        os.makedirs(self.cfg.directory.save_loss_plot, exist_ok=True)
        plot_path = os.path.join(self.cfg.directory.save_loss_plot, f"{self.cfg.directory.model_name}_loss.png")
        plt.savefig(plot_path)
        
        # Log plot to CometML
        if self.logger is not None:
            self.logger.log_image(plot_path, name="training_losses")
        
        plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="./conf/config_m2p2", type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    learner = TRONPretrainer(cfg_path)
    learner.train()