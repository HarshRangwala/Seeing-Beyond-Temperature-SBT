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
from sklearn.manifold import TSNE
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from comet_ml.integration.pytorch import log_model, watch

import cv2
from model.m2p2_dataloader import TronDataset
from model.m2p2_model import VisionEncoder,DepthDecoder, Projector, TronModel, DepthLoss
from model.traversability_model import TraversabilityDecoder, TraversabilityLoss

from model.depthv2_model import load_DepthAnythingv2_model
from utils.nn import check_grad_norm, op_counter, init_weights
from utils.io import save_checkpoint, load_checkpoint
from utils.helpers import get_conf, init_logger, init_device, timeit, print_model_stats, compute_and_plot_correlation
from scheduler.custom_sceduler import WarmupCosineScheduler, CosineWDSchedule


matplotlib.use('Agg')


class Learner:
    def __init__(self, cfg_dir: str):
        # load config file and initialize the logger and the device
        self.cfg = get_conf(cfg_dir)
        self.cfg.directory.model_name = self.cfg.train_params.experiment_name
        self.cfg.directory.model_name += f"-{self.cfg.model.rep_size}-{datetime.now():%m-%d-%H-%M}"
        self.cfg.train_params.experiment_name = self.cfg.directory.model_name
        self.cfg.directory.save = str(
            Path(self.cfg.directory.save) / self.cfg.directory.model_name
        )
        self.logger = init_logger(self.cfg)
        self.device = init_device(self.cfg)
        # creating dataset interface and dataloader for trained data
        # self.data = self.init_dataloader()
        self.train_data = self.init_dataloader(split='train')
        self.val_data = self.init_dataloader(split='validation')
        self.logger.log_parameters(
            {"train_len": len(self.train_data), "val_len": len(self.val_data)}
        )
        # create model  and initialize its weights and move them to the device
        self.model = self.init_m2p2_model() # Model for VisionEncoder and IMUEncoder
        # print_model_stats(self.model)
        self.model.vision_encoder.apply(init_weights(**self.cfg.init_weights))
        watch(self.model)
        self.logger.log_code(folder="./model")
        self.depth_decoder = DepthDecoder(latent_size=self.cfg.model.rep_size, num_layers=self.cfg.model.num_layers_enc).to(self.device)
        self.depth_decoder.apply(init_weights(**self.cfg.init_weights))
        watch(self.depth_decoder)
        self.depthv2_model, _ = load_DepthAnythingv2_model(depth_measurement='metric', encoder='vitl', dataset='vkitti', max_depth=20) # Model for depth anything inference

        # print_model_stats(self.depth_decoder)
        # LOss for depth reconstruction
        self.recon_criterion = DepthLoss()
        # initialize the optimizer
        self.m2p2_optimizer, self.m2p2_scheduler, self.m2p2_wd_scheduler = self.init_optimizer(self.model, "m2p2")
        self.decoder_optimizer, self.decoder_scheduler, self.decoder_wd_scheduler = self.init_optimizer(self.depth_decoder, "decoder")
        # Best reconstruction loss
        self.best_recon_loss = float('inf')
        # if resuming, load the checkpoint
        self.if_resume()

        # Parameters for dynamic loss weighting
        self.kappa = 2.63
        self.alpha = 0.99 # smoothing factor for moving average
        self.constraint_ma = None
        torch.tensor(0.0001, device=self.device, requires_grad=False) # initial lambda

        # logg losses for plot
        self.m2p2_losses = []
        self.recon_losses = []
        self.m2p2_val_loss = []
        self.recon_val_loss = []
        self.l1_loss = []
        self.loss_interval = 5

    def train(self):
        """Trains the self-supervised model and decoder"""
        cycle_length = 600
        for epoch in range(self.epoch, self.cfg.train_params.epochs+1):
            self.epoch = epoch
            if (epoch % cycle_length) <= 500:
                self.depth_decoder.eval()
                # Train TRON Model
                # for param in self.model.projector.parameters():
                #     param.requires_grad = False
                for param in self.model.vision_encoder.parameters():
                    param.requires_grad = True
                for param in self.model.projector.parameters():
                    param.requires_grad = True
                    
                running_loss = []
                running_loss_vpt_inv = []

                bar = tqdm(
                    self.train_data,
                    desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training M2P2: ",
                )
                for data in bar:
                    self.iteration += 1
                    (loss, loss_vpt_inv, grad_norm), t_train = self.forward_batch(data)
                    t_train /= self.train_data.batch_size
                    running_loss.append(loss)
                    running_loss_vpt_inv.append(loss_vpt_inv)

                    # self.m2p2_scheduler.step()
                    # current_wd = self.m2p2_wd_scheduler.step()
                    with self.logger.context_manager("trainer_info"):
                        self.logger.log_metric("train/m2p2/grad_norm", grad_norm, step=self.iteration)
                        self.logger.log_metric("train/m2p2/learning_rate", self.m2p2_optimizer.param_groups[0]['lr'], step=self.iteration)
                        # self.logger.log_metric("train/m2p2/weight_decay", current_wd, step=self.iteration)
                    
                    bar.set_postfix(loss=loss, loss_vpt_inv=loss_vpt_inv, Grad=grad_norm, Time=t_train)
                bar.close()

                # val_loss, _ = self.validate(self.val_data)
                # print(f"Validation Loss for M2P2: {val_loss:.4f}")
                if self.epoch % self.loss_interval == 0:
                    self.m2p2_losses.append(np.mean(running_loss))
                    #self.m2p2_val_loss.append(val_loss)
                    if self.epoch < self.cfg.train_params.bt_start:
                        self.l1_loss.append(np.mean(running_loss))
                # average loss for an epoch
                self.e_loss.append(np.mean(running_loss))  # epoch loss
                    
                # # average loss for an epoch
                # self.e_loss.append(np.mean(running_loss))  # epoch loss

                print(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                    + f"Iteration {self.iteration:05} summary: train Loss: {self.e_loss[-1]:.2f} || "
                    + f"VPT Loss: {np.mean(running_loss_vpt_inv):.2f}"
                )
                self.logger.log_metric("train/m2p2/epoch_loss", self.e_loss[-1], step=self.epoch)
                self.logger.log_metric("train/m2p2/l1_loss", self.l1_loss, step=self.epoch)
                self.logger.log_metric("train/m2p2/epoch_loss_vpt_inv", np.mean(running_loss_vpt_inv), step=self.epoch)
                # for param in self.model.projector.parameters():
                #     param.requires_grad = True
            else:
                self.model.eval()
                self.depth_decoder.eval()
                # Train decoder
                for param in self.model.parameters():
                    param.requires_grad = False
                running_recon_loss = []
                bar = tqdm(self.train_data, desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training Decoder: ")
                for data in bar:
                    self.iteration += 1
                    recon_loss = self.decode_batch(data)
                    running_recon_loss.append(recon_loss)
                    #Step LR and WD schedulers for decoder_optimizer
                    # self.decoder_scheduler.step()
                    # decoder_current_wd = self.decoder_wd_scheduler.step()
                    # with self.logger.context_manager("decoder_info"):
                    #     self.logger.log_metric("learning_rate", self.m2p2_optimizer.param_groups[0]['lr'], step=self.iteration)
                    #     self.logger.log_metric("weight_decay", decoder_current_wd, step=self.iteration)

                    bar.set_postfix(recon_loss=recon_loss)
                                                 

                with torch.no_grad():
                    patch1, patch2, acc, gyro = data
                    patch1 = patch1.to(device=self.device)
                    patch2 = patch2.to(device=self.device)

                    v_encoded_thermal, thermal_features = self.model.vision_encoder(patch1, return_features=True)
                    depth_recon = self.depth_decoder(v_encoded_thermal, thermal_features)

                # Move to CPU and convert to numpy for visualization
                depth_recon_np = depth_recon.detach().cpu().numpy()
                thermal_gt_np = patch1.detach().cpu().numpy()
                depth_gt_np = patch2.detach().cpu().numpy()

                # Let's visualize just the first image in the batch
                # depth_recon_np: (B, 1, H, W) -> (H, W) by selecting the first sample
                recon_img = depth_recon_np[0, 0, :, :]
                gtt_img = thermal_gt_np[0, 0, :, :]
                gt_img = depth_gt_np[0, 0, :, :]

                gt_depth_denorm = gt_img * 20.0
                recon_depth_denorm = recon_img * 20.0

                min_valid_depth = 0.1
                max_valid_depth = 20.0
                valid_mask = (gt_depth_denorm >= min_valid_depth) & (gt_depth_denorm <= max_valid_depth)
                masked_depth = np.copy(gt_depth_denorm)
                masked_depth[~valid_mask] = 0  # Set invalid pixels to zero

                # Ensure the 'depth_vis' directory exists
                os.makedirs('depth_recon', exist_ok=True)
                save_dir = f'depth_recon/depth_reconstruction_epoch_{self.epoch:03}_iter_{self.iteration:05}.png'
                # Plot side-by-side
                # Plot visualization with mask
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 4, 1)
                plt.title("Thermal")
                plt.imshow(gtt_img, cmap='gray')
                plt.axis('off')

                plt.subplot(1, 4, 2)
                plt.title("Ground Truth Depth (Denorm)")
                plt.imshow(gt_depth_denorm, cmap='gray')
                plt.axis('off')

                plt.subplot(1, 4, 3)
                plt.title("Predicted Depth")
                plt.imshow(recon_depth_denorm, cmap='gray')
                plt.axis('off')

                plt.subplot(1, 4, 4)
                plt.title("Masked Depth")
                plt.imshow(masked_depth, cmap='gray')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(save_dir)
                plt.close()

                with self.logger.context_manager("train_images"):
                    self.logger.log_image(save_dir, name=f"train_depth_recon_epoch_{self.epoch:03}_iter_{self.iteration:05}", step=self.epoch)
                bar.close()  
                # Print statistics about the mask
                # valid_percentage = valid_mask.sum() / valid_mask.size * 100
                # print(f"Valid depth pixels: {valid_percentage:.2f}% of image")              

                # Validate decoder
                _ , val_recon_loss = self.validate(self.val_data)
                print(f"Validation Reconstruction Loss: {val_recon_loss:.4f}")

                # self.decoder_scheduler.step(val_recon_loss)
                # # decoder_current_wd = self.decoder_wd_scheduler.step()

                # with self.logger.context_manager("decoder_info"):
                #     self.logger.log_metric("learning_rate", self.decoder_optimizer.param_groups[0]['lr'], step=self.iteration)
                    # self.logger.log_metric("weight_decay", decoder_current_wd, step=self.iteration)

                avg_recon_loss = np.mean(running_recon_loss)
                if self.epoch % self.loss_interval == 0:
                    self.recon_losses.append(np.mean(running_recon_loss))
                    self.recon_val_loss.append(val_recon_loss)
                print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                    f"Iteration {self.iteration:05} summary: Recon Loss: {avg_recon_loss:.2f}")
                
                if avg_recon_loss < self.best_recon_loss:
                    self.best_recon_loss = avg_recon_loss
                    #self.save(name="best_decoder")

            if self.epoch % self.cfg.train_params.save_every == 0:
                self.save(save_best_only=False)  # Regular interval save
            elif (self.e_loss[-1] < self.best * 0.95 and  # Only if 5% better than previous best
                self.epoch >= self.cfg.train_params.start_saving_best):
                self.save(save_best_only=True)  # Save as best model
            

            gc.collect()

        self.plot_losses()
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Training is DONE!")

    @timeit
    def forward_batch(self, data):
        """Forward pass of a batch"""
        self.depth_decoder.eval()
        self.model.train()
        # move data to device
        patch1, patch2, acc, gyro = data
        patch1 = patch1.to(device=self.device)
        patch2 = patch2.to(device=self.device)
        zv1, zv2, _, _, _, _ = self.model(patch1, patch2)
        # Compute L1 loss for alignment
        l1_loss = F.l1_loss(zv1, zv2)
        # compute viewpoint invariance
        loss_vpt_inv = self.model.barlow_loss(zv1, zv2)
        # compute visual-inertial
        # loss_vi = self.model.barlow_loss(zv1, zi)  + 0.5 * self.model.barlow_loss(zv2, zi)

        if self.epoch < self.cfg.train_params.bt_start:
            loss = l1_loss
        else:
            loss = self.cfg.model.l1_coeff * loss_vpt_inv + (1.0 - self.cfg.model.l1_coeff) * l1_loss

        # forward, backward
        self.m2p2_optimizer.zero_grad()
        loss.backward()
        # check grad norm for debugging
        grad_norm = check_grad_norm(self.model)
        # gradient clipping
        if self.cfg.train_params.grad_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train_params.grad_clipping
            )
        # update
        self.m2p2_optimizer.step()
        if self.epoch % 10 == 0: # only plot at certain epochs
            correlation_plot_path = compute_and_plot_correlation(zv1, zv2, self.epoch, self.iteration)
            with self.logger.context_manager("train_correlation"):
                self.logger.log_image(correlation_plot_path, name=f"train_correlation_epoch_{self.epoch:03}_iter_{self.iteration:05}", step=self.epoch)

        return loss.detach().cpu().item(), loss_vpt_inv.detach().item(), grad_norm
    
    def decode_batch(self, data):
        self.model.eval() # ENsure that the m2p2 model is in eval mode
        self.depth_decoder.train() # Set the decoder model to training mode
        patch1, patch2, acc, gyro = data
        patch1 = patch1.to(device=self.device)
        patch2 = patch2.to(device=self.device)
        with torch.no_grad():
            # Obtain encoder output
            # _, _, _, v_encoded_thermal, _, _ = self.model(patch1, patch2, acc, gyro)
            v_encoded_thermal, thermal_features = self.model.vision_encoder(patch1, return_features=True)
        # Forward pass through decoder
        depth_recon = self.depth_decoder(v_encoded_thermal,thermal_features)
        loss = self.recon_criterion(depth_recon, patch2, isTraining = True)
        self.decoder_optimizer.zero_grad()
        loss.backward()
        self.decoder_optimizer.step()
        return loss.cpu().item()
   
    def validate(self, val_data):
        self.model.eval()
        self.depth_decoder.eval()
        running_recon_val_loss = []
        running_m2p2_val_loss = []
        running_metrics = {'abs_rel': [], 'rmse': [], 'delta1': [], 'delta2': [], 'delta3': []}
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_data, desc=f"Epoch {self.epoch:03}, validating...")):
                patch1, patch2, acc, gyro = data
                patch1 = patch1.to(self.device)
                patch2 = patch2.to(self.device)
                # Calculate a global step (epoch * batches) for logging
                num_batches = len(self.train_data) 

                # Get M2P2 losses
                # zv1, zv2, zi, v_encoded_thermal, _, _ = self.model(patch1, patch2, acc, gyro) # TODO: only vision
                v_encoded_thermal, thermal_features = self.model.vision_encoder(patch1, return_features=True)
                # loss_vpt_inv = self.model.barlow_loss(zv1, zv2)
                # loss_vi = self.model.barlow_loss(zv1, zi) + 0.5 * self.model.barlow_loss(zv2, zi)
                # m2p2_loss = self.cfg.model.l1_coeff * loss_vpt_inv + (1.0 - self.cfg.model.l1_coeff) * loss_vi
                
                # Get reconstruction loss
                depth_recon = self.depth_decoder(v_encoded_thermal, thermal_features)
                recon_loss, metrics = self.recon_criterion(depth_recon, patch2, isTraining = False)
                
                # running_m2p2_val_loss.append(m2p2_loss.cpu().item())
                running_recon_val_loss.append(recon_loss.cpu().item())

                for k, v in metrics.items():
                    running_metrics[k].append(v)                
                
                # Visualize every nth batch
                if i % 10 == 0:  # Adjust frequency as needed
                    # Convert tensors to numpy arrays
                    depth_recon_np = depth_recon.cpu().numpy()
                    thermal_gt_np = patch1.cpu().numpy()
                    depth_gt_np = patch2.cpu().numpy()

                    recon_img = depth_recon_np[0, 0, :, :]
                    gtt_img = thermal_gt_np[0, 0, :, :]
                    gtd_img = depth_gt_np[0, 0, :, :]

                    # Perform Depth Anything v2
                    thermal_rgb = cv2.cvtColor((gtt_img*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    depthv2_pred = self.depthv2_model.infer_image(thermal_rgb)

                    # Create visualization directory
                    os.makedirs('validation_vis', exist_ok=True)
                    save_dir = f'validation_vis/val_depth_comparison_epoch{self.epoch}_batch{i}.png'
                    plt.figure(figsize=(15,5))

                    plt.subplot(1,4,1)
                    plt.title("Thermal")
                    plt.imshow(gtt_img, cmap='gray')
                    plt.axis('off')

                    plt.subplot(1,4,2)
                    plt.title("Ground Truth Depth")
                    plt.imshow(gtd_img, cmap='gray')
                    plt.axis('off')

                    plt.subplot(1,4,3)
                    plt.title("Reconstructed Depth")
                    plt.imshow(recon_img, cmap='gray')
                    plt.axis('off')

                    plt.subplot(1,4,4)
                    plt.title("Depth Anything v2 Prediction")
                    plt.imshow(depthv2_pred, cmap = 'gray')
                    plt.axis('off')

                    plt.tight_layout()
                    plt.savefig(f'validation_vis/val_depth_comparison_epoch{self.epoch}_batch{i}.png')
                    plt.close()
                    with self.logger.context_manager("validation_images"):
                        self.logger.log_image(save_dir, name=f"val_depth_recon_epoch_{self.epoch:03}_iter_{self.iteration:05}_batch_{i}", step=self.epoch)

        avg_metrics = {k: np.mean(v) for k, v in running_metrics.items()}
        print(f"Validation Metrics - Abs Rel: {avg_metrics['abs_rel']:.4f}, "
          f"RMSE: {avg_metrics['rmse']:.4f}, "
          f"Delta1: {avg_metrics['delta1']:.4f}, "
          f"Delta2: {avg_metrics['delta2']:.4f}, "
          f"Delta3: {avg_metrics['delta3']:.4f}")
        self.logger.log_metric("val/decoder/epoch_recon_loss", np.mean(running_recon_val_loss), step=self.epoch)
        self.logger.log_metric("val_epoch_abs_rel", avg_metrics['abs_rel'] , step=self.epoch)
        self.logger.log_metric("val_epoch_rmse", avg_metrics['rmse'] , step=self.epoch)
        self.logger.log_metric("val_epoch_delta1", avg_metrics['delta1'] , step=self.epoch)
        self.logger.log_metric("val_epoch_delta2", avg_metrics['delta2'] , step=self.epoch)
        self.logger.log_metric("val_epoch_delta3", avg_metrics['delta3'] , step=self.epoch)
        return np.mean(running_m2p2_val_loss), np.mean(running_recon_val_loss)

    
    def init_m2p2_model(self):
        """Initializes the model"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the model with {self.cfg.model.rep_size} representation dim!")
        # encoder architecture
        vision_encoder = VisionEncoder(latent_size=self.cfg.model.rep_size, num_layers=self.cfg.model.num_layers_enc, pretrained=self.cfg.model.pretrained).to(self.device)
        # imu_encoder = IMUEncoder(latent_size=self.cfg.model.rep_size).to(self.device)
        # # projector head
        # projector = nn.Sequential(
        #     nn.Linear(self.cfg.model.rep_size, self.cfg.model.projection_dim), nn.PReLU(),
        #     nn.Linear(self.cfg.model.projection_dim, self.cfg.model.projection_dim)
        # ).to(self.device)
        projector = Projector(
            in_dim=self.cfg.model.rep_size,
            hidden_dim=4*self.cfg.model.rep_size,
            out_dim=self.cfg.model.rep_size
        ).to(self.device)

        model = TronModel(vision_encoder, projector, latent_size=self.cfg.model.rep_size)

        return model.to(self.device)

    def init_optimizer(self, model, model_type):
        """Initializes the optimizer and learning rate scheduler with model-specific parameters"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the optimizer and scheduler for {model_type}!")

        # --- Optimizer ---
        if model_type.lower() == "m2p2":
            # Use M2P2 specific parameters
            if self.cfg.train_params.optimizer.lower() == "adamw":
                optimizer = optim.AdamW(model.parameters(), **self.cfg.adamw)
            elif self.cfg.train_params.optimizer.lower() == "rmsprop":
                optimizer = optim.RMSprop(model.parameters(), **self.cfg.rmsprop)
            elif self.cfg.train_params.optimizer.lower() == "sgd":
                optimizer = optim.SGD(model.parameters(), **self.cfg.sgd)
            else:
                raise ValueError(f"Unknown optimizer {self.cfg.train_params.optimizer}")
        elif model_type.lower() == "decoder":
            # Use depth decoder specific parameters
            if self.cfg.train_params.optimizer.lower() == "adamw":
                optimizer = optim.AdamW(model.parameters(), **self.cfg.depth_decoder_adamw)
            elif self.cfg.train_params.optimizer.lower() == "rmsprop":
                optimizer = optim.RMSprop(model.parameters(), **self.cfg.rmsprop)
            elif self.cfg.train_params.optimizer.lower() == "sgd":
                optimizer = optim.SGD(model.parameters(), **self.cfg.sgd)
            else:
                raise ValueError(f"Unknown optimizer {self.cfg.train_params.optimizer}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # --- LR Scheduler ---
        # scheduler = WarmupCosineScheduler(
        #     optimizer=optimizer,
        #     steps_per_epoch=len(self.train_data) // self.cfg.dataloader.batch_size,
        #     start_lr=self.cfg.scheduler.start_lr,
        #     base_lr=self.cfg.depth_decoder_adamw.lr if model_type.lower() == "decoder" else self.cfg.adamw.lr,
        #     epochs=self.cfg.train_params.epochs,
        #     warmup_epochs=self.cfg.scheduler.warmup_epochs,
        #     final_lr=self.cfg.scheduler.final_lr,
        # )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='min',                         # Reduce LR when metric stops decreasing
                factor=0.1,                         # Reduce LR by a factor of 10
                patience=5,                         # Wait 5 epochs without improvement before reducing LR
                threshold=0.001,                    # Minimum change to count as improvement
                threshold_mode='rel',               # Use relative threshold
                cooldown=2,                         # Wait 2 epochs after LR reduction before resuming normal operation
                min_lr=self.cfg.scheduler.final_lr, # Minimum LR value
                eps=1e-8                            # Minimum LR update
            )

        # --- WD Scheduler ---
        # wd_scheduler = CosineWDSchedule(
        #     optimizer=optimizer,
        #     epochs=self.cfg.train_params.epochs,
        #     steps_per_epoch=len(self.train_data) // self.cfg.dataloader.batch_size,
        #     init_weight_decay=self.cfg.depth_decoder_adamw.weight_decay if model_type.lower() == "decoder" else self.cfg.adamw.weight_decay,
        #     final_weight_decay=self.cfg.scheduler.final_weight_decay,
        # )

        return optimizer, scheduler, None


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

    def if_resume(self):
        if self.cfg.train_params.resume:
            # load checkpoint
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - LOADING checkpoint!!!")
            save_dir = self.cfg.directory.load
            checkpoint = load_checkpoint(save_dir, self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.m2p2_optimizer.load_state_dict(checkpoint["m2p2_optimizer"])
            self.epoch = checkpoint["epoch"] + 1
            self.e_loss = checkpoint["e_loss"]
            self.iteration = checkpoint["iteration"] + 1
            self.best = checkpoint["best"]
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} "
                + f"LOADING checkpoint was successful, start from epoch {self.epoch}"
                + f" and loss {self.best}"
            )
        else:
            self.epoch = 1
            self.iteration = 0
            self.best = np.inf
            self.e_loss = []

    def save(self, save_best_only=False, name=None):
        checkpoint = {
            "time": str(datetime.now()),
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model": self.model.state_dict(),
            "depth_decoder": self.depth_decoder.state_dict(),
            "m2p2_optimizer": self.m2p2_optimizer.state_dict(),
            "decoder_optimizer": self.decoder_optimizer.state_dict(),
            "m2p2_optimizer_name": type(self.m2p2_optimizer).__name__,
            "decoder_optimizer_name": type(self.decoder_optimizer).__name__,
            "best": self.best,
            "best_recon_loss": self.best_recon_loss,
            "e_loss": self.e_loss,
            "recon_losses": self.recon_losses,
            "recon_val_loss": self.recon_val_loss,
        }

        if name is None:
            save_name = f"{self.cfg.directory.model_name}_{self.epoch}"
        else:
            save_name = name

        if self.e_loss[-1] < self.best:
            self.best = self.e_loss[-1]
            checkpoint["best"] = self.best
            
            # Save best model
            if save_best_only:
                save_checkpoint(checkpoint, True, self.cfg.directory.save, "best_model")
            else:
                # Save both best and regular checkpoint
                save_checkpoint(checkpoint, True, self.cfg.directory.save, "best_model")
                save_checkpoint(checkpoint, False, self.cfg.directory.save, save_name)
        elif not save_best_only:
            # Only save regular checkpoint if not in save_best_only mode
            save_checkpoint(checkpoint, False, self.cfg.directory.save, save_name)


    def plot_losses(self):
        plt.figure(figsize=(12, 6))
        
        # Plot M2P2 Loss
        plt.subplot(1, 2, 1)
        epochs = range(1, len(self.m2p2_losses) + 1)
        plt.plot(epochs, self.m2p2_losses, '#2E86C1', linewidth=2, label='Training Loss')
        plt.title('M2P2 Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot Reconstruction Loss
        plt.subplot(1, 2, 2)
        epochs_recon = range(1, len(self.recon_losses) + 1) 
        plt.plot(epochs_recon, self.recon_losses, '#2E86C1', linewidth=2, label='Training Loss')
        plt.plot(epochs_recon, self.recon_val_loss, '#E74C3C', linewidth=2, label='Validation Loss')
        plt.title('Decoder Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        os.makedirs('loss_plots', exist_ok=True)
        plt.savefig('loss_plots/TrainVal_losses.png')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="./conf/config_m2p2", type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    learner = Learner(cfg_path)
    learner.train()
