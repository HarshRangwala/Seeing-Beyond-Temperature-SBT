"""General utility functions"""
from time import time
from datetime import datetime
import functools
import logging

import comet_ml
import torch
import torch.optim as optim
from omegaconf import OmegaConf, DictConfig
import numpy as np
from rich import print

import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def compute_and_plot_correlation(thermal_features, depth_features, epoch, iteration, stage="train"):
    """
    Computes and plots correlation matrices.

    Args:
        thermal_features: PyTorch tensor of thermal features (B x N)
        depth_features: PyTorch tensor of depth features (B x N)
        epoch: Current epoch number.
        iteration: Current iteration number.
        stage: "train" or "val"
    """

    # 1. Move to CPU and convert to NumPy
    thermal_features = thermal_features.detach().cpu().numpy()
    depth_features = depth_features.detach().cpu().numpy()

    # 2. Compute Auto-correlation Matrices
    thermal_auto_corr = np.corrcoef(thermal_features, rowvar=False)  # rowvar=False means columns are variables
    depth_auto_corr = np.corrcoef(depth_features, rowvar=False)

    # 3. Compute Cross-correlation Matrix
    cross_corr = np.corrcoef(thermal_features, depth_features, rowvar=False)
    # Take only the relevant part of the cross-correlation (thermal vs. depth)
    cross_corr = cross_corr[:thermal_features.shape[1], thermal_features.shape[1]:]


    # 4. Visualize (using matplotlib)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    # Thermal Auto-correlation
    im1 = axes[0].imshow(thermal_auto_corr, cmap="RdBu", vmin=-1, vmax=1)
    axes[0].set_title("Thermal Auto-correlation")
    axes[0].set_xlabel("Feature Dimension")
    axes[0].set_ylabel("Feature Dimension")
    fig.colorbar(im1, ax=axes[0])

    # Depth Auto-correlation
    im2 = axes[1].imshow(depth_auto_corr, cmap="RdBu", vmin=-1, vmax=1)
    axes[1].set_title("Depth Auto-correlation")
    axes[1].set_xlabel("Feature Dimension")
    axes[1].set_ylabel("Feature Dimension")
    fig.colorbar(im2, ax=axes[1])

    # Cross-correlation
    im3 = axes[2].imshow(cross_corr, cmap="RdBu", vmin=-1, vmax=1)
    axes[2].set_title("Thermal-Depth Cross-correlation")
    axes[2].set_xlabel("Depth Feature Dimension")
    axes[2].set_ylabel("Thermal Feature Dimension")
    fig.colorbar(im3, ax=axes[2])


    plt.tight_layout()
    os.makedirs('correlation_plots', exist_ok=True)
    save_path = f'correlation_plots/{stage}_correlation_epoch_{epoch:03}_iter_{iteration:05}.png'
    plt.savefig(save_path)
    plt.close(fig)
    return save_path 

def get_conf(name: str):
    """Returns yaml config file in DictConfig format

    Args:
        name: (str) name of the yaml file
    """
    name = name if name.split(".")[-1] == "yaml" else name + ".yaml"
    cfg = OmegaConf.load(name)
    return cfg

def init_device(cfg):
        """Initializes the device"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the device!")
        is_cuda_available = torch.cuda.is_available()
        device = cfg.train_params.device

        if "cpu" in device:
            print(f"Performing all the operations on CPU.")
            return torch.device(device)

        elif "cuda" in device:
            if is_cuda_available:
                device_idx = device.split(":")[1]
                if device_idx == "a":
                    print(
                        f"Performing all the operations on CUDA; {torch.cuda.device_count()} devices."
                    )
                    cfg.dataloader.batch_size *= torch.cuda.device_count()
                    return torch.device(device.split(":")[0])
                else:
                    print(f"Performing all the operations on CUDA device {device_idx}.")
                    return torch.device(device)
            else:
                print("CUDA device is not available, falling back to CPU!")
                return torch.device("cpu")
        else:
            raise ValueError(f"Unknown {device}!")

def init_logger(cfg: DictConfig):
    """Initializes the cometml logger

    Args:
        cfg: (DictConfig) the configuration
    """
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the logger!")
    logger = None
    cfg_full = cfg
    cfg = cfg.logger
    # Check to see if there is a key in environment:
    EXPERIMENT_KEY = cfg.experiment_key
    print(EXPERIMENT_KEY)
    # First, let's see if we continue or start fresh:
    CONTINUE_RUN = cfg.resume
    if EXPERIMENT_KEY and CONTINUE_RUN:
        # There is one, but the experiment might not exist yet:
        api = comet_ml.API()  # Assumes API key is set in config/env
        try:
            api_experiment = api.get_experiment_by_id(EXPERIMENT_KEY)
        except Exception:
            api_experiment = None
        if api_experiment is not None:
            CONTINUE_RUN = True
            # We can get the last details logged here, if logged:
            # step = int(api_experiment.get_parameters_summary("batch")["valueCurrent"])
            # epoch = int(api_experiment.get_parameters_summary("epochs")["valueCurrent"])

    if CONTINUE_RUN:
        # 1. Recreate the state of ML system before creating experiment
        # otherwise it could try to log params, graph, etc. again
        # ...
        # 2. Setup the existing experiment to carry on:
        logger = comet_ml.ExistingExperiment(
            previous_experiment=EXPERIMENT_KEY,
            log_env_details=cfg.log_env_details,  # to continue env logging
            log_env_gpu=True,  # to continue GPU logging
            log_env_cpu=True,  # to continue CPU logging
            auto_histogram_weight_logging=True,
            auto_histogram_gradient_logging=True,
            auto_histogram_activation_logging=True,
        )
        # Retrieved from above APIExperiment
        # logger.set_epoch(epoch)

    else:
        # 1. Create the experiment first
        #    This will use the COMET_EXPERIMENT_KEY if defined in env.
        #    Otherwise, you could manually set it here. If you don't
        #    set COMET_EXPERIMENT_KEY, the experiment will get a
        #    random key!
        if cfg.online:
            logger = comet_ml.Experiment(
                api_key=cfg.api_key,
                disabled=cfg.disabled,
                project_name=cfg.project,
                log_env_details=cfg.log_env_details,
                log_env_gpu=True,  # to continue GPU logging
                log_env_cpu=True,  # to continue CPU logging
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=True,
                auto_histogram_activation_logging=True,
            )
            logger.set_name(cfg.experiment_name)
            logger.add_tags(cfg.tags.split())
            logger.log_parameters(cfg_full)
        else:
            logger = comet_ml.OfflineExperiment(
                disabled=cfg.disabled,
                project_name=cfg.project,
                offline_directory=cfg.offline_directory,
                auto_histogram_weight_logging=True,
                log_env_details=cfg.log_env_details,
                log_env_gpu=True,  # to continue GPU logging
                log_env_cpu=True,  # to continue CPU logging
            )
            logger.set_name(cfg.experiment_name)
            logger.add_tags(cfg.tags.split())
            logger.log_parameters(cfg_full)

    return logger


def timeit(fn):
    """Calculate time taken by fn().

    A function decorator to calculate the time a function needed for completion on GPU.
    returns: the function result and the time taken
    """
    # first, check if cuda is available
    cuda = True if torch.cuda.is_available() else False
    if cuda:

        @functools.wraps(fn)
        def wrapper_fn(*args, **kwargs):
            torch.cuda.synchronize()
            t1 = time()
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            t2 = time()
            take = t2 - t1
            return result, take

    else:

        @functools.wraps(fn)
        def wrapper_fn(*args, **kwargs):
            t1 = time()
            result = fn(*args, **kwargs)
            t2 = time()
            take = t2 - t1
            return result, take

    return wrapper_fn


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def print_model_stats(model):
    """Counts parameters"""
    if 'depth_decoder' in str(model):
        print("Depth Decoder \n")
    else:
        print("SSL model \n")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2

    print("\n## Architecture Statistics")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {size_mb:.2f} MB")
    
    # Print layer-wise information
    print("\n** Layer Details **")
    for name, layer in model.named_children():
        params = sum(p.numel() for p in layer.parameters())
        print(f"{name}: {params:,} parameters")