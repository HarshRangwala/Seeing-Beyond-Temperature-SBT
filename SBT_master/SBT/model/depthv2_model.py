import sys
from datetime import datetime
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import requests
import logging
from termcolor import cprint
import cv2
import matplotlib.pyplot as plt


def setup_paths(model_name: str = 'Depth-Anything-V2', model_type: str = 'metric_depth'):
    current_dir = Path().resolve()
    current_dir = current_dir.parent
    dv2_base = current_dir / 'TRON'/ 'model' / model_name
    if not dv2_base.exists():
        raise FileNotFoundError(f"Base directory '{dv2_base}' not found.")
    depth_anything_path = dv2_base / model_type

    if not depth_anything_path.exists():
        raise FileNotFoundError(f"Model path '{depth_anything_path}' not found.")
    if str(depth_anything_path) not in sys.path:
        sys.path.append(str(depth_anything_path))
        # print(f"Added {depth_anything_path} to Python Path.")
    
    return depth_anything_path

def load_DepthAnythingv2_model(depth_measurement: str, encoder: str = 'vitl', dataset: str = 'vkitti', max_depth: int = 20):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the Depth Anything V2 model with metric depth and {encoder} encoder.")
    depth_anything_path = setup_paths()
    # print(f"Using depth-anything repository from: {depth_anything_path}")
    from depth_anything_v2.dpt import DepthAnythingV2
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    model_config = model_configs[encoder].copy()
    model_config['max_depth'] = max_depth
    model = DepthAnythingV2(**model_config)
    ckp_name = f'depth_anything_v2_metric_{dataset}_{encoder}.pth'
    ckpt = depth_anything_path / 'checkpoints' / ckp_name
    if not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt}")
    cprint(f'Loading {depth_measurement} dept model with encoder {encoder}', color='yellow')
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only = True))
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    model.eval()
    cprint(f'Model loaded successfully!', color='green')
    return model, device

def infer_depth(model: any, image_path: Path):
    cprint(f"Estimating depth for {image_path}")
    raw_img = cv2.imread(str(image_path))
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        depth = model.infer_image(img)
    return depth



    
