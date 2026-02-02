from pathlib import Path
import copy
import pickle
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from scipy.signal import periodogram
import matplotlib.pyplot as plt
import os
import random

def merge(base_dict: dict, new_dict: dict):
    """Merges two dictionary together, handling both lists and NumPy arrays"""
    if base_dict.keys() != new_dict.keys():
        print("Warning: Dictionaries have different keys during merge. Skipping mismatched keys.")
        common_keys = set(base_dict.keys()).intersection(set(new_dict.keys()))
    else:
        common_keys = base_dict.keys()
    
    for key in common_keys:
        if key == 'patches_found':
            continue
        if isinstance(base_dict[key], list):
            if isinstance(new_dict[key], list):
                base_dict[key].extend(new_dict[key])
            elif isinstance(new_dict[key], np.ndarray):
                base_dict[key].extend(new_dict[key].tolist())
        elif isinstance(base_dict[key], np.ndarray):
            if isinstance(new_dict[key], np.ndarray):
                base_dict[key] = np.concatenate([base_dict[key], new_dict[key]])
            elif isinstance(new_dict[key], list):
                base_dict[key] = np.concatenate([base_dict[key], np.array(new_dict[key])])
    return base_dict

def imread(address: str):
    # Reads in BGR or Grayscale depending on image
    img = cv2.imread(address, cv2.IMREAD_UNCHANGED)
    return np.array(img) if img is not None else None

def preprocess_thermal(img):
    """
    Applies CLAHE and Normalization.
    RETURNS: Numpy Array (H, W) float32, range [0, 1]
    """
    # Ensure image is uint8 for CLAHE
    if img.dtype != np.uint8:
        # Normalize to 0-255 if it's not already
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    cl_img = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5, 5))
    cl_img = cl_img.apply(img)
    
    # Convert to float32 and normalize to 0-1
    # We return Numpy here so Albumentations can process it later
    cl_img = cl_img.astype(np.float32) / 255.0
    return cl_img

class DataTransforms:
    """
    Defines the Geometry Pipeline: Crop -> Resize -> Augment
    """
    @staticmethod
    def get_train_transforms(target_h, target_w, crop_top, crop_bottom, orig_w=1280, orig_h=1024):
        return A.Compose([
           # 1. Smart Crop (Geometry Preservation)
           # Removes the U-shape black borders
           A.Crop(x_min=0, y_min=crop_top, x_max=orig_w, y_max=orig_h - crop_bottom),
           
           # 2. Resize (Downsampling)
           # INTER_NEAREST is safest for Depth to avoid creating fake values
           A.Resize(target_h, target_w, interpolation=cv2.INTER_NEAREST),
           
           # 3. Augmentation (Applied to both Thermal and Depth equally)
           A.HorizontalFlip(p=0.5),
           ], additional_targets={'depth_img': 'image'})

    @staticmethod
    def get_val_transforms(target_h, target_w, crop_top, crop_bottom, orig_w=1280, orig_h=1024):
        return A.Compose([
            # 1. Smart Crop
            A.Crop(x_min=0, y_min=crop_top, x_max=orig_w, y_max=orig_h - crop_bottom),
            
            # 2. Resize
            A.Resize(target_h, target_w, interpolation=cv2.INTER_NEAREST),
        ], additional_targets={'depth_img': 'image'})

class TronDataset(Dataset):
    def __init__(self, root: str, stats: str, resize: tuple[int, int]=(256, 256), frequency_rate: int = 200, seed: int=42, split = 'train'):
        torch.manual_seed(seed)
        self.resize = resize
        
        # --- UPDATED CROP VALUES ---
        self.crop_top = 165
        self.crop_bottom = 74
        # ---------------------------
        
        self.clip_distance = 30.0
        self.reg_factor = 3.7
        self.frequency_rate = frequency_rate  
        data_root = Path(root) / split
        
        files = list(data_root.glob("*.pkl"))
        self.data = dict()
        for file in files:
            with file.open("rb") as f:
                data = pickle.load(f)
            if bool(self.data):
                self.data = merge(self.data, data)
            else:
                self.data = data
        
        self.stats = None
        with open(stats, 'rb') as f:
            self.stats = pickle.load(f)

        # --- Initialize Transforms with Crop Values ---
        if split == 'train':
            self.transforms = DataTransforms.get_train_transforms(
                target_h=resize[0], target_w=resize[1],
                crop_top=self.crop_top, crop_bottom=self.crop_bottom
            )
        elif split == 'validation':
            self.transforms = DataTransforms.get_val_transforms(
                target_h=resize[0], target_w=resize[1],
                crop_top=self.crop_top, crop_bottom=self.crop_bottom
            )
        else:
            raise ValueError("split must be either 'train' or 'validation'")

    def __len__(self):
        return len(self.data['time_stamp'])
    
    def __getitem__(self, idx):
        try:
            # 1. READ THERMAL
            thermal_path = self.data['thermal_paths'][idx]
            thermal = imread(thermal_path)
            if thermal is None:
                print(f"WARNING: None type found at: {thermal_path}", flush=True)
                return None 

            # 2. PREPROCESS THERMAL (CLAHE / Normalize) -> Returns Numpy
            # thermal = preprocess_thermal(thermal)
            
            # 3. READ DEPTH (Raw Metric Float16/32)
            depth_path = self.data['depth_paths'][idx]
            depth = np.load(depth_path)
            if depth is None:
                print(f"WARNING: None type found at: {depth_path}", flush=True)
                return None
            
            if depth.dtype != 'float32':
                depth = depth.astype('float32')
            
            # 4. APPLY GEOMETRIC TRANSFORMS (CROP -> RESIZE -> FLIP)
            # Albumentations handles both images simultaneously to ensure alignment
            augmented = self.transforms(image=thermal, depth_img=depth)
            thermal = augmented['image']
            depth = augmented['depth_img']
            
            # 5. CONVERT TO TENSOR
            thermal = torch.tensor(thermal, dtype=torch.float32) / 255.0
            # print(thermal.min(), thermal.max())
            depth = torch.tensor(depth, dtype=torch.float32)
            
            # 6. DEPTH SCALING (Metric -> Log Space)
            # We do this AFTER crop/resize to ensure we transform valid pixels
            depth = torch.clamp(depth, min=0.1, max=self.clip_distance)
            depth = depth / self.clip_distance  # Normalize to [0, 1]
            depth = 1.0 + torch.log(depth) / self.reg_factor  # Apply log scaling
            depth = torch.clamp(depth, min=0.0, max=1.0)

            thermal = (thermal - 0.495356) / 0.191781
            # depth = (depth - 0.561041) / 0.295559
            
            # 7. ADD CHANNEL DIMENSION (H, W) -> (1, H, W)
            thermal = thermal.unsqueeze(0)
            depth = depth.unsqueeze(0)
            
            # 8. LOAD IMU (Accel & Gyro)
            # Shapes: (1200,) each
            accel = self.data['accel_msg'][idx]
            gyro = self.data['gyro_msg'][idx]
            
            # Convert to tensor
            accel = torch.tensor(accel, dtype=torch.float32)
            gyro = torch.tensor(gyro, dtype=torch.float32)

            return thermal, depth, accel, gyro

        except Exception as e:
            print(f"ERROR loading or processing data at index {idx}: {str(e)}", flush=True)
            return None

class BCDataset(Dataset):
    def __init__(self, root: str, stats: str, resize=(256, 256), frequency_rate=200, seed=42, split = 'train'):
        torch.manual_seed(seed)
        self.resize = resize
        self.frequency_rate = frequency_rate
        folder_name = f"{split}_dt4"
        data_root = Path(root) / folder_name
        print(data_root, "ROOT")
        files = list(data_root.glob("*.pkl"))
        self.data = dict()
        for file in files:
            with file.open("rb") as f:
                data = pickle.load(f)
            if bool(self.data):
                self.data = merge(self.data, data)
            else:
                self.data = data

        with open(stats, 'rb') as f:
            self.stats = pickle.load(f)

        first_thermal = imread(self.data['thermal_paths'][0])
        height, width = first_thermal.shape[:2]
        if split == 'train':
            self.transforms = DataTransforms.get_train_transforms(height, width)
        elif split == 'validation':
            self.transforms = DataTransforms.get_val_transforms(height, width)
        else:
            raise ValueError("split must be either 'train' or 'val'")
        self.resize_transform = transforms.Resize(self.resize, antialias=True)

    def __len__(self):
        return len(self.data['time_stamp'])

    def __getitem__(self, idx):
        # In place of patch 1, we use thermal images
        thermal = imread(self.data['thermal_paths'][idx])
        # Apply augmentation
        augmented = self.transforms(image=copy.deepcopy(thermal))
        thermal = augmented['image']

        thermal = torch.tensor(thermal, dtype=torch.float32)
        thermal = preprocess_thermal(thermal)
        thermal = thermal.unsqueeze(0)
        thermal = self.resize_transform(thermal)          
        # depth = self.resize_transform(depth)
        # normalize the image patches and cast to torch tensor
        # patch1 = torch.tensor(np.asarray(patch1, dtype=np.float32) / 255.0).permute(2, 0, 1)
        # patch2 = torch.tensor(np.asarray(patch2, dtype=np.float32) / 255.0).permute(2, 0, 1)

        cmd_vel_msg = self.generate_tensor(self.data['sm_cmd_vel'][idx])
        # print(cmd_vel_msg, "IN DATALOADER CMD VEL")
        # cmd_vel_msg = (cmd_vel_msg - self.stats['sm_cmd_vel_mean']) / (self.stats['sm_cmd_vel_std'] + 0.000006)
        # print(cmd_vel_msg, "IN DATALOADER CMD VEL AFTER NORMALIZATION")
        cmd_vel_msg = cmd_vel_msg[-2:]
        return thermal, cmd_vel_msg
    def generate_tensor(self, data):
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        elif isinstance(data, list):
            return torch.tensor(data).float()
        elif isinstance(data, tuple):
            return torch.tensor(data).float()
        elif isinstance(data, torch.Tensor):
            return data.float()
# if __name__ == "__main__":
# pass