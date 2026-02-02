from pathlib import Path
import pickle
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
import matplotlib.pyplot as plt
import os
import random
import copy

def merge(base_dict: dict, new_dict: dict):
    """Merges two dictionary together, handling both lists and NumPy arrays"""
    # Check if dictionaries have the same keys
    if base_dict.keys() != new_dict.keys():
        print("Warning: Dictionaries have different keys during merge. Skipping mismatched keys.")
        common_keys = set(base_dict.keys()).intersection(set(new_dict.keys()))
    else:
        common_keys = base_dict.keys()
    
    for key in common_keys:
        if key == 'patches_found':
            continue
            
        # Handle different data types appropriately
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
    img = cv2.imread(address, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Warning: Failed to load image from {address}")
        # Return a placeholder black image if the image can't be loaded
        return np.zeros((256, 256), dtype=np.uint8)
    return np.array(img)

def preprocess_thermal(img):
    img = (img - img.mean()) / (img.std() + 1e-6)
    img = torch.clip(img, min=-3, max=2)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img

class DataTransforms:
    @staticmethod
    def get_train_transforms(height, width):
        return A.Compose([
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                always_apply=False, 
                p=0.75, 
                shift_limit_x=(-0.1, 0.1), 
                shift_limit_y=(-0.1, 0.1), 
                rotate_limit=(-10, 10), 
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101, 
                value=0, 
                mask_value=0
            ),
        ])

    @staticmethod
    def get_val_transforms(height, width):
        return A.Compose([
            A.Resize(height, width),
        ])

class RoughnessDataset(Dataset):
    def __init__(self, root: str, resize: tuple[int, int] = (256, 256), seed: int = 42, split='train'):
        torch.manual_seed(seed)
        self.resize = resize
        # folder_name = f"{split}_dt"
        self.root = Path(root) / split  # Assuming root/train and root/validation directory structure
        self.split = split
        
        # Load and merge data from pickle files
        self.data = self._load_data()
        
        # Set appropriate transforms based on split
        if split == 'train':
            self.transforms = DataTransforms.get_train_transforms(resize[0], resize[1])
        elif split in ['validation', 'val']:
            self.transforms = DataTransforms.get_val_transforms(resize[0], resize[1])
        else:
            raise ValueError("split must be either 'train', 'val', or 'validation'")

    def _load_data(self):
        files = list(self.root.glob("*.pkl"))
        
        if not files:
            raise FileNotFoundError(f"No pickle files found in {self.root}")
        
        all_data = {}
        for file in files:
            with file.open("rb") as f:
                data = pickle.load(f)
            
            if bool(all_data):
                all_data = merge(all_data, data)
            else:
                all_data = data
        
        # print(f"Loaded {len(all_data['thermal_paths'])} samples for {self.split} split")
        return all_data

    def __len__(self):
        return len(self.data['thermal_paths'])

    def __getitem__(self, idx):
        try:
            # Load thermal image
            thermal = imread(self.data['thermal_paths'][idx])
            augmented = self.transforms(image = thermal)
            thermal = augmented['image']
            
            thermal = torch.tensor(thermal, dtype=torch.float32)        
            thermal = preprocess_thermal(thermal)
            thermal = thermal.unsqueeze(0)  # Add channel dimension: [H, W] -> [1, H, W]
            
            # Get roughness score (ground truth)
            roughness_score = torch.tensor(self.data['roughness_score'][idx], dtype=torch.float32)
            
            return thermal,  roughness_score
        
        except Exception as e:
            print(f"Error processing item at index {idx}: {str(e)}")
            # Return placeholder data in case of error
            return (torch.zeros(1, self.resize[0], self.resize[1]),
                    torch.tensor(0.0))


# if __name__ == "__main__":
#     # Create dataset
#     train_dataset = RoughnessDataset(
#         root="/mnt/sbackup/Server_3/harshr/m2p2_data",  # Root directory containing train and validation folders
#         resize=(256, 256),
#         split="validation"
#     )
    
#     # Create dataloader
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=8,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )

    # for batch_idx, (thermal, roughness_score) in enumerate(train_dataloader):
    #     print(thermal.shape, "Thermal Shape")
    #     print(roughness_score.shape, "Roughness score Shape")
        
    
