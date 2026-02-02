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
            A.HorizontalFlip(p=0.65),
            # A.ShiftScaleRotate(
            #     always_apply=False, 
            #     p=0.75, 
            #     shift_limit_x=(-0.1, 0.1), 
            #     shift_limit_y=(-0.1, 0.1), 
            #     rotate_limit=(-13, 13), 
            #     interpolation=cv2.INTER_LINEAR,
            #     border_mode=cv2.BORDER_REFLECT_101, 
            #     value=0, 
            #     mask_value=0
            # ),
        ], additional_targets={'mask': 'mask'})

    @staticmethod
    def get_val_transforms(height, width):
        return A.Compose([
            A.Resize(height, width),
        ], additional_targets={'mask': 'mask'})

class TraversabilityDataset(Dataset):
    def __init__(self, root: str, resize: tuple[int, int] = (256, 256), seed: int = 42, split='train'):
        torch.manual_seed(seed)
        self.resize = resize
        folder_name = f"{split}"
        self.root = Path(root) / folder_name  # Assuming root/train and root/validation directory structure
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
        
        print(f"Loaded {len(all_data['thermal_paths'])} samples for {self.split} split")
        return all_data

    def __len__(self):
        return len(self.data['thermal_paths'])

    def __getitem__(self, idx):
        # Load thermal image
        thermal_path = self.data['thermal_paths'][idx]
        thermal_img = imread(thermal_path)
        
        # Load traversability mask
        mask_path = self.data['traversability_mask_paths'][idx]
        mask_img = imread(mask_path)
        
        # Apply transformations
        if self.transforms:
            transformed = self.transforms(image=thermal_img, mask=mask_img)
            thermal_img = transformed['image']
            mask_img = transformed['mask']
        
        # Convert to tensor and preprocess
        thermal_img = torch.tensor(thermal_img, dtype=torch.float32)
        thermal_img = preprocess_thermal(thermal_img)
        thermal_img = thermal_img.unsqueeze(0)  # Add channel dimension: [H, W] -> [1, H, W]
        
        # Process mask
        if len(mask_img.shape) == 3 and mask_img.shape[2] == 3:
            # Convert RGB mask to grayscale if needed
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
        
        mask_img = torch.tensor(mask_img, dtype=torch.float32).unsqueeze(0) / 255.0
        
        return thermal_img, mask_img

    def visualize_samples(self, num_samples=4):
        """Visualize random samples from the dataset"""
        indices = random.sample(range(len(self)), min(num_samples, len(self)))
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))
        
        for i, idx in enumerate(indices):
            thermal, mask = self[idx]
            
            # Convert tensors for visualization
            thermal_img = thermal.squeeze().cpu().numpy()
            mask_img = mask.squeeze().cpu().numpy()
            
            # Plot thermal image
            axes[i, 0].imshow(thermal_img, cmap='gray')
            axes[i, 0].set_title(f"Thermal {idx}")
            axes[i, 0].axis('off')
            
            # Plot traversability mask
            axes[i, 1].imshow(mask_img, cmap='gray')
            axes[i, 1].set_title(f"Traversability Mask {idx}")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.show()


# # Usage example:
# if __name__ == "__main__":
#     # Create dataset
#     train_dataset = TraversabilityDataset(
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
    
#     # Visualize some samples
#     train_dataset.visualize_samples(4)
    
#     # Test dataloader
#     for batch_idx, (thermal, mask) in enumerate(train_dataloader):
#         print(f"Batch {batch_idx}: Thermal shape: {thermal.shape}, Mask shape: {mask.shape}")
#         if batch_idx == 0:
#             # Visualize first batch
#             fig, axes = plt.subplots(2, 4, figsize=(16, 8))
#             for i in range(min(4, thermal.size(0))):
#                 # Show thermal
#                 axes[0, i].imshow(thermal[i, 0].cpu().numpy(), cmap='gray')
#                 axes[0, i].set_title(f"Thermal {i}")
#                 axes[0, i].axis('off')
                
#                 # Show mask
#                 axes[1, i].imshow(mask[i, 0].cpu().numpy(), cmap='gray')
#                 axes[1, i].set_title(f"Mask {i}")
#                 axes[1, i].axis('off')
            
#             plt.tight_layout()
#             plt.savefig('dataloader_batch_preview.png')
#             plt.close()
        
#         # Only process a few batches for testing
#         if batch_idx >= 2:
#             break
