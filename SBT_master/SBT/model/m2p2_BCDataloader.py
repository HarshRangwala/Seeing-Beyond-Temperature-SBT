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
    try:
        img = cv2.imread(address, cv2.IMREAD_UNCHANGED)
        return np.array(img) if img is not None else None
    except Exception as e:
        print(f"Error reading image at {address}: {str(e)}", flush=True)
        return None

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
           A.HorizontalFlip(p=0.5), # 50% chance of hflip
        #    A.Rotate(limit=(-10, 10), interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=0.6)
        A.ShiftScaleRotate( always_apply=False, p=0.75, shift_limit_x=(-0.1, 0.1), 
                           shift_limit_y=(-0.1, 0.1), rotate_limit=(-13, 13), interpolation=cv2.INTER_LINEAR,
                             border_mode=0, value=0,  mask_value=0),
        ], additional_targets={'depth_img': 'image'})
    @staticmethod
    def get_val_transforms(height, width):
        # Validation should only have essential transforms, no augmentations
        return A.Compose([
            A.Resize(height, width),
        ], additional_targets={'depth_img': 'image'})

class TronDataset(Dataset):
    def __init__(self, root: str, stats: str, resize: tuple[int, int]=(256, 256), frequency_rate: int = 200, seed: int=42, split = 'train'):
        torch.manual_seed(seed)
        self.resize = resize
        self.crop_top = 40
        self.crop_bottom = 225
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
        # load stats
        self.stats = None
        with open(stats, 'rb') as f:
            self.stats = pickle.load(f)

        # first_thermal = imread(self.data['thermal_paths'][0])
        # height, width = first_thermal.shape[:2]
        if split == 'train':
            self.transforms = DataTransforms.get_train_transforms(resize[0], resize[1])
        elif split == 'validation':
            self.transforms = DataTransforms.get_val_transforms(resize[0], resize[1])
        else:
            raise ValueError("split must be either 'train' or 'val'")
        self.resize_transform = transforms.Resize(self.resize, antialias=True)

    def __len__(self):
        return len(self.data['time_stamp'])
    
    def __getitem__(self, idx):
        # read thermal images
        thermal_path = self.data['thermal_paths'][idx]
        try:
            thermal = imread(thermal_path)
            if thermal is None:
                print(f"WARNING: None type found at: {thermal_path}", flush=True)
                return None 
            # Resize thermal image to 256x256
            thermal_resized = cv2.resize(thermal, (256, 256), interpolation=cv2.INTER_AREA)
            thermal = thermal_resized[self.crop_top:self.crop_bottom, :]

            # read depth images
            # Read depth images
            depth_path = self.data['depth_paths'][idx]
            depth = imread(depth_path)
            if depth is None:
                print(f"WARNING: None type found at: {depth_path}", flush=True)
                return None

            # read elevation
            elevation_path = self.data['elevation_image_paths'][idx]
            elevation = imread(elevation_path)
            if elevation is None:
                print(f"WARNING: None type found at: {elevation_path}", flush=True)
                return None
            elevation = elevation[self.crop_top:150, :]
            elevation = cv2.resize(elevation, (256, 256), interpolation=cv2.INTER_AREA)

            # Apply augmentation
            augmented = self.transforms(image=copy.deepcopy(thermal), depth_img=copy.deepcopy(depth))
            thermal = augmented['image']
            depth = augmented['depth_img']
            
            thermal = torch.tensor(thermal, dtype=torch.float32)
            depth = torch.tensor(depth, dtype=torch.float32) / 255.0
            elevation = torch.tensor(elevation, dtype=torch.float32) / 255.0

            thermal = preprocess_thermal(thermal)
            # if idx < 4: # Visualize first 4 images of the batch

            #     plt.figure(figsize=(8, 4))
            #     plt.subplot(1, 2, 1)
            #     plt.title(f"Augmented Thermal (Index {idx})")
            #     plt.imshow(thermal, cmap='gray')
            #     plt.subplot(1, 2, 2)
            #     plt.title(f"Augmented Depth (Index {idx})")
            #     plt.imshow(depth, cmap='gray')
            #     plt.tight_layout()
            #     os.makedirs('dataloader_augmentations_debug', exist_ok=True)
            #     plt.savefig(f'dataloader_augmentations_debug/aug_index_{idx}.png')
            #     plt.close()
            thermal = thermal.unsqueeze(0)
            depth = depth.unsqueeze(0)
            elevation = elevation.unsqueeze(0)
            # thermal = self.resize_transform(thermal)          
            # depth = self.resize_transform(depth)
            
            # normalize the image patches and cast to torch tensor
            # patch1 = torch.tensor(np.asarray(patch1, dtype=np.float32) / 255.0).permute(2, 0, 1)
            # patch2 = torch.tensor(np.asarray(patch2, dtype=np.float32) / 255.0).permute(2, 0, 1)

            accel_msg = torch.tensor(self.data['accel_msg'][idx])
            gyro_msg = torch.tensor(self.data['gyro_msg'][idx])

            accel_msg = accel_msg.view(2, self.frequency_rate, 3)
            gyro_msg = gyro_msg.view(2, self.frequency_rate, 3)

            accel_msg = (accel_msg - self.stats['accel_mean']) / (self.stats['accel_std'] + 0.000006)
            gyro_msg = (gyro_msg - self.stats['gyro_mean']) / (self.stats['gyro_std'] + 0.000006)

            accel_msg = periodogram(accel_msg.view(2 * self.frequency_rate, 3), fs=self.frequency_rate, axis=0)[1]
            gyro_msg = periodogram(gyro_msg.view(2 * self.frequency_rate, 3), fs=self.frequency_rate, axis=0)[1]

            return thermal, depth, elevation, torch.from_numpy(accel_msg).float(), torch.from_numpy(gyro_msg).float()
        except Exception as e:
            print(f"ERROR loading or processing data at index {idx}, path {thermal_path}: {str(e)}", flush=True)
            return None


class BCDataset(Dataset):
    def __init__(self, root: str, stats: str, resize=(256, 256), seed=42, split='train', use_smoothed=True):
        torch.manual_seed(seed)
        self.resize = resize
        self.crop_top = 40
        self.crop_bottom = 225
        self.use_smoothed = use_smoothed
        folder_name = f"{split}_dt4"
        data_root = Path(root) / folder_name
        
        files = list(data_root.glob("*.pkl"))
        self.data = dict()
        for file in files:
            with file.open("rb") as f:
                data = pickle.load(f)
            if bool(self.data):
                self.data = merge(self.data, data)
            else:
                self.data = data
        # load stats
        self.stats = None
        with open(stats, 'rb') as f:
            self.stats = pickle.load(f)
        
        self.cmd_vel_key = 'sm_cmd_vel' if use_smoothed and 'sm_cmd_vel' in self.data else 'cmd_vel_msg'
        stats_key_prefix = 'sm_cmd_vel' if use_smoothed else 'cmd_vel'
        # Verify that required statistics are present
        required_stats = [f"{stats_key_prefix}_mean", f"{stats_key_prefix}_std"]
        for stat in required_stats:
            if stat not in self.stats:
                raise ValueError(f"Required statistic {stat} not found in stats file")
            
        # Setup transforms
        if split == 'train':
            self.transforms = DataTransforms.get_train_transforms(resize[0], resize[1])
        elif split == 'validation':
            self.transforms = DataTransforms.get_val_transforms(resize[0], resize[1])
        else:
            raise ValueError("split must be either 'train' or 'validation'")
            
        self.resize_transform = transforms.Resize(self.resize, antialias=True)

    def __len__(self):
        """Return the number of valid samples"""
        return len(self.data['time_stamp'])

    def __getitem__(self, idx):
        """Get a sample by index, ensuring proper alignment between thermal and command velocity"""
        
        
        # Load thermal image
        thermal_path = self.data['thermal_paths'][idx]
        thermal = imread(thermal_path)
        
        if thermal is None:
            print(f"WARNING: None type found at: {thermal_path}", flush=True)
            # Return a default or skip this sample
            # Here we'll return a zero image and zero command
            thermal = np.zeros((256, 256), dtype=np.uint8)
            cmd_vel = torch.zeros(2, dtype=torch.float32)
            return thermal, cmd_vel
            
        # Process thermal image
        thermal_resized = cv2.resize(thermal, (256, 256), interpolation=cv2.INTER_AREA)
        thermal = thermal_resized[self.crop_top:self.crop_bottom, :]
        # thermal = cv2.resize(thermal, (256, 256), interpolation=cv2.INTER_AREA)
        augmented = self.transforms(image=copy.deepcopy(thermal))
        thermal = augmented['image']

        # Convert thermal to tensor and preprocess
        thermal = torch.tensor(thermal, dtype=torch.float32)
        thermal = preprocess_thermal(thermal)
        thermal = thermal.unsqueeze(0)  # Add channel dimension

        # Get the corresponding command velocity
        cmd_vel = self.generate_tensor(self.data[self.cmd_vel_key][idx])
        
        # Determine which statistics to use based on whether we're using smoothed commands
        stats_key_prefix = 'sm_cmd_vel' if self.use_smoothed else 'cmd_vel'
        
        # Normalize the command velocity
        cmd_vel = (cmd_vel - self.stats[f"{stats_key_prefix}_mean"]) / (self.stats[f"{stats_key_prefix}_std"] + 1e-6)
        
        # Extract the last 2 values (linear and angular velocity)
        if len(cmd_vel.shape) > 1 and cmd_vel.shape[0] > 2:
            cmd_vel = cmd_vel[-2:]
        cmd_vel = cmd_vel[-2:]
        return thermal, cmd_vel.float()
    
    def generate_tensor(self, data):
        """Convert various data types to a torch tensor"""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        elif isinstance(data, list):
            return torch.tensor(data).float()
        elif isinstance(data, tuple):
            return torch.tensor(data).float()
        elif isinstance(data, torch.Tensor):
            return data.float()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
# if __name__ == "__main__":
# # #     # pass
#     root = '/mnt/sbackup/Server_3/harshr/m2p2_data/'
#     stats_path = '/home/harshr/NV_cahsor/CAHSOR-master/DataProcessingPipeline/BC_dt4_stats.pkl'
#     dataset = BCDataset(root=root, stats=stats_path, split='train')
#     val_dataset = BCDataset(root=root, stats=stats_path, split='validation')
#     print(f"Train dataset size: {len(dataset)}")
#     print(f"Validation dataset size: {len(val_dataset)}")

#     thermal, cmd_vel_msg = dataset[0]
#     print("Sample from train dataset:")
#     print(f"Thermal shape: {thermal.shape}, Command Velocity Message shape: {cmd_vel_msg.shape}")

#     thermal, cmd_vel_msg = val_dataset[0]
#     print("Sample from validation dataset:")
#     print(f"Thermal shape: {thermal.shape}, Command Velocity Message shape: {cmd_vel_msg.shape}")

#     # random_indices = random.sample(range(len(dataset)), 4)
#     random_indices = [799, 49, 140, 70]

#     fig, axs = plt.subplots(4, 1, figsize=(18, 16)) 
#     fig.suptitle("Random Samples: Thermal Images")

#     for i, idx in enumerate(random_indices):
#         thermal, cmd_vel_msg = dataset[idx]
        
#         thermal_img = thermal.squeeze().numpy()

#         axs[i].imshow(thermal_img, cmap='gray')
#         axs[i].set_title(f"Thermal {idx}\nShape: {thermal.shape}")
#         axs[i].axis('off')

#         print(f"Sample {i+1}:")
#         print(f"Thermal file: {dataset.data['thermal_paths'][idx]}")
#         print(f"Thermal shape: {thermal.shape}")
#         print(f"Thermal min/max: {thermal_img.min():.4f} / {thermal_img.max():.4f}")
#         print(f"Command Velocity min/max: {cmd_vel_msg.min():.4f} / {cmd_vel_msg.max():.4f}")
        
#     plt.tight_layout()
#     plt.show()

#     dataset = TronDataset(root=root, stats=stats_path, split='train')
#     val_dataset = TronDataset(root=root, stats=stats_path, split='validation')

#     print(f"Train dataset size: {len(dataset)}")
#     print(f"Validation dataset size: {len(val_dataset)}")
#     thermal, depth, elev, accel, gyro = dataset[0]
#     print("Sample from train dataset:")
#     print(f"Thermal shape: {thermal.shape}, Depth shape: {depth.shape}, Elev shape: {elev.shape}, Accel shape: {accel.shape}, Gyro shape: {gyro.shape}")

#     # Example of accessing data from validation dataset
#     thermal_val, depth_val, elev_val, accel_val, gyro_val = val_dataset[0]
#     print("Sample from validation dataset:")
#     print(f"Thermal shape: {thermal_val.shape}, Depth shape: {depth_val.shape}, Elev shape: {elev_val.shape}, Accel shape: {accel_val.shape}, Gyro shape: {gyro_val.shape}")

#     # Randomly select 4 indices
#     # random_indices = random.sample(range(len(dataset)), 4)
#     random_indices = [799, 49, 140, 70]

#     fig, axs = plt.subplots(4, 3, figsize=(18, 16)) 
#     fig.suptitle("Random Samples: Thermal and Depth Images")

#     for i, idx in enumerate(random_indices):
#         thermal, depth, elev, _, _ = dataset[idx]
        
#         thermal_img = thermal.squeeze().numpy()
#         depth_img = depth.squeeze().numpy()
#         elevation_img = elev.squeeze().numpy()

#         axs[i, 0].imshow(thermal_img, cmap='gray')
#         axs[i, 0].set_title(f"Thermal {idx}\nShape: {thermal.shape}")
#         axs[i, 0].axis('off')

#         axs[i, 1].imshow(depth_img, cmap='gray')
#         axs[i, 1].set_title(f"Depth {idx}\nShape: {depth.shape}")
#         axs[i, 1].axis('off')

#         axs[i, 2].imshow(elevation_img, cmap='terrain')
#         axs[i, 2].set_title(f"Elevation {idx}\nShape: {elev.shape}")
#         axs[i, 2].axis('off')

#         print(f"Sample {i+1}:")
#         print(f"Thermal file: {dataset.data['thermal_paths'][idx]}")
#         print(f"Depth file: {dataset.data['depth_paths'][idx]}")
#         print(f"Elevation file: {dataset.data['elevation_image_paths'][idx]}")
#         print(f"Thermal shape: {thermal.shape}, Depth shape: {depth.shape}, Elev shape: {elev.shape}")
#         print(f"Thermal min/max: {thermal_img.min():.4f} / {thermal_img.max():.4f}")
#         print(f"Depth min/max: {depth_img.min():.4f} / {depth_img.max():.4f}")
#         print(f"Elevation min/max: {elevation_img.min():.4f} / {elevation_img.max():.4f}")
#         print()

#     plt.tight_layout()
#     plt.show()


    
   
    