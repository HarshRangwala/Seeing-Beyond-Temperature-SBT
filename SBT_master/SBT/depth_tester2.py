import gc
from pathlib import Path
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm

# Attempt to add project root to sys.path for module discovery
# This assumes the script is run from the project root or models/utils are otherwise findable.
try:
    project_root = Path(".").resolve()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
except Exception as e:
    print(f"Warning: Could not automatically add project root to sys.path: {e}")

from model.m2p2_model import VisionEncoder, DepthDecoder
from model.thMonoDepth_model_load import load_ThermalMonoDepth_model
from utils.io import load_checkpoint
from utils.helpers import get_conf

# --- Hardcoded Configuration ---
# !!! USER: PLEASE UPDATE THESE PATHS AND PARAMETERS !!!
CONFIG_PATH = "./conf/config_depth.yaml"  # Path to your .yaml configuration file
PRETRAINED_ENCODER_CHECKPOINT_PATH = "/mnt/sbackup/Server_3/harshr/home/NV_cahsor/CAHSOR-master/TRON/checkpoint/tron/ssl-ptr_aug-thermal_lidar-2048-04-13-14-42/ssl-ptr_aug-thermal_lidar-2048-04-15-14-31/ssl-ptr_aug-thermal_lidar-2048-04-15-14-31_500.pth"
DEPTH_DECODER_CHECKPOINT_PATH = "/mnt/sbackup/Server_3/harshr/home/NV_cahsor/CAHSOR-master/TRON/checkpoint/tron/decoder_depth_thermal_lidar_ckpts/decoder_depth_2-2048-04-16-03-43/depth_decoder_epoch_100.pth"
THERMAL_MONO_DEPTH_WEIGHTS_PATH = "/mnt/sbackup/Server_3/harshr/home/NV_cahsor/CAHSOR-master/TRON/checkpoint/thMonoDepth/dispnet_disp_model_best.pth.tar"

INPUT_IMAGE_DIR = "/mnt/sbackup/Server_3/harshr/depth_test/" # Directory containing thermal and ground truth depth images
OUTPUT_DIR = "/mnt/sbackup/Server_3/harshr/depth_test/tester2/WC10_55" # Directory to save output images

# Preprocessing parameters (should match your training setup from TronDataset)
RESIZE_DIM_LOAD = (256, 256)  # Initial resize dimensions for thermal images
CROP_TOP = 40                 # Top crop pixel
CROP_BOTTOM = 225             # Bottom crop pixel (exclusive, e.g., up to 224)
MODEL_INPUT_DIM = (256, 256)  # Final (Height, Width) for your custom model's input
# --- End of Hardcoded Configuration ---

def preprocess_thermal_custom(img_tensor: torch.Tensor) -> torch.Tensor:
    """Custom thermal image preprocessing, mirrors m2p2_dataloader.preprocess_thermal."""
    img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-6)
    img_tensor = torch.clip(img_tensor, min=-3, max=2)
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-6)
    return img_tensor

def main():
    cfg = get_conf(CONFIG_PATH)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR, "thermal_input").mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR, "ground_truth_depth").mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR, "our_model_prediction").mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR, "thermal_mono_depth_prediction").mkdir(parents=True, exist_ok=True)

    # --- Load Custom Model (Vision Encoder + Depth Decoder) ---
    vision_encoder = VisionEncoder(
        latent_size=cfg.model.rep_size, 
        num_layers=cfg.model.num_layers_enc
    ).to(device)
    
    print(f"Loading pretrained vision encoder from: {PRETRAINED_ENCODER_CHECKPOINT_PATH}")
    encoder_checkpoint = load_checkpoint(PRETRAINED_ENCODER_CHECKPOINT_PATH, device)
    
    if "vision_encoder" in encoder_checkpoint:
        vision_encoder.load_state_dict(encoder_checkpoint["vision_encoder"])
        print("Loaded vision encoder weights from 'vision_encoder' key.")
    elif "model" in encoder_checkpoint: # Checkpoint from TronModel
        full_model_state_dict = encoder_checkpoint["model"]
        vision_encoder_state_dict = {
            k.replace("vision_encoder.", ""): v 
            for k, v in full_model_state_dict.items() 
            if k.startswith("vision_encoder.")
        }
        if not vision_encoder_state_dict:
            raise ValueError("Encoder checkpoint has 'model' key, but no 'vision_encoder.' prefixed keys found within it.")
        vision_encoder.load_state_dict(vision_encoder_state_dict)
        print("Loaded vision encoder weights from 'model' key (filtered for 'vision_encoder.*').")
    else: # Fallback: checkpoint is the state_dict of vision_encoder itself
        vision_encoder.load_state_dict(encoder_checkpoint)
        print("Loaded vision encoder weights assuming checkpoint file is the raw state_dict.")
    vision_encoder.eval()

    depth_decoder = DepthDecoder(
        latent_size=cfg.model.rep_size, 
        num_layers=cfg.model.num_layers_enc # As per your trainer
    ).to(device)
    
    print(f"Loading depth decoder from: {DEPTH_DECODER_CHECKPOINT_PATH}")
    decoder_checkpoint = load_checkpoint(DEPTH_DECODER_CHECKPOINT_PATH, device)
    depth_decoder.load_state_dict(decoder_checkpoint["depth_decoder"])
    depth_decoder.eval()

    # --- Load ThermalMonoDepth Model ---
    tmd_resnet_layers = cfg.thMonoLoad.get("resnet_layers", 18)
    tmd_scene_type = cfg.thMonoLoad.get("scene_type", "outdoor")
    tmd_max_depth = cfg.thMonoLoad.get("max_depth", 30)
    
    print(f"Loading ThermalMonoDepth model from: {THERMAL_MONO_DEPTH_WEIGHTS_PATH}")
    thermal_mono_depth_model = load_ThermalMonoDepth_model(
        weights_path=THERMAL_MONO_DEPTH_WEIGHTS_PATH,
        resnet_layers=tmd_resnet_layers,
        scene_type=tmd_scene_type,
        max_depth=tmd_max_depth
    ) # This function also sets the model to eval mode.

    # --- Image Processing Loop ---
    input_base_dir = Path(INPUT_IMAGE_DIR)
    
    # Find all subdirectories that look like thermal data folders
    processed_thermal_folders = [d for d in input_base_dir.iterdir() if d.is_dir() and d.name.startswith("thermal_") and d.name.endswith("_processed")]

    if not processed_thermal_folders:
        print(f"No 'thermal_*_processed' subfolders found in '{INPUT_IMAGE_DIR}'")
        return

    for thermal_folder_path in tqdm(processed_thermal_folders, desc="Processing data chunks"):
        # Derive corresponding depth folder name and path
        depth_folder_name = thermal_folder_path.name.replace("thermal_", "depth_", 1) # Replace only the first occurrence
        depth_folder_path = thermal_folder_path.parent / depth_folder_name

        if not depth_folder_path.is_dir():
            print(f"Warning: Corresponding depth folder '{depth_folder_path.name}' not found for '{thermal_folder_path.name}'. Skipping this chunk.")
            continue

        print(f"\nProcessing images in chunk: {thermal_folder_path.name} / {depth_folder_path.name}")

        # Create subdirectories in output for each processed chunk to keep results organized
        chunk_name_for_output = thermal_folder_path.name # e.g., "thermal_BL_2024-09-04_19-10-17_chunk0001_processed"
        
        current_output_thermal_input_dir = Path(OUTPUT_DIR, "thermal_input", chunk_name_for_output)
        current_output_gt_depth_dir = Path(OUTPUT_DIR, "ground_truth_depth", chunk_name_for_output)
        current_output_our_model_dir = Path(OUTPUT_DIR, "our_model_prediction", chunk_name_for_output)
        current_output_tmd_dir = Path(OUTPUT_DIR, "thermal_mono_depth_prediction", chunk_name_for_output)

        current_output_thermal_input_dir.mkdir(parents=True, exist_ok=True)
        current_output_gt_depth_dir.mkdir(parents=True, exist_ok=True)
        current_output_our_model_dir.mkdir(parents=True, exist_ok=True)
        current_output_tmd_dir.mkdir(parents=True, exist_ok=True)

        # Get all .png images from the current thermal folder
        thermal_image_files = sorted(list(thermal_folder_path.glob('*.png')))

        if not thermal_image_files:
            print(f"No .png images found in '{thermal_folder_path}'. Skipping this chunk.")
            continue

        for thermal_path in tqdm(thermal_image_files, desc=f"Images in {thermal_folder_path.name}", leave=False):
            image_filename = thermal_path.name # e.g., "00001.png"
            depth_gt_path = depth_folder_path / image_filename # Assumes same filename in depth folder
            
            if not depth_gt_path.exists():
                print(f"Warning: Ground truth depth for '{thermal_path.name}' (expected at '{depth_gt_path}') not found. Skipping this image.")
                continue
            
            # Load thermal image (as grayscale)
            thermal_img_orig = cv2.imread(str(thermal_path), cv2.IMREAD_UNCHANGED)
            if thermal_img_orig is None: print(f"Warning: Could not read thermal image '{thermal_path}'. Skipping."); continue
            if len(thermal_img_orig.shape) == 3 and thermal_img_orig.shape[2] >= 3:
                thermal_img_orig = cv2.cvtColor(thermal_img_orig, cv2.COLOR_BGR2GRAY)
            
            # Load ground truth depth image (as grayscale)
            depth_gt_orig = cv2.imread(str(depth_gt_path), cv2.IMREAD_UNCHANGED)
            if depth_gt_orig is None: print(f"Warning: Could not read depth_gt image '{depth_gt_path}'. Skipping."); continue
            if len(depth_gt_orig.shape) == 3 and depth_gt_orig.shape[2] >= 3:
                 depth_gt_orig = cv2.cvtColor(depth_gt_orig, cv2.COLOR_BGR2GRAY)

            # --- Preprocessing for Custom Model (matches TronDataset steps) ---
            thermal_resized_load = cv2.resize(thermal_img_orig, RESIZE_DIM_LOAD, interpolation=cv2.INTER_AREA)
            thermal_cropped = thermal_resized_load[CROP_TOP:CROP_BOTTOM, :]
            thermal_for_custom_model_np = cv2.resize(thermal_cropped, (MODEL_INPUT_DIM[1], MODEL_INPUT_DIM[0]), interpolation=cv2.INTER_LINEAR)
            
            thermal_tensor = torch.tensor(thermal_for_custom_model_np, dtype=torch.float32)
            thermal_tensor_processed = preprocess_thermal_custom(thermal_tensor)
            
            thermal_input_to_encoder = thermal_tensor_processed.unsqueeze(0).unsqueeze(0).to(device)

            # --- Inference with Custom Model ---
            with torch.no_grad():
                v_encoded_thermal, thermal_features = vision_encoder(thermal_input_to_encoder, return_features=True)
                depth_pred_custom_tensor = depth_decoder(v_encoded_thermal, thermal_features)
            depth_pred_custom_np = depth_pred_custom_tensor.squeeze().cpu().numpy()

            # --- Input for ThermalMonoDepth Model ---
            thermal_for_tmd_np = thermal_tensor_processed.cpu().numpy()
            
            # --- Inference with ThermalMonoDepth ---
            depth_pred_tmd_np = thermal_mono_depth_model.infer_image(thermal_for_tmd_np)

            # --- Saving Outputs ---
            output_filename_base = thermal_path.stem # This will be the number, e.g., "00001"
            
            plt.imsave(current_output_thermal_input_dir / f"{output_filename_base}_input.png", thermal_img_orig, cmap='gray')
            plt.imsave(current_output_gt_depth_dir / f"{output_filename_base}_gt.png", depth_gt_orig, cmap='gray')
            plt.imsave(current_output_our_model_dir / f"{output_filename_base}_pred_ours.png", depth_pred_custom_np, cmap='gray', vmin=0.0, vmax=1.0)
            plt.imsave(current_output_tmd_dir / f"{output_filename_base}_pred_tmd.png", depth_pred_tmd_np, cmap='gray')

    print(f"Inference complete. Results saved to '{OUTPUT_DIR}'")
    gc.collect()

if __name__ == "__main__":
    main()