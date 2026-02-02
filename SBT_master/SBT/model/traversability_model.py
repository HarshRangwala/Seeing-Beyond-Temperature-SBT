import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        attention = self.activation(self.norm(self.conv(x)))
        return x * attention

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                             padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class TraversabilityDecoder(nn.Module):
    def __init__(self, latent_size=2048, num_layers=50):
        super(TraversabilityDecoder, self).__init__()
        if num_layers == 50:
            self.encoder_channels = {
                'layer4': 2048, 
                'layer3': 1024, 
                'layer2': 512, 
                'layer1': 256, 
                'conv1': 64
            }
        else:  # ResNet18
            self.encoder_channels = {
                'layer4': 512, 
                'layer3': 256, 
                'layer2': 128, 
                'layer1': 64, 
                'conv1': 64
            }
            
        # Initial processing of latent vector
        self.fc = nn.Linear(latent_size, self.encoder_channels['layer4'] * 8 * 8)
        self.unflatten = nn.Unflatten(1, (self.encoder_channels['layer4'], 8, 8))

        # Upsampling blocks with skip connections
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv1 = ConvBlock(self.encoder_channels['layer4'], self.encoder_channels['layer3'])
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv2 = ConvBlock(self.encoder_channels['layer3'] + self.encoder_channels['layer4'], 
                                self.encoder_channels['layer2'])
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv3 = ConvBlock(self.encoder_channels['layer2'] + self.encoder_channels['layer3'], 
                                self.encoder_channels['layer1'])
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv4 = ConvBlock(self.encoder_channels['layer1'] + self.encoder_channels['layer2'], 
                                64)
        
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv5 = ConvBlock(64 + self.encoder_channels['layer1'], 32)
        
        # Final convolution with skip connection
        self.conv_final = ConvBlock(32 + self.encoder_channels['conv1'], 16)
        
        # Output layer - sigmoid activation for binary classification
        self.final_block = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Sigmoid for binary traversability confidence
        )

        # Optional attention modules for feature fusion
        self.attn_layer4 = SpatialAttention(self.encoder_channels['layer4'])
        self.attn_layer3 = SpatialAttention(self.encoder_channels['layer3'])
        self.attn_layer2 = SpatialAttention(self.encoder_channels['layer2'])
        self.attn_layer1 = SpatialAttention(self.encoder_channels['layer1'])
        self.attn_conv1 = SpatialAttention(self.encoder_channels['conv1'])

    def forward(self, x, skip_features=None):
        # Process latent vector
        x = self.fc(x)
        x = self.unflatten(x)
        
        # Apply attention to features
        skip4 = self.attn_layer4(skip_features['layer4'])
        skip3 = self.attn_layer3(skip_features['layer3'])
        skip2 = self.attn_layer2(skip_features['layer2'])
        skip1 = self.attn_layer1(skip_features['layer1'])
        skip0 = self.attn_conv1(skip_features['conv1'])
        
        # Initial deconvolution
        x = self.deconv1(x)
        
        # Stage 1: 8x8 -> 16x16
        x = self.up1(x)
        skip4_resized = F.interpolate(skip4, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip4_resized], dim=1)
        x = self.deconv2(x)
        
        # Stage 2: 16x16 -> 32x32
        x = self.up2(x)
        skip3_resized = F.interpolate(skip3, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip3_resized], dim=1)
        x = self.deconv3(x)
        
        # Stage 3: 32x32 -> 64x64
        x = self.up3(x)
        skip2_resized = F.interpolate(skip2, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip2_resized], dim=1)
        x = self.deconv4(x)
        
        # Stage 4: 64x64 -> 128x128
        x = self.up4(x)
        skip1_resized = F.interpolate(skip1, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip1_resized], dim=1)
        x = self.deconv5(x)
        
        # Stage 5: 128x128 -> 256x256
        x = self.up5(x)
        skip0_resized = F.interpolate(skip0, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip0_resized], dim=1)
        x = self.conv_final(x)
        
        # Final prediction
        x = self.final_block(x)
        return x

class TraversabilityLoss(nn.Module):
    def __init__(self, weights={'bce': 1.0, 'dice': 1.0, 'boundary': 0.5}):
        super().__init__()
        self.weights = weights
        
    def forward(self, pred, target, isTraining=True):
        # Binary Cross Entropy Loss
        bce_loss = F.binary_cross_entropy(pred, target, reduction='mean')
        
        # Dice Loss
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1 - (2 * intersection / (union + 1e-8)).mean()
        
        # Boundary Loss - penalizes errors at the boundaries between traversable and non-traversable regions
        # target_boundaries = self._get_boundaries(target)
        # pred_boundaries = self._get_boundaries(pred)
        # boundary_loss = F.binary_cross_entropy(pred_boundaries, target_boundaries, reduction='mean')
        
        # Combined loss
        total_loss = (self.weights['bce'] * bce_loss + 
                      self.weights['dice'] * dice_loss )
                      # self.weights['boundary'] * boundary_loss)
        
        if not isTraining:
            metrics = self.compute_metrics(pred, target)
            return total_loss, metrics
        return total_loss
    
    # def _get_boundaries(self, mask, kernel_size=3):
    #     # Extract boundaries using erosion
    #     kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
    #     eroded = F.conv2d(mask, kernel, padding=kernel_size//2) > (kernel_size**2 - 0.5)
    #     boundaries = mask - eroded
    #     return boundaries
    
    def compute_metrics(self, pred, target):
        # Threshold predictions for binary metrics
        pred_binary = (pred > 0.5).float()
        
        # IoU
        intersection = (pred_binary * target).sum().item()
        union = (pred_binary + target).clamp(0, 1).sum().item()
        iou = intersection / (union + 1e-8)
        
        # F1 Score
        precision = intersection / (pred_binary.sum().item() + 1e-8)
        recall = intersection / (target.sum().item() + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'iou': iou,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

# if __name__ == '__main__':
#     print("--- Running Model Sanity Check ---")

#     # --- Configuration ---
#     BATCH_SIZE = 4
#     HEIGHT = 256
#     WIDTH = 256
#     LATENT_SIZE = 2048 # Make sure this matches the expected latent size from your encoder
#     NUM_LAYERS = 50 # Or 18, depending on the encoder architecture simulated
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {DEVICE}")

#     # --- Instantiate Model and Loss ---
#     print("\nInstantiating model and loss function...")
#     model = TraversabilityDecoder(latent_size=LATENT_SIZE, num_layers=NUM_LAYERS).to(DEVICE)
#     loss_fn = TraversabilityLoss().to(DEVICE) # Use default weights for testing
#     print("Model and Loss function instantiated.")

#     # --- Create Dummy Inputs ---
#     print("\nCreating dummy input data...")
#     # 1. Dummy Latent Vector (Simulating output from encoder's global average pooling/flatten)
#     dummy_latent_vector = torch.randn(BATCH_SIZE, LATENT_SIZE, device=DEVICE)
#     print(f"Dummy latent vector shape: {dummy_latent_vector.shape}")

#     # 2. Dummy Skip Features (Simulating intermediate features from an encoder like ResNet)
#     #    Shapes depend on the NUM_LAYERS choice and standard ResNet architectures
#     #    Example for ResNet50 (adjust channels/dims if using ResNet18 or different base)
#     if NUM_LAYERS == 50:
#         skip_channels = {'layer4': 2048, 'layer3': 1024, 'layer2': 512, 'layer1': 256, 'conv1': 64}
#     else: # Assuming ResNet18 otherwise
#         skip_channels = {'layer4': 512, 'layer3': 256, 'layer2': 128, 'layer1': 64, 'conv1': 64}

#     # Assuming input image size is H=256, W=256 to the original encoder
#     # These are the typical output sizes *before* the decoder might resize them further
#     dummy_skip_features = {
#         'layer4': torch.randn(BATCH_SIZE, skip_channels['layer4'], HEIGHT // 32, WIDTH // 32, device=DEVICE), # e.g., 8x8
#         'layer3': torch.randn(BATCH_SIZE, skip_channels['layer3'], HEIGHT // 16, WIDTH // 16, device=DEVICE), # e.g., 16x16
#         'layer2': torch.randn(BATCH_SIZE, skip_channels['layer2'], HEIGHT // 8,  WIDTH // 8,  device=DEVICE), # e.g., 32x32
#         'layer1': torch.randn(BATCH_SIZE, skip_channels['layer1'], HEIGHT // 4,  WIDTH // 4,  device=DEVICE), # e.g., 64x64
#         'conv1':  torch.randn(BATCH_SIZE, skip_channels['conv1'],  HEIGHT // 2,  WIDTH // 2,  device=DEVICE)  # e.g., 128x128 (after initial conv/pool)
#     }
#     print("Dummy skip features shapes:")
#     for key, tensor in dummy_skip_features.items():
#         print(f"  '{key}': {tensor.shape}")

#     # 3. Dummy Target Mask (Simulating ground truth from dataloader)
#     #    Should have shape (batch_size, 1, H, W) with values between 0 and 1
#     dummy_target_mask = torch.rand(BATCH_SIZE, 1, HEIGHT, WIDTH, device=DEVICE)
#     # Or, for a binary mask test:
#     # dummy_target_mask = torch.randint(0, 2, (BATCH_SIZE, 1, HEIGHT, WIDTH), dtype=torch.float32, device=DEVICE)
#     print(f"Dummy target mask shape: {dummy_target_mask.shape}")

#     # --- Perform Forward Pass ---
#     print("\nPerforming forward pass...")
#     model.eval() # Set model to evaluation mode for consistency (affects BatchNorm, Dropout if any)
#     with torch.no_grad(): # No need to calculate gradients for this test
#         predicted_mask = model(dummy_latent_vector, dummy_skip_features)
#     print(f"Predicted mask shape: {predicted_mask.shape}")

#     # --- Check Output Range (Sigmoid) ---
#     print(f"Predicted mask value range: min={predicted_mask.min().item():.4f}, max={predicted_mask.max().item():.4f}")
#     if not (predicted_mask.min() >= 0.0 and predicted_mask.max() <= 1.0):
#          print("Warning: Predicted mask values are outside the expected [0, 1] range after Sigmoid.")
#     else:
#          print("Predicted mask values are within the expected [0, 1] range.")


#     # --- Calculate Loss ---
#     print("\nCalculating loss...")
#     # Test with isTraining=True (just returns combined loss)
#     loss_train_mode = loss_fn(predicted_mask, dummy_target_mask, isTraining=True)
#     print(f"Calculated Loss (isTraining=True): {loss_train_mode.item():.4f}")
#     print(f"Loss tensor type: {loss_train_mode.dtype}")

#     # Test with isTraining=False (returns loss and metrics dict)
#     loss_val_mode, metrics = loss_fn(predicted_mask, dummy_target_mask, isTraining=False)
#     print(f"Calculated Loss (isTraining=False): {loss_val_mode.item():.4f}")
#     print("Calculated Metrics (isTraining=False):")
#     for key, value in metrics.items():
#         print(f"  {key}: {value:.4f}")


#     # --- Basic Assertions (Optional but Recommended) ---
#     try:
#         assert predicted_mask.shape == dummy_target_mask.shape, \
#             f"Shape mismatch: Predicted {predicted_mask.shape} vs Target {dummy_target_mask.shape}"
#         assert predicted_mask.dtype == torch.float32, f"Predicted mask dtype is {predicted_mask.dtype}, expected float32"
#         assert loss_train_mode.dtype == torch.float32, f"Loss dtype is {loss_train_mode.dtype}, expected float32"
#         print("\nBasic assertions passed.")
#     except AssertionError as e:
#         print(f"\nAssertion Failed: {e}")

#     print("\n--- Sanity Check Complete ---")