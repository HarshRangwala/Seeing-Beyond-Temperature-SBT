import torch
from torch import nn
import torch.nn.functional as F

class VisionEncoder(nn.Module):
    def __init__(self, latent_size=512, l2_normalize=True, p=0.3):
        super().__init__()

        # Block 1: Initial Feature Extraction
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Skip Connection Block 1: Residual Learning
        self.skipblock1 = nn.Sequential(  # Changed name to skipblock1
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(8),
        )

        # Block 2: Deeper Feature Extraction
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Skip Connection Block 2
        self.skipblock2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(16),
        )

        # Block 3: Further Feature Abstraction
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # Skip Connection Block 3
        self.skipblock3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(32),
        )

        # Block 4: Final Feature Compression
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(64),
        )

        # Skip Connection Block 4
        self.skipblock4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(64),
        )

        # Final Linear Projection
        self.patch_encoder = nn.Linear(6400, latent_size)
        self.l2_normalize = l2_normalize

    def forward(self, x):
        # Progressive feature extraction with skip connections
        f1 = self.block1(x)
        f1 = self.skipblock1(f1) + f1  # Skip connection 1, store for later

        f2 = self.block2(f1)
        f2 = self.skipblock2(f2) + f2  # Skip connection 2, store for later

        f3 = self.block3(f2)
        f3 = self.skipblock3(f3) + f3  # Skip connection 3, store for later

        f4 = self.block4(f3)
        f4 = self.skipblock4(f4) + f4  # Skip connection 4, store for later

        # Flatten and encode to latent space
        x = f4.reshape(f4.size(0), -1)  # Flatten
        out = self.patch_encoder(x)

        # Optional L2 normalization
        if self.l2_normalize:
            out = F.normalize(out, dim=-1)

        return out, [f1, f2, f3, f4]  # Return encoded output AND skip connection features

class DepthDecoder(nn.Module):
    def __init__(self, latent_size=512, use_skip_connections=True):
        super().__init__()
        self.use_skip_connections = use_skip_connections

        self.linear = nn.Sequential(
            nn.Linear(latent_size, 256 * 16 * 16),
            nn.Mish()
        )

        # Decoder block 1 (with optional skip connection)
        self.deconv1 = nn.ConvTranspose2d(256 + (64 if use_skip_connections else 0), 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # Decoder block 2 (with optional skip connection)
        self.deconv2 = nn.ConvTranspose2d(128 + (32 if use_skip_connections else 0), 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Decoder block 3 (with optional skip connection)
        self.deconv3 = nn.ConvTranspose2d(64 + (16 if use_skip_connections else 0), 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Decoder block 4 (with optional skip connection)
        self.deconv4 = nn.ConvTranspose2d(32 + (8 if use_skip_connections else 0), 16, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, x, skip_features=None):
        x = self.linear(x)
        x = x.view(-1, 256, 16, 16)

        # --- Decoder Block 1 ---
        if self.use_skip_connections and skip_features is not None:
            # Resize skip_features[3] to match x's spatial dimensions (16x16)
            skip = F.interpolate(skip_features[3], size=(16, 16), mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        x = F.relu(self.bn1(self.deconv1(x)))

        # --- Decoder Block 2 ---
        if self.use_skip_connections and skip_features is not None:
            skip = F.interpolate(skip_features[2], size=(32, 32), mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn2(self.deconv2(x)))


        # --- Decoder Block 3 ---
        if self.use_skip_connections and skip_features is not None:
            skip = F.interpolate(skip_features[1], size=(64, 64), mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn3(self.deconv3(x)))

        # --- Decoder Block 4 ---
        if self.use_skip_connections and skip_features is not None:
            skip = F.interpolate(skip_features[0], size=(128, 128), mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn4(self.deconv4(x)))

        x = self.final_conv(x)  # No activation here, as it's a regression task
        return x


class IMUEncoder(nn.Module):
    def __init__(self, latent_size=64, p=0.3, l2_normalize=True):
        super(IMUEncoder, self).__init__()

        # Acceleration Encoder
        self.accel_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(201 * 3, 128),
            nn.Mish(),
            nn.Dropout(p),
            nn.Linear(128, latent_size // 2),
        )

        # Gyroscope Encoder
        self.gyro_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(201 * 3, 128),
            nn.Mish(),
            nn.Dropout(p),
            nn.Linear(128, latent_size // 2),
        )

        # Fully Connected Layer to Combine Encoded Features
        self.fc = nn.Sequential(
            nn.Linear(2 * (latent_size // 2), latent_size),
            nn.Mish(),
            nn.Linear(latent_size, latent_size),
        )

        self.l2_normalize = l2_normalize

    def forward(self, accel, gyro):
        accel_encoded = self.accel_encoder(accel)
        gyro_encoded = self.gyro_encoder(gyro)
        combined = torch.cat([accel_encoded, gyro_encoded], dim=1)
        combined = self.fc(combined)
        if self.l2_normalize:
            combined = F.normalize(combined, dim=-1)
        return combined

class Projector(nn.Module):
    """Projector for Barlow Twins"""
    def __init__(self, in_dim, hidden_dim, out_dim):  # Corrected argument names
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=False),  # output layer
            nn.BatchNorm1d(out_dim) # ADDED
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # No ReLU after the final layer
        return x
    
class TronModel(nn.Module):
    def __init__(self, vision_encoder, imu_encoder, projector):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.imu_encoder = imu_encoder
        self.projector = projector

    def forward(self, thermal, depth, accel, gyro):
        # Encode thermal image and get skip features
        v_encoded_thermal, skip_features_thermal = self.vision_encoder(thermal)
        v_encoded_thermal = F.normalize(v_encoded_thermal, dim=-1)

        # Encode depth image and get skip features
        v_encoded_depth, skip_features_depth = self.vision_encoder(depth)
        v_encoded_depth = F.normalize(v_encoded_depth, dim=-1)

        # Encode IMU data
        i_encoded = self.imu_encoder(accel, gyro)

        # Project the encodings
        zv1 = self.projector(v_encoded_thermal)
        zv2 = self.projector(v_encoded_depth)
        zi = self.projector(i_encoded)

        return zv1, zv2, zi, v_encoded_thermal, v_encoded_depth, i_encoded, skip_features_thermal, skip_features_depth
    
    def barlow_loss(self, z1, z2):
        B = z1.shape[0]
        z1 = (z1 - z1.mean(dim=0)) / z1.std(dim=0)
        z2 = (z2 - z2.mean(dim=0)) / z2.std(dim=0)
        c = z1.T @ z2
        c.div_(B)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + 0.0051 * off_diag
        return loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m, "Input matrix must be square"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, isTraining=True):
        # Denormalizing to max depth for loss calculation
        pred_denorm = pred * 20
        target_denorm = target * 20

        # --- Create the mask ---
        valid_mask = (target_denorm > 1e-3) & (target_denorm < 20.0)
        # IMPORTANT: Ensure the mask has the same shape as pred and target
        valid_mask = valid_mask.float()  # Convert to float for multiplication

        # --- Apply the mask using multiplication ---
        masked_pred = pred_denorm * valid_mask
        masked_target = target_denorm * valid_mask

        # Calculate losses using the *masked* tensors
        smoothness_loss = self.edge_aware_smoothness(masked_pred, masked_target, valid_mask) # Pass the mask
        ssim_loss = self.ssim_masked(masked_pred, masked_target, valid_mask)
        grad_loss = self.gradient_matching_loss(masked_pred, masked_target, valid_mask)  # Pass the mask
        l1_loss = self.l1_masked(masked_pred, masked_target, valid_mask)

        total_loss = l1_loss + 0.4 * smoothness_loss + 0.85 * ssim_loss

        if not isTraining:
            # For metrics, use boolean indexing to get *only* the valid pixels
            metrics = self.compute_depth_metrics(pred_denorm[valid_mask.bool()], target_denorm[valid_mask.bool()])
            return total_loss, metrics
        return total_loss
    def l1_masked(self, pred, target, mask):
        """Calculates L1 loss, handling masked regions."""
        loss = torch.abs(pred - target) * mask  # Apply mask element-wise
        return loss.sum() / (mask.sum() + 1e-8)  # Normalize by the number of valid pixels


    def edge_aware_smoothness(self, pred, target, mask):
      """Calculates edge-aware smoothness loss, handling masked regions."""
      pred_gradients_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
      pred_gradients_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

      target_gradients_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
      target_gradients_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

      weights_x = torch.exp(-target_gradients_x)
      weights_y = torch.exp(-target_gradients_y)

      # Apply the mask to the gradients.  Crucially, we need to adjust the mask
      # to match the size of the gradients (which are one element smaller in
      # the respective dimensions).
      mask_x = mask[:, :, :, :-1]  # Mask for x-gradients
      mask_y = mask[:, :, :-1, :]  # Mask for y-gradients


      smoothness_x = (pred_gradients_x * weights_x * mask_x).sum() / (mask_x.sum() + 1e-8)
      smoothness_y = (pred_gradients_y * weights_y * mask_y).sum() / (mask_y.sum() + 1e-8)
      return smoothness_x + smoothness_y

    def gradient_matching_loss(self, pred, target, mask):
        """Calculates gradient matching loss, handling masked regions."""

        pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

        # Apply the mask to the gradients, adjusting for size differences.
        mask_x = mask[:, :, :, :-1]
        mask_y = mask[:, :, :-1, :]

        loss_x = torch.abs(pred_grad_x - target_grad_x) * mask_x
        loss_y = torch.abs(pred_grad_y - target_grad_y) * mask_y


        return loss_x.sum() / (mask_x.sum() + 1e-8) + loss_y.sum() / (mask_y.sum() + 1e-8)
    def ssim_masked(self, pred, target, mask):
        """Calculates SSIM loss, handling masked regions."""
        # Ensure mask is float and has the correct number of dimensions
        mask = mask.float()

        # Apply the mask before calculating SSIM
        pred_masked = pred * mask
        target_masked = target * mask

        # Use ms_ssim (multi-scale SSIM) for better robustness. You could also use ssim.
        # Ensure data_range is appropriate for your denormalized data (0-20).
        ssim_val = ms_ssim(pred_masked, target_masked, data_range=20.0, size_average=False)

        # Weight the SSIM loss by the mask.  This is equivalent to averaging only over
        # the valid pixels.
        masked_ssim = (1 - ssim_val) * mask

        # Average the masked SSIM loss, adding a small constant to prevent division by zero.
        return masked_ssim.sum() / (mask.sum() + 1e-8)

    def compute_depth_metrics(self, pred, target):
        """Compute additional depth metrics for evaluation"""
        abs_rel = torch.mean(torch.abs(pred - target) / (target + 1e-10))
        rmse = torch.sqrt(torch.mean((pred - target) ** 2))
        thresh = torch.max((target / (pred + 1e-10)), (pred / (target + 1e-10)))
        delta1 = (thresh < 1.25).float().mean()
        return {
            "abs_rel": abs_rel.item(),
            "rmse": rmse.item(),
            "delta1": delta1.item(),
        }