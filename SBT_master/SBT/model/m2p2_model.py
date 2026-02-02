import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights

class ResnetEncoder(nn.Module):
    def __init__(self, num_layers, pretrained, num_channel=1):
        super(ResnetEncoder, self).__init__()
        if num_layers not in [18, 50]:
            raise ValueError(f"{num_layers} is not a valid number of ResNet layers")

        if pretrained:
            if num_layers == 18:
                resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            elif num_layers == 50:
                resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                raise ValueError("Pretrained weights only implemented for ResNet-18 and ResNet-50")
        else:
            if num_layers == 18:
                resnet = models.resnet18(pretrained=False)
            elif num_layers == 50:
                resnet = models.resnet50(pretrained=False)
            else:
                raise ValueError("Must specify a valid ResNet variant")

        if num_channel != 3:  # Efficient handling of different input channels
            resnet.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained: # Properly handle weight initialization
                if num_channel == 1:
                    new_conv1_weight = resnet.conv1.weight.mean(dim=1, keepdim=True)
                else: # num_channel > 3
                    new_conv1_weight = resnet.conv1.weight.mean(dim=1, keepdim=True)
                    new_conv1_weight = new_conv1_weight.repeat(1, num_channel, 1, 1) / num_channel
                resnet.conv1.weight = nn.Parameter(new_conv1_weight)
            # else:  # No need for manual weight initialization if not pretrained, PyTorch does it.


        # Store individual components for skip connections
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Identity()  # Important:  Replace the fully connected layer


    def forward(self, x, return_features=False):
        skip_features = {}

        x = self.conv1(x)
        skip_features['conv1'] = x

        x = self.bn1(x)
        x = self.relu(x)
        skip_features['layer0'] = x

        x = self.maxpool(x)
        x = self.layer1(x)
        skip_features['layer1'] = x

        x = self.layer2(x)
        skip_features['layer2'] = x

        x = self.layer3(x)
        skip_features['layer3'] = x

        x = self.layer4(x)
        skip_features['layer4'] = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)  # Apply identity

        if return_features:
            return x, skip_features
        return x

class Projector(nn.Module):
    """Projector for Barlow Twins"""
    def __init__(self, in_dim, hidden_dim, out_dim):  # Explicitly define all dimensions
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class IMUEncoder(nn.Module):
    def __init__(self, latent_size=2048, p=0.1):
        super(IMUEncoder, self).__init__()
        # Input is flattened 6-axis IMU (Accel+Gyro)
        # Dataset inspection shows shape (1200,), which is likely 400 samples * 3 axes (2 sec @ 200Hz)
        
        self.accel_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1200, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(512, latent_size // 2),
            nn.BatchNorm1d(latent_size // 2),
            nn.ReLU(inplace=True)
        )

        self.gyro_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1200, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(512, latent_size // 2),
            nn.BatchNorm1d(latent_size // 2),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, accel, gyro):
        a_emb = self.accel_encoder(accel)
        g_emb = self.gyro_encoder(gyro)
        x = torch.cat([a_emb, g_emb], dim=1)
        return self.fc(x)

class VisionEncoder(nn.Module):
    def __init__(self, latent_size=2048, num_layers=50, pretrained=False, num_channel=1):
        super().__init__()
        self.vision_encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained, num_channel=num_channel)        
        # self.projection = Projector(resnet_output_size, 4*resnet_output_size, latent_size)

    def forward(self, input_image, return_features=False):
        if return_features:
            rep, skip_features = self.vision_encoder(input_image, return_features=True)
            # out = self.projection(features)
            return rep, skip_features
        else:
            rep = self.vision_encoder(input_image)
            # out = self.projection(features)
            return rep

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                             padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DepthDecoder(nn.Module):
    def __init__(self, latent_size=2048, num_layers = 50):
        super(DepthDecoder, self).__init__()
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
            
        self.fc = nn.Linear(latent_size, self.encoder_channels['layer4'] * 8 * 8)
        self.unflatten = nn.Unflatten(1, (self.encoder_channels['layer4'], 8, 8))

        # Upsampling and convolution blocks
        # First block processes the latent vector without skip connection
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv1 = ConvBlock(self.encoder_channels['layer4'], self.encoder_channels['layer3'])
        
        
        # Subsequent blocks incorporate skip connections
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
        
        # Final convolution after last skip connection
        self.conv_final = ConvBlock(32 + self.encoder_channels['conv1'], 16)
        self.final_block = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
            # nn.ReLU()
        )

    def forward(self, x, skip_features=None):
        # Process latent vector
        x = self.fc(x)
        x = self.unflatten(x) # [B, encoder_channels['layer4'], 8, 8]
        
        # Initial deconvolution without skip connection
        x = self.deconv1(x)    # [B, encoder_channels['layer3'], 8, 8]
        
        # Stage 1: 8x8 -> 16x16
        x = self.up1(x)    # [B, encoder_channels['layer3'], 16, 16]
        skip4 = F.interpolate(skip_features['layer4'], scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip4], dim=1)  # [B, layer3+layer4, 16, 16]
        x = self.deconv2(x)  # [B, layer2, 16, 16]
        
        # Stage 2: 16x16 -> 32x32
        x = self.up2(x)    # [B, layer2, 32, 32]
        skip3 = F.interpolate(skip_features['layer3'], scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip3], dim=1)  # [B, layer2+layer3, 32, 32]
        x = self.deconv3(x)  # [B, layer1, 32, 32]
        
        # Stage 3: 32x32 -> 64x64
        x = self.up3(x)    # [B, layer1, 64, 64]
        skip2 = F.interpolate(skip_features['layer2'], scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip2], dim=1)  # [B, layer1+layer2, 64, 64]
        x = self.deconv4(x)  # [B, 64, 64, 64]
        
        # Stage 4: 64x64 -> 128x128
        x = self.up4(x)    # [B, 64, 128, 128]
        skip1 = F.interpolate(skip_features['layer1'], scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip1], dim=1)  # [B, 64+layer1, 128, 128]
        x = self.deconv5(x)  # [B, 32, 128, 128]
        
        # Stage 5: 128x128 -> 256x256
        x = self.up5(x)    # [B, 32, 256, 256]
        skip0 = F.interpolate(skip_features['conv1'], scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip0], dim=1)  # [B, 32+conv1, 256, 256]
        x = self.conv_final(x)  # [B, 16, 256, 256]
        
        # Final prediction
        x = self.final_block(x)  # [B, 1, 256, 256]
        return x

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM()
    
    def forward(self, pred, target, isTraining = True):
        l1_loss = F.l1_loss(pred, target, reduction='mean')
        smoothness_loss = self.edge_aware_smoothness(pred, target)
        #ssim_loss = 1 - ssim(pred, target, data_range = 1.0)
        ssim_loss = torch.mean(self.ssim(pred, target))
        grad_loss = self.gradient_matching_loss(pred, target)
        total_loss = 0.5*l1_loss + 0.85*ssim_loss + 0.3*smoothness_loss # + 0.6*grad_loss + 0.3*smoothness_loss

        if not isTraining:
            metrics = self.compute_depth_metrics(pred, target)
            return total_loss, metrics
        return total_loss
    
    def edge_aware_smoothness(self, pred, target):
        
        pred_gradients_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_gradients_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        target_gradients_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_gradients_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        weights_x = torch.exp(-target_gradients_x)
        weights_y = torch.exp(-target_gradients_y)
        
        smoothness_x = pred_gradients_x * weights_x
        smoothness_y = pred_gradients_y * weights_y
        
        return smoothness_x.mean() + smoothness_y.mean()

    def gradient_matching_loss(self, pred, target):
        pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)

    def compute_depth_metrics(self, pred, target):
        """Compute additional depth metrics for evaluation"""
        # valid_mask = (target > 0).float()
        
        # # Mask out invalid pixels
        # pred = pred * valid_mask
        # target = target * valid_mask
        gt_depth_denorm = target * 30.0
        recon_depth_denorm = pred * 30.0
        min_valid_depth = 0.1
        max_valid_depth = 30.0
        valid_mask = (gt_depth_denorm >= min_valid_depth) & (gt_depth_denorm <= max_valid_depth)
        pred = recon_depth_denorm[valid_mask]
        target = gt_depth_denorm[valid_mask]
        # Absolute Relative Error
        abs_rel = torch.mean(torch.abs(pred - target) / (target + 1e-10))
        
        # RMSE
        rmse = torch.sqrt(torch.mean((pred - target) ** 2))
        
        # δ1 accuracy (threshold = 1.25)
        thresh = torch.max((target / (pred + 1e-10)), (pred / (target + 1e-10)))
        delta1 = (thresh < 1.25).float().mean()
        delta2 = (thresh < 1.25**2).float().mean()  # Delta2
        delta3 = (thresh < 1.25**3).float().mean()  # Delta3

        return {
            'abs_rel': abs_rel.item(),
            'rmse': rmse.item(),
            'delta1': delta1.item(),
            'delta2': delta2.item(),  # Add delta2
            'delta3': delta3.item()   # Add delta3
        }


class TronModel(nn.Module):
    def __init__(self, vision_encoder, imu_encoder, projector, latent_size=2048):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.imu_encoder = imu_encoder
        self.projector = projector  # Shared projector
        self.vision_norm = nn.LayerNorm(latent_size) 
        self.imu_norm = nn.LayerNorm(latent_size)


    def forward(self, thermal, depth, accel=None, gyro=None):
        # Vision Encoder
        v_encoded_patch1, patch1_feats = self.vision_encoder(thermal, return_features=True)
        v_encoded_patch1 = self.vision_norm(v_encoded_patch1) # Normalize

        v_encoded_patch2, patch2_feats = self.vision_encoder(depth, return_features=True)
        v_encoded_patch2 = self.vision_norm(v_encoded_patch2) #Normalize

        # Projector (Shared)
        zv1 = self.projector(v_encoded_patch1)
        zv2 = self.projector(v_encoded_patch2)

        zi = None
        i_encoded = None
        if accel is not None and gyro is not None:
             i_encoded = self.imu_encoder(accel, gyro)
             i_encoded = self.imu_norm(i_encoded)
             zi = self.projector(i_encoded)

        return zv1, zv2, zi, v_encoded_patch1, v_encoded_patch2, i_encoded, patch1_feats, patch2_feats

    def barlow_loss(self, z1, z2):
        # Normalize the projections along the batch dimension
        z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-8) # Add epsilon for numerical stability
        z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-8)

        # Cross-correlation matrix
        c = torch.matmul(z1_norm.T, z2_norm) / z1_norm.size(0)  # div by batch size

        # Loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        loss = on_diag + 0.0051 * off_diag  # lambda = 0.0051
        return loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m, "Input matrix must be square"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Instantiate components with consistent latent size
#     latent_size = 2048  # Define a consistent latent size
#     vision_encoder = VisionEncoder(latent_size=latent_size, num_layers=50, pretrained=True, num_channel=1)
#     depth_decoder = DepthDecoder(latent_size=latent_size, num_layers=50).to(device)
#     projector = Projector(in_dim=latent_size, hidden_dim=4*latent_size, out_dim=latent_size).to(device)
    
#     # Move vision encoder to device
#     vision_encoder = vision_encoder.to(device)
    
#     # Create TronModel (without IMU encoder for simplicity)
#     model = TronModel(vision_encoder, projector, latent_size).to(device)
    
#     # Create dummy inputs
#     batch_size = 4
#     thermal_dummy = torch.randn(batch_size, 1, 256, 256).to(device)
#     depth_dummy = torch.randn(batch_size, 1, 256, 256).to(device)
    
#     # Forward pass through TronModel
#     zv1, zv2, v_encoded_thermal, v_encoded_depth, thermal_features, depth_features = model(thermal_dummy, depth_dummy)
    
#     print("\n--- Model Output Shapes ---")
#     print("zv1 shape:", zv1.shape)
#     print("zv2 shape:", zv2.shape)
#     print("v_encoded_thermal shape:", v_encoded_thermal.shape)
#     print("v_encoded_depth shape:", v_encoded_depth.shape)
    
#     print("\n--- Testing Depth Decoder with Skip Connections ---")
#     # Now, predict depth using the thermal features and encoded thermal data
#     pred_depth = depth_decoder(v_encoded_thermal, thermal_features)
#     print("\nPredicted depth shape:", pred_depth.shape)
    
#     # Test Barlow Twins loss
#     barlow_loss = model.barlow_loss(zv1, zv2)
#     print("\nBarlow Loss:", barlow_loss.item())
    
#     # Test Depth Loss
#     depth_loss_fn = DepthLoss()
#     loss = depth_loss_fn(pred_depth, depth_dummy)
#     print("Depth Loss:", loss.item())
    