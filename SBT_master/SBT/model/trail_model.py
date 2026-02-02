# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
# from torchvision.models import ResNet50_Weights, ResNet18_Weights  # Import weights


# class ResnetEncoder(nn.Module):
#     # Same ResnetEncoder as before (for consistency).  Good practice!
#     def __init__(self, num_layers, pretrained, num_channel=1):
#         super(ResnetEncoder, self).__init__()
#         if num_layers not in [18, 50]:
#             raise ValueError(f"{num_layers} is not a valid number of ResNet layers")

#         if pretrained:
#             if num_layers == 18:
#                 resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
#             elif num_layers == 50:
#                 resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
#             else:
#                 raise ValueError("Pretrained weights only implemented for ResNet-18 and ResNet-50")
#         else:
#             if num_layers == 18:
#                 resnet = models.resnet18(pretrained=False)
#             elif num_layers == 50:
#                 resnet = models.resnet50(pretrained=False)
#             else:
#                 raise ValueError("Must specify a valid ResNet variant")

#         if num_channel != 3:  # Efficient handling of different input channels
#             resnet.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
#             if pretrained: # Properly handle weight initialization
#                 if num_channel == 1:
#                     new_conv1_weight = resnet.conv1.weight.mean(dim=1, keepdim=True)
#                 else: # num_channel > 3
#                     new_conv1_weight = resnet.conv1.weight.mean(dim=1, keepdim=True)
#                     new_conv1_weight = new_conv1_weight.repeat(1, num_channel, 1, 1) / num_channel
#                 resnet.conv1.weight = nn.Parameter(new_conv1_weight)

#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool
#         self.layer1 = resnet.layer1
#         self.layer2 = resnet.layer2
#         self.layer3 = resnet.layer3
#         self.layer4 = resnet.layer4
#         self.avgpool = resnet.avgpool
#         self.fc = nn.Identity()

#     def forward(self, x, return_features=False):
#         skip_features = {}

#         x = self.conv1(x)
#         skip_features['conv1'] = x  # 128
#         x = self.bn1(x)
#         x = self.relu(x)
#         skip_features['layer0'] = x  #128
#         x = self.maxpool(x) # 64
#         x = self.layer1(x)
#         skip_features['layer1'] = x  #64
#         x = self.layer2(x)
#         skip_features['layer2'] = x  #32
#         x = self.layer3(x)
#         skip_features['layer3'] = x  #16
#         x = self.layer4(x)
#         skip_features['layer4'] = x #8

#         if return_features:
#             return skip_features  # Return the skip_features dictionary
#         else:
#             x = self.avgpool(x)
#             x = torch.flatten(x, 1)
#             x = self.fc(x)
#             return x

# class MultiScaleBlock1(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(MultiScaleBlock1, self).__init__()
#         # Define the four parallel convolutional layers
#         self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
#         self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)

#     def forward(self, x):
#         # Pass the input through each parallel convolution
#         out1x1 = self.conv1x1(x)
#         out3x3 = self.conv3x3(x)
#         out5x5 = self.conv5x5(x)
#         out7x7 = self.conv7x7(x)

#         # Concatenate the outputs along the channel dimension
#         out = torch.cat([out1x1, out3x3, out5x5, out7x7], dim=1)
#         return out


# class MultiScaleBlock2(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(MultiScaleBlock2, self).__init__()
#         # Define the plain convolution and dilated convolutions
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.dilated_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
#         self.dilated_conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)

#     def forward(self, x):
#         # Pass the input through each convolution
#         out = self.conv(x)
#         out_dilated2 = self.dilated_conv2(x)
#         out_dilated4 = self.dilated_conv4(x)

#         # Concatenate the outputs
#         out = torch.cat([out, out_dilated2, out_dilated4], dim=1)
#         return out


# class MonocularDepthEstimator(nn.Module):
#     def __init__(self, num_layers=50, pretrained=True, num_input_channels = 1):
#         super(MonocularDepthEstimator, self).__init__()

#         # Encoder (ResNet)
#         self.encoder = ResnetEncoder(num_layers, pretrained, num_channel=num_input_channels)

#         # Get output channels from ResNet encoder layers
#         if num_layers == 50:
#             encoder_channels = {
#                 'layer4': 2048,
#                 'layer3': 1024,
#                 'layer2': 512,
#                 'layer1': 256,
#                 'conv1': 64,
#                 'layer0': 64,  # Add layer0 for consistency
#             }
#         else:  # ResNet18
#             encoder_channels = {
#                 'layer4': 512,
#                 'layer3': 256,
#                 'layer2': 128,
#                 'layer1': 64,
#                 'conv1': 64,
#                 'layer0': 64,  # Add layer0 for consistency
#             }


#         # --- Determine intermediate channel sizes ---
#         # After ResNet, we transition to MultiScaleBlock1. We need to decide
#         # on the number of output channels for each conv in this block.
#         # A common practice is to reduce the number of channels progressively.

#         # Example strategy (you can adjust these):
#         # We'll use the same number of channels for *all* convolutions within
#         # a MultiScaleBlock for simplicity, and reduce the number of channels
#         # in subsequent blocks.

#         block1_out_channels = 512  # Output channels for EACH conv in Block 1
#         block2_out_channels = 128  # Output channels for EACH conv in Block 2
#         # --- Multi-Scale Block 1 (repeated 4 times) ---

#         self.ms_block1_1 = MultiScaleBlock1(encoder_channels['layer4'], block1_out_channels)  # Initial input from ResNet
#         self.ms_block1_2 = MultiScaleBlock1(block1_out_channels * 4, block1_out_channels) # Input is concatenated output of prev
#         self.ms_block1_3 = MultiScaleBlock1(block1_out_channels * 4, block1_out_channels)
#         self.ms_block1_4 = MultiScaleBlock1(block1_out_channels * 4, block1_out_channels)


#         # --- Multi-Scale Block 2 (repeated 4 times) ---

#         # Input to Block 2 is the concatenated output of the last Block 1
#         self.ms_block2_1 = MultiScaleBlock2(block1_out_channels * 4, block2_out_channels)
#         self.ms_block2_2 = MultiScaleBlock2(block2_out_channels * 3, block2_out_channels) # x3 because of 3 convs
#         self.ms_block2_3 = MultiScaleBlock2(block2_out_channels * 3, block2_out_channels)
#         self.ms_block2_4 = MultiScaleBlock2(block2_out_channels * 3, block2_out_channels)

#         # --- Final Convolution ---
#         # After the last Block 2, we need a final convolution to get the single-channel depth map.
#         self.final_conv = nn.Conv2d(block2_out_channels * 3, 1, kernel_size=1)
#         self.sigmoid = nn.Sigmoid() #To bound between 0 and 1

#     def forward(self, x):
#         # --- Encoder ---
#         skip_features = self.encoder(x, return_features = True) # returns feature dictionary
#         features = skip_features['layer4'] # taking layer4 for the architecture
#         # --- Multi-Scale Block 1 ---
#         x = F.relu(self.ms_block1_1(features))  # Apply ReLU after each block
#         x = F.relu(self.ms_block1_2(x))
#         x = F.relu(self.ms_block1_3(x))
#         x = F.relu(self.ms_block1_4(x))
#         # --- Multi-Scale Block 2 ---
#         x = F.relu(self.ms_block2_1(x))
#         x = F.relu(self.ms_block2_2(x))
#         x = F.relu(self.ms_block2_3(x))
#         x = F.relu(self.ms_block2_4(x))
#         # --- Final Convolution ---
#         x = self.final_conv(x)
#         x = self.sigmoid(x)  # Apply sigmoid for the final output
#         return x

# if __name__ == '__main__':
#     # Instantiate the model
#     model = MonocularDepthEstimator(num_layers=50, pretrained=True, num_input_channels = 1)

#     # Create a dummy input tensor (adjust dimensions as needed)
#     batch_size = 4
#     dummy_input = torch.randn(batch_size, 1, 256, 256)  # Example: 1 channel, 256x256 images

#     # Pass the dummy input through the model
#     output = model(dummy_input)

#     # Print the output shape
#     print("Output shape:", output.shape)  # Expected: torch.Size([4, 1, H, W]) - where H and W depend on input size

#     # Example with ResNet-18
#     model18 = MonocularDepthEstimator(num_layers=50, pretrained=True, num_input_channels=3)
#     dummy_input_rgb = torch.randn(batch_size, 3, 256, 256) # RGB
#     output18 = model18(dummy_input_rgb)
#     print("Output shape (ResNet-18, RGB):", output18.shape)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from pytorch_msssim import ssim

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
class VisionEncoder(nn.Module):
    def __init__(self, latent_size=2048, num_layers=50, pretrained=True, num_channel=1):
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
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
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

class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()

    
    def forward(self, pred, target, isTraining = True):
        l1_loss = F.l1_loss(pred, target, reduction='mean')
        smoothness_loss = self.edge_aware_smoothness(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range = 1.0)
        grad_loss = self.gradient_matching_loss(pred, target)
        total_loss = 0.15*l1_loss + 0.85*ssim_loss + 0.6*grad_loss + 0.2*smoothness_loss

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
        gt_depth_denorm = target * 20.0
        recon_depth_denorm = pred * 20.0
        min_valid_depth = 0.1
        max_valid_depth = 20.0
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

        return {
            'abs_rel': abs_rel.item(),
            'rmse': rmse.item(),
            'delta1': delta1.item()
        }


class TronModel(nn.Module):
    def __init__(self, vision_encoder, projector, latent_size=2048):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.projector = projector  # Shared projector
        self.vision_norm = nn.LayerNorm(latent_size) 


    def forward(self, thermal, depth):
        # Vision Encoder
        v_encoded_thermal, thermal_feats = self.vision_encoder(thermal, return_features=True)
        v_encoded_thermal = self.vision_norm(v_encoded_thermal) # Normalize

        v_encoded_depth, depth_feats = self.vision_encoder(depth, return_features=True)
        v_encoded_depth = self.vision_norm(v_encoded_depth) #Normalize

        # Projector (Shared)
        zv1 = self.projector(v_encoded_thermal)
        zv2 = self.projector(v_encoded_depth)

        return zv1, zv2, v_encoded_thermal, v_encoded_depth, thermal_feats, depth_feats

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
    