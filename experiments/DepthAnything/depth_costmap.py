import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from datetime import datetime

# ==========================================
# 1. ENCODER ARCHITECTURE (ResNet50)
# ==========================================
class ResnetEncoder(nn.Module):
    def __init__(self, num_layers, pretrained, num_channel=1):
        super(ResnetEncoder, self).__init__()
        # Standard ResNet50
        resnet = models.resnet50(weights=None) # Weights loaded later via checkpoint

        # Handle 1-channel input (Thermal)
        if num_channel != 3:
            resnet.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Identity()

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
        x = self.fc(x)
        if return_features:
            return x, skip_features
        return x

class VisionEncoder(nn.Module):
    def __init__(self, latent_size=2048, num_layers=50, pretrained=False, num_channel=1):
        super().__init__()
        self.vision_encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained, num_channel=num_channel)

    def forward(self, input_image, return_features=False):
        return self.vision_encoder(input_image, return_features)

# ==========================================
# 2. DECODER ARCHITECTURE (PixelShuffle)
# ==========================================
class ConvGNBlock(nn.Module):
    """
    Convolution + GroupNorm + ReLU.
    Standard block for decoding to maintain training stability.
    """
    def __init__(self, in_ch, out_ch, groups=32):
        super().__init__()
        # Ensure groups doesn't exceed channels
        g = min(groups, in_ch, out_ch)
        if out_ch < 32: g = 1 
            
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=out_ch),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    """
    Upsampling -> Concatenation -> ConvGNBlock
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # The input channel count is the sum of the upsampled features 
        # plus the skip connection features
        self.conv = ConvGNBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        # 1. Bilinear Upsample (2x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        # 2. Handle Padding (in case of odd dimensions)
        if x.shape[2:] != skip.shape[2:]:
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            
        # 3. Concatenate with Skip Connection
        x = torch.cat([x, skip], dim=1)
        
        # 4. Refine Features
        return self.conv(x)

class DepthDecoder(nn.Module):
    def __init__(self, latent_size=2048, num_layers=50):
        super().__init__()
        
        # Determine channel widths based on ResNet size
        if num_layers == 50:
            encoder_channels = [64, 256, 512, 1024, 2048] # [Conv1, Layer1, Layer2, Layer3, Layer4]
        elif num_layers == 18:
            encoder_channels = [64, 64, 128, 256, 512]
        else:
            raise ValueError("Only ResNet 18 and 50 are supported manually here.")

        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[4], 512, kernel_size=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True)
        )

        # 2. Decoder Stages -- INCREASED WIDTH
        # OLD: 512->256->128->64->32
        # NEW: 512->512->256->128->64 (Carries more info)
        
        # Up1: In(512) + Skip(1024) -> Out(512)
        self.up1 = UpBlock(512, encoder_channels[3], 512)
        
        # Up2: In(512) + Skip(512) -> Out(256)
        self.up2 = UpBlock(512, encoder_channels[2], 256)
        
        # Up3: In(256) + Skip(256) -> Out(128)
        self.up3 = UpBlock(256, encoder_channels[1], 128)
        
        # Up4: In(128) + Skip(64) -> Out(64)
        self.up4 = UpBlock(128, encoder_channels[0], 64)
        
        # 3. Final Projection
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # Input is now 64 channels
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, bias=True) 
        )

    def forward(self, embedding_1d, skip_features):
        """
        Args:
            embedding_1d: The 1D vector from the encoder [B, 2048]. (Ignored, see logic above)
            skip_features: Dict containing ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        """
        
        # Step 1: Start with the "Spatial" Thermal Embedding (Layer 4)
        # We prefer this over embedding_1d because it retains spatial context (H/32, W/32).
        x = self.bottleneck(skip_features['layer4']) 
        
        # Step 2: Decode upwards
        x = self.up1(x, skip_features['layer3'])
        x = self.up2(x, skip_features['layer2'])
        x = self.up3(x, skip_features['layer1'])
        x = self.up4(x, skip_features['conv1'])
        
        # Step 3: Final Resolution
        # Output is in Z-Score space (unbounded)
        out = self.final_up(x)
        
        return out
# ==========================================
# 3. WRAPPER & LOADING
# ==========================================
class DeploymentModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    @torch.no_grad()
    def forward(self, x):
        # 1. Encode
        _, skip_features = self.encoder(x, return_features=True)
        # 2. Decode
        # Pass dummy embedding (decoder ignores it, uses skips)
        depth_logits = self.decoder(None, skip_features)
        return depth_logits

def load_deployment_model(encoder_path, decoder_path, device):
    print(f"Loading Encoder: {encoder_path}")
    vision_encoder = VisionEncoder(latent_size=2048, num_layers=50, pretrained=False, num_channel=1)
    enc_ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
    print("Loaded the Encoder Checkpoint")
    # Handle checkpoint structure
    if "vision_encoder" in enc_ckpt:
        state_dict = enc_ckpt["vision_encoder"]
    elif "model" in enc_ckpt:
        state_dict = {k.replace("vision_encoder.", ""): v for k, v in enc_ckpt["model"].items() if "vision_encoder" in k}
    else:
        state_dict = enc_ckpt # fallback
        
    vision_encoder.load_state_dict(state_dict, strict=False)
    
    print(f"Loading Decoder: {decoder_path}")
    depth_decoder = DepthDecoder(latent_size=2048, num_layers=50)
    dec_ckpt = torch.load(decoder_path, map_location=device, weights_only=False)
    print("Loaded the Decoder Checkpoint")
    if "depth_decoder" in dec_ckpt:
        depth_decoder.load_state_dict(dec_ckpt["depth_decoder"], strict=True)
    else:
        raise KeyError("Decoder Checkpoint missing 'depth_decoder' key")

    model = DeploymentModel(vision_encoder, depth_decoder).to(device)
    model.eval()
    return model


if __name__ == "__main__":
    model = load_deployment_model(encoder_path="/mnt/sbackup/Server_3/harshr/home/NV_cahsor/CAHSOR-master/TRON/checkpoint/tron/ssl-ptr-thermal_lidar_video_2/SBT_Plan2-2048-11-25-19-09/SBT_Plan2-2048-11-25-19-09_100.pth",
                decoder_path="/mnt/sbackup/Server_3/harshr/home/NV_cahsor/CAHSOR-master/TRON/checkpoint/tron/decoder_depth_ckpts/SBT_move_base_exp/depth_estimation_move_base-2048-11-27-23-50/depth_decoder_epoch_050.pth",
                # latent_size=2048,
                # num_layers= 50,
                device= 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
                )
    
