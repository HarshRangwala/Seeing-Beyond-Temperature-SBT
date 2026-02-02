import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up1(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec1(x)
        
        x = self.up2(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec2(x)
        
        x = self.up3(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec3(x)
        
        x = self.up4(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec4(x)
        
        x = self.final_conv(x)
        x = self.sigmoid(x)
        
        return x

    
class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        # Optionally add SSIM (install pytorch_msssim: pip install pytorch-msssim)
        # from pytorch_msssim import SSIM
        # self.ssim_loss = SSIM(data_range=1.0, channel=1)  # Assuming depth range [0, 1]

    def forward(self, pred, target):
        # valid_mask = (target > 0).float()
        # pred_masked = pred * valid_mask
        # target_masked = target * valid_mask
        pred_denorm = pred * 20
        target_denorm = target * 20.0
        min_valid_depth = 0.1
        max_valid_depth = 20.0
        valid_mask = (target_denorm >= min_valid_depth) & (target_denorm <= max_valid_depth)
        pred_masked = pred_denorm[valid_mask]
        target_masked = target_denorm[valid_mask]
        l1 = self.l1_loss(pred_masked, target_masked)
        return l1
    @torch.no_grad()
    # def compute_depth_metrics(self, pred, target):
    #     """Compute additional depth metrics for evaluation"""
    #     # valid_mask = (target > 0).float()
        
    #     # # Mask out invalid pixels
    #     # pred = pred * valid_mask
    #     # target = target * valid_mask

    #     target_denorm = target * 20.0
    #     valid_mask = (target_denorm > 1e-3) & (target_denorm < 20.0)
    #     pred = pred[valid_mask]
    #     target = target_denorm[valid_mask]
    #     # print(f"Pred range: {pred.min().item():.2f}-{pred.max().item():.2f}m")
    #     # print(f"GT range: {target_denorm.min().item():.2f}-{target_denorm.max().item():.2f}m")
    #     # print(f"Valid pixels: {valid_mask.float().mean().item()*100:.1f}%")
        
    #     # Absolute Relative Error
    #     abs_rel = torch.mean(torch.abs(pred - target) / (target + 1e-10))
        
    #     # RMSE
    #     rmse = torch.sqrt(torch.mean((pred - target) ** 2))
        
    #     # δ1 accuracy (threshold = 1.25)
    #     thresh = torch.max((target / (pred + 1e-10)), (pred / (target + 1e-10)))
    #     delta1 = (thresh < 1.25).float().mean()

    #     return {
    #         'abs_rel': abs_rel.item(),
    #         'rmse': rmse.item(),
    #         'delta1': delta1.item()
    #     }
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
        delta2 = (thresh < 1.25**2).float().mean()  # Delta2
        delta3 = (thresh < 1.25**3).float().mean()  # Delta3

        return {
            'abs_rel': abs_rel.item(),
            'rmse': rmse.item(),
            'delta1': delta1.item(),
            'delta2': delta2.item(),  # Add delta2
            'delta3': delta3.item()   # Add delta3
        }

    # def eval_depth(self, pred, target):
    #     assert pred.shape == target.shape

    #     thresh = torch.max((target / pred), (pred / target))

    #     d1 = torch.sum(thresh < 1.25).float() / len(thresh)

    #     diff = pred - target
    #     diff_log = torch.log(pred) - torch.log(target)

    #     abs_rel = torch.mean(torch.abs(diff) / target)

    #     rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    #     mae = torch.mean(torch.abs(diff))

    #     silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    #     return {'d1': d1.detach(), 'abs_rel': abs_rel.detach(),'rmse': rmse.detach(), 'mae': mae.detach(), 'silog':silog.detach()}
    
def unet_model(bilinear=False):
    """Creates a UNet model instance."""
    model = UNet(in_channels=1, out_channels=1)  # 1 input channel (thermal), 1 output channel (depth)
    return model
    
# if __name__ == '__main__':
#     # --- Example Usage ---
#     model = unet_model()  # Create the model
#     print(model)  # Print model architecture

#     # Dummy input: Batch size 4, 1 channel, 256x256 image
#     dummy_input = torch.randn(4, 1, 256, 256)
#     output = model(dummy_input)
#     print("Output shape:", output.shape) # Expected output: torch.Size([4, 1, 256, 256])

#     # Example loss calculation:
#     criterion = DepthLoss()
#     dummy_target = torch.rand(4, 1, 256, 256)  # Dummy target depth maps
#     loss = criterion(output, dummy_target)
#     print("Loss:", loss.item())

    # Example with checkpointing (if you need it for large models):
    # model_checkpointed = unet_model(bilinear=True)
    # model_checkpointed.use_checkpointing()
    # output_checkpointed = model_checkpointed(dummy_input)
    # print("Output shape (checkpointed):", output_checkpointed.shape)