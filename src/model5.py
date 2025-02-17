import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# A simple decoder block that upsamples and fuses encoder features (skip connection)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, skip):
        # Upsample input x to match skip's spatial size.
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        # Concatenate along the channel dimension.
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class SwinUNet(nn.Module):
    def __init__(self, num_classes=1, img_size=224, encoder_name="swin_base_patch4_window7_224", pretrained=True):
        """
        Args:
            num_classes: Number of segmentation classes.
            img_size: Input image resolution (assumed square).
            encoder_name: Name of the Swin Transformer model from timm.
            pretrained: Whether to load pretrained weights.
        """
        super().__init__()
        
        # Create a Swin Transformer backbone that outputs features from multiple stages.
        # Note: timm's features_only mode returns a list of feature maps.
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)  # returns features from 4 stages
        )
        
        # Get encoder feature channels from timm's feature_info.
        self.enc_channels = self.encoder.feature_info.channels()
        # For example, for swin_base_patch4_window7_224, this might be [128, 256, 512, 1024].
        
        # Decoder: Build U-Net style upsampling blocks.
        # We fuse the deepest features (stage 3) with stage 2, then stage 1, then stage 0.
        self.decoder3 = DecoderBlock(in_channels=self.enc_channels[3],
                                     skip_channels=self.enc_channels[2],
                                     out_channels=512)
        self.decoder2 = DecoderBlock(in_channels=512,
                                     skip_channels=self.enc_channels[1],
                                     out_channels=256)
        self.decoder1 = DecoderBlock(in_channels=256,
                                     skip_channels=self.enc_channels[0],
                                     out_channels=128)
        
        # Final block to upsample to the original image resolution.
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Get multi-scale features from the encoder.
        features = self.encoder(x)
        
        # --- Ensure features are in channels-first format ---
        # Some backbones might return features in channels-last format.
        fixed_features = []
        for i, feat in enumerate(features):
            # If the second dimension does not match the expected channel count,
            # we assume the tensor is channels-last and permute it.
            if feat.ndim == 4 and feat.shape[1] != self.enc_channels[i]:
                # Check if the last dimension matches the expected channel count.
                if feat.shape[-1] == self.enc_channels[i]:
                    feat = feat.permute(0, 3, 1, 2).contiguous()
            fixed_features.append(feat)
        features = fixed_features
        # -----------------------------------------------
        
        # features list: [stage0, stage1, stage2, stage3]
        # For a typical swin_base, expected shapes are:
        #  - stage0: [B, 128, 56, 56]
        #  - stage1: [B, 256, 28, 28]
        #  - stage2: [B, 512, 14, 14]
        #  - stage3: [B, 1024, 7, 7]
        
        # Start decoding from the deepest features.
        x_deep = features[-1]  # e.g., [B, 1024, 7, 7]
        
        # Decoder stage 3: fuse with encoder stage 2.
        x = self.decoder3(x_deep, features[-2])  # Expected output: [B, 512, 14, 14]
        # Decoder stage 2: fuse with encoder stage 1.
        x = self.decoder2(x, features[-3])         # Expected output: [B, 256, 28, 28]
        # Decoder stage 1: fuse with encoder stage 0.
        x = self.decoder1(x, features[-4])         # Expected output: [B, 128, 56, 56]
        
        # Upsample to the original resolution.
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        out = self.final_conv(x)
        return out

# Example usage:
if __name__ == "__main__":
    model = SwinUNet(num_classes=1,
                     img_size=224,
                     encoder_name="swin_base_patch4_window7_224",
                     pretrained=True)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (2, 1, 224, 224)
