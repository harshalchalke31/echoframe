import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large
from torchsummary import summary

# ---------------------------------------------------------------------------
# UpsampleBlock
# ---------------------------------------------------------------------------
class UpsampleBlock(nn.Module):
    """
    An upsampling block that:
      - Upsamples the input feature map to the spatial dimensions of a skip connection.
      - Concatenates the upsampled feature with the skip feature.
      - Applies two consecutive Conv-BatchNorm-ReLU operations.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample x to match the spatial size of the skip connection
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        # Concatenate along the channel dimension
        x = torch.cat([x, skip], dim=1)
        # Two conv layers with BN and ReLU activation
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

# ---------------------------------------------------------------------------
# MobileNetV3 Encoder
# ---------------------------------------------------------------------------
class MobileNetV3Encoder(nn.Module):
    """
    Pretrained MobileNetV3-large backbone used as the encoder.
    It returns features from several downsampling stages to serve as skip connections.
    """
    def __init__(self):
        super().__init__()
        # Always use the pretrained MobileNetV3-large
        self.backbone = mobilenet_v3_large(pretrained=True)
        # Remove the classifier head
        self.backbone.classifier = nn.Identity()
        self.features = self.backbone.features
        
        # Define the indices corresponding to layers that downsample the input.
        # (These indices are chosen based on the MobileNetV3-large architecture.)
        self.downsample_indices = [0, 2, 3, 5, len(self.features) - 1]
        if 0 not in self.downsample_indices:
            self.downsample_indices.insert(0, 0)

    def forward(self, x: torch.Tensor):
        features = []
        # Pass through each layer and store outputs at selected indices.
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.downsample_indices:
                features.append(x)
        return tuple(features)

# ---------------------------------------------------------------------------
# MobileNetV3-based UNet Segmentation Model
# ---------------------------------------------------------------------------
class MobileNetV3UNet(nn.Module):
    """
    UNet-style segmentation model using a pretrained MobileNetV3-large encoder.
    The decoder fuses encoder skip connections with upsampling blocks.
    The network outputs a segmentation map with the desired number of classes.
    """
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.encoder = MobileNetV3Encoder()
        
        # Retrieve the number of channels at each encoder stage via a dummy forward pass.
        self.feature_channels = self._get_encoder_channels()
        
        # Build the decoder blocks.
        decoder_blocks = []
        in_channels = self.feature_channels[-1]  # The bottleneck channels
        # Reverse iterate over encoder features (excluding the bottleneck) for skip connections.
        for skip_ch in reversed(self.feature_channels[:-1]):
            # Define output channels as half of the skip connection channels (with a floor of 16)
            out_channels = max(16, skip_ch // 2)
            decoder_blocks.append(UpsampleBlock(in_channels, skip_ch, out_channels))
            in_channels = out_channels
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        
        # Final upsampling (if needed) and segmentation head (1x1 convolution).
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.segmentation_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def _get_encoder_channels(self, input_size: tuple = (3, 224, 224)) -> list:
        """
        Perform a dummy forward pass to get the channel dimensions of encoder features.
        This is a common technique to automatically adapt to the backbone's architecture.
        """
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_size)
            feats = self.encoder(dummy_input)
        channels = [feat.size(1) for feat in feats]
        return channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: obtain multi-scale features.
        encoder_features = self.encoder(x)
        x_dec = encoder_features[-1]  # Start with the bottleneck features
        
        # Decoder: iteratively upsample and fuse skip connections.
        for idx, block in enumerate(self.decoder_blocks):
            # Note: encoder_features are ordered from shallow to deep;
            # we pick them in reverse order (skipping the bottleneck).
            skip_feature = encoder_features[-(idx + 2)]
            x_dec = block(x_dec, skip_feature)
        
        # Final upsampling and segmentation head.
        x_dec = self.final_upsample(x_dec)
        seg_map = self.segmentation_head(x_dec)
        return seg_map

# ---------------------------------------------------------------------------
# Testing the Model
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3UNet(num_classes=1).to(device)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    output = model(dummy_input)
    
    print(model)
    summary(model, input_size=(3, 256, 256), device=str(device))
    print("Output shape:", output.shape)
