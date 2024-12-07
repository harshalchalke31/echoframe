import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large
from torchsummary import summary

# ------------------------------
# Utility Blocks
# ------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, act=True, bn=True):
        super().__init__()
        padding = kernel_size // 2
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # A double conv block following the UNet pattern
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        # Upsample to the skip feature map size
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


# ------------------------------
# MobileNetV3-based Encoder
# ------------------------------
class MobileNetV3Encoder(nn.Module):
    def __init__(self, inchannels=3, backbone_pretrained=True):
        super().__init__()
        # Load a pretrained MobileNetV3-Large backbone
        self.backbone_model = mobilenet_v3_large(pretrained=backbone_pretrained)
        # Replace classifier with identity since we only need features
        self.backbone_model.classifier = nn.Identity()

        # Downsample indices are chosen to extract features at various scales:
        # - 0: After first downsampling
        # - 2, 3, 5: Further downsamplings in MobileNetV3 features
        # - len(...) - 1: final high-level feature
        # Adjust these indices if you want different feature maps as skip connections.
        self.downsample_indices = [0, 2, 3, 5, len(self.backbone_model.features)-1]

        self.inchannels = inchannels
        if inchannels != 3:
            # We will not directly modify the first conv layer of MobileNet.
            # Instead, we handle extra channels outside and then feed 3-channel features into MobileNet.
            # Hence, no direct modification needed here.
            pass

    def forward(self, x):
        """
        x is expected to be a 3-channel tensor when passed to MobileNet.
        Preprocessing outside this encoder should ensure x is 3-channel,
        possibly by fusing extra channels beforehand.
        """
        out = x
        feats = []
        for i, layer in enumerate(self.backbone_model.features):
            out = layer(out)
            # If i is in downsample_indices, store this feature map
            if i in self.downsample_indices:
                feats.append(out)
        # returns tuple of (f1, f2, f3, f4, f5)
        return tuple(feats)


# ------------------------------
# Improved UNet with MobileNetV3 Encoder
# and a separate path for the mask channel
# ------------------------------
class MobileNetV3UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, backbone_pretrained=True):
        super().__init__()

        # We separate the input into two parts:
        # - The first 3 channels (echocardiogram image)
        # - The 1 channel (previous segmentation mask)

        # Pretrained MobileNet expects 3-channel input. We'll keep that.
        # We'll create a separate small block to process the mask channel
        # and then fuse it with the image features before feeding into the encoder.

        self.image_channels = 3
        self.mask_channels = in_channels - self.image_channels

        # Small adapter for mask channel:
        # This will convert 1 mask channel into a feature map compatible with the initial MobileNet layer.
        # We try to be lightweight: a single conv layer to transform mask into a 16-channel feature map,
        # same dimension as the initial MobileNet output. Then we fuse by addition.
        self.mask_adapter = nn.Sequential(
            nn.Conv2d(self.mask_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # We'll take the MobileNet first layer weights as is. After first layer, MobileNet produces 16-channel features.
        # MobileNet input: 3-channel image
        self.initial_image_conv = self._extract_initial_conv()

        # Use encoder
        self.encoder = MobileNetV3Encoder(inchannels=3, backbone_pretrained=backbone_pretrained)

        # Test a dummy pass to get the sizes of features
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, 112, 112)
            dummy_feats = self.encoder(dummy_img)
        # dummy_feats = (f1, f2, f3, f4, f5)
        # Let's store channel sizes:
        feat_channels = [f.size(1) for f in dummy_feats]

        # We have 5 feature maps: f1, f2, f3, f4, f5
        # We'll build decoder blocks in a symmetrical manner (like UNet).
        # Typically:
        # top-level (f5) -> upsample and combine with f4
        # then that -> upsample and combine with f3
        # ... -> upsample and combine with f2
        # and finally -> upsample and combine with f1

        # We'll build decoder blocks accordingly:
        decoder_blocks = []
        # Pair (f5 with f4), (then f4 with f3), (then f3 with f2), (then f2 with f1)
        # Start from top (f5)
        in_ch = feat_channels[-1]  # channels in f5
        for i in range(len(feat_channels)-2, -1, -1):
            skip_ch = feat_channels[i]
            out_ch = max(16, skip_ch // 2)
            decoder_blocks.append(UpsampleBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        # After final decoder block, we might upsample once more if we need original resolution
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.seg_head = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def _extract_initial_conv(self):
        # Extract the very first conv, bn, activation from MobileNetV3
        # We'll use these weights to process the 3 image channels.
        base_model = mobilenet_v3_large(pretrained=True)
        # The first layer is features[0][0], features[0][1], features[0][2]
        # which is Conv2d, BatchNorm2d, and Activation. We'll replicate that block.
        first_conv_block = nn.Sequential(
            base_model.features[0][0],
            base_model.features[0][1],
            base_model.features[0][2]
        )
        return first_conv_block

    def forward(self, x):
        # x: B,4,H,W (3 image channels + 1 mask channel)
        # Separate image and mask
        img = x[:, :3, :, :]
        mask = x[:, 3:, :, :]

        # Pass image through initial conv (using pretrained weights)
        img_feat = self.initial_image_conv(img)
        # Pass mask through adapter
        mask_feat = self.mask_adapter(mask)
        # Fuse by addition
        fused_feat = img_feat + mask_feat

        # Now pass fused_feat through the remaining MobileNet layers
        # However, note that our encoder expects a 3-channel input to MobileNet.
        # We have already passed the image through the first MobileNet block (initial_image_conv).
        # fused_feat is now the output of that block plus mask features.
        # To continue, we must feed fused_feat into the next layers of the MobileNet encoder.
        # The encoder's forward includes the entire features stack of MobileNet.
        # We need a slight modification to skip the first block of MobileNet since we've already applied it.

        # Let's do that by accessing encoder.backbone_model.features directly:
        # We used up the first block of MobileNet (features[0]) externally.
        # So we start from features[1]:
        out = fused_feat
        feats = []
        for i, layer in enumerate(self.encoder.backbone_model.features[1:], start=1):
            out = layer(out)
            if i in self.encoder.downsample_indices:
                feats.append(out)

        # Remember, the encoder.downsample_indices included 0 which we already used:
        # We must also include the fused_feat as the first feature map (f1).
        # The original downsample_indices = [0,2,3,5,...]
        # We took care of index=0 by the initial_image_conv and fuse step.
        # So feats currently excludes the first stage output (fused_feat).
        # Prepend fused_feat to feats to maintain the (f1, f2, f3, f4, f5) structure:
        feats = [fused_feat] + feats  # now feats = (f1,f2,f3,f4,f5)

        # Now decode:
        # feats = [f1, f2, f3, f4, f5]
        x = feats[-1]  # start from the top feature (f5)
        skip_idx = len(feats) - 2
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, feats[skip_idx])
            skip_idx -= 1

        # x is now at the same spatial level as f1
        # Optionally upsample to match original input size
        x = self.final_upsample(x)
        x = self.seg_head(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3UNet(in_channels=4, out_channels=1, backbone_pretrained=True).to(device)

    dummy_input = torch.randn(1, 4, 112, 112).to(device)
    output = model(dummy_input)
    print(model)
    summary(model, input_size=(4,112,112), device=str(device))
    print("Output shape:", output.shape)
