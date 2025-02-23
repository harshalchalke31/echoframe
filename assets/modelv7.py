import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large
from torchsummary import summary
from utils import get_config

class SEblock(nn.Module):
    def __init__(self, in_ch:int, reduction:int=4):
        super().__init__()
        C = in_ch
        r = max(1, C//reduction)
        self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        self.squeeze = nn.Linear(C, r, bias=True)
        self.excite = nn.Linear(r, C, bias=True)
        self.relu = nn.ReLU()
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        f = self.globalpool(x)
        f = torch.flatten(f,1)
        f = self.relu(self.squeeze(f))
        f = self.hsigmoid(self.excite(f))
        f = f.view(f.size(0), f.size(1), 1, 1)
        return x * f

class Bottleneck(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, kernel:int, stride:int,
                 exp_size:int, se:bool=False, nl:nn.Module=nn.ReLU()):
        super().__init__()
        self.add = (in_ch == out_ch and stride == 1)
        layers = []
        # expand
        if exp_size != in_ch:
            layers.append(nn.Conv2d(in_ch, exp_size, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(exp_size))
            layers.append(nl)
        else:
            # no expansion needed if exp_size == in_ch
            pass

        # depthwise
        layers.append(nn.Conv2d(exp_size, exp_size, kernel, stride, kernel//2, groups=exp_size, bias=False))
        layers.append(nn.BatchNorm2d(exp_size))
        layers.append(nl)

        # Squeeze-Excitation
        if se:
            layers.append(SEblock(exp_size))

        # project
        layers.append(nn.Conv2d(exp_size, out_ch, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        res = self.block(x)
        if self.add:
            res += x
        return res

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

class MobileNetV3Encoder(nn.Module):
    def __init__(self, inchannels=3, backbone=True, config_name="large", backbone_pretrained=True):
        super().__init__()
        self.backbone = backbone
        self.config_name = config_name
        self.inchannels = inchannels

        if self.backbone:
            # Use pretrained backbone
            self.backbone_model = mobilenet_v3_large(pretrained=backbone_pretrained)
            self.backbone_model.classifier = nn.Identity()
            self.downsample_indices = [0, 2, 3, 5, len(self.backbone_model.features)-1]
        else:
            # Build from scratch using get_config
            self.config = get_config(config_name)
            # Initial layer (similar to MobileNetV3)
            # stem
            layers = []
            layers.append(nn.Conv2d(inchannels, 16, 3, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(16))
            layers.append(nn.Hardswish())

            input_channels = 16
            downsample_count = 0
            self.downsample_indices = []
            current_index = 0

            # Build InvertedResidual blocks
            for (k, exp_size, in_c, out_c, se, nl, s) in self.config:
                # Verify that in_c matches input_channels
                # The configs define in_c and out_c. We assume correct configs:
                block = Bottleneck(input_channels, out_c, k, s, exp_size, se, nl)
                layers.append(block)
                input_channels = out_c
                current_index += 1
                if s == 2:
                    # whenever stride=2, we consider this a downsample stage
                    self.downsample_indices.append(current_index-1) 
                    downsample_count += 1

            # In original MobileNetV3, final conv
            layers.append(nn.Conv2d(input_channels, 960, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(960))
            layers.append(nn.Hardswish())
            current_index += 1
            # final layer considered as highest level feature
            self.downsample_indices.append(current_index-1)

            self.backbone_model = nn.Sequential(*layers)

        # Ensure the first stage is considered as a downsample too
        # The first conv reduces resolution, so index 0 is a downsample stage
        if 0 not in self.downsample_indices:
            self.downsample_indices = [0] + self.downsample_indices

    def forward(self, x: torch.Tensor):
        if self.backbone:
            # Using pretrained backbone
            out = x
            feats = []
            for i, layer in enumerate(self.backbone_model.features):
                out = layer(out)
                if i in self.downsample_indices:
                    feats.append(out)
            return tuple(feats)
        else:
            # Using config-based manual construction
            out = x
            feats = []
            for i, layer in enumerate(self.backbone_model):
                out = layer(out)
                if i in self.downsample_indices:
                    feats.append(out)
            return tuple(feats)

class MobileNetV3UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, config_name="large", backbone=True):
        super().__init__()
        self.image_channels = 3
        self.mask_channels = in_channels - self.image_channels

        # Adapter for mask channel
        self.mask_adapter = nn.Sequential(
            nn.Conv2d(self.mask_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Extract initial image conv (3-channel)
        self.initial_image_conv = self._extract_initial_conv()

        # Use encoder
        # If backbone=True, uses pretrained MobileNetV3 large
        # If backbone=False, builds from scratch using config_name
        self.encoder = MobileNetV3Encoder(inchannels=3, backbone=backbone, config_name=config_name, backbone_pretrained=backbone)

        # Test a dummy pass to get the sizes of features
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, 112, 112)
            dummy_feats = self.encoder(dummy_img)
        feat_channels = [f.size(1) for f in dummy_feats]

        # Build decoder
        decoder_blocks = []
        in_ch = feat_channels[-1]  # top feature map channels
        for i in range(len(feat_channels)-2, -1, -1):
            skip_ch = feat_channels[i]
            out_ch = max(16, skip_ch // 2)
            decoder_blocks.append(UpsampleBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.seg_head = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def _extract_initial_conv(self):
        # We'll rely on MobileNetV3 pretrained weights for the first conv block if backbone=True
        # If backbone=False, we don't actually need pretrained here, but let's keep it consistent.
        base_model = mobilenet_v3_large(pretrained=True)
        first_conv_block = nn.Sequential(
            base_model.features[0][0],
            base_model.features[0][1],
            base_model.features[0][2]
        )
        return first_conv_block

    def forward(self, x):
        # x: B,4,H,W
        img = x[:, :3, :, :]
        mask = x[:, 3:, :, :]

        img_feat = self.initial_image_conv(img)
        mask_feat = self.mask_adapter(mask)
        fused_feat = img_feat + mask_feat

        # Now, since the encoder is built assuming a first downsample step,
        # and we already have fused_feat which corresponds to the output of that step,
        # we feed fused_feat directly into subsequent layers of encoder.
        # For backbone=True, encoder is pretrained. For backbone=False, encoder is custom from config.

        # We must differentiate handling:
        # The encoder either is a pretrained backbone or a custom build.
        # In both cases, we treated fused_feat as the output after the first downsample block.

        # For the pretrained backbone scenario:
        # We've already used the equivalent of features[0], so start from features[1]
        # For custom scenario:
        # We've done the stem conv (equivalent to index=0 layer)
        # So start from layer index=1 in that scenario as well.

        # Extract the layers after the first block:
        if self.encoder.backbone:
            layers = self.encoder.backbone_model.features[1:]
            downsample_indices = self.encoder.downsample_indices
            # Adjust because we've consumed the first layer externally
            # The encoder.downsample_indices includes 0,2,3,5,...
            # We used block 0 externally, so remove it from consideration here
            effective_down_indices = [idx for idx in downsample_indices if idx != 0]
        else:
            layers = self.encoder.backbone_model[1:]
            downsample_indices = self.encoder.downsample_indices
            effective_down_indices = [idx for idx in downsample_indices if idx != 0]

        out = fused_feat
        feats = []
        for i, layer in enumerate(layers, start=1):
            out = layer(out)
            if i in effective_down_indices:
                feats.append(out)

        # Prepend fused_feat as f1
        feats = [fused_feat] + feats  # (f1,f2,f3,f4,f5)

        # Decoder
        x = feats[-1]
        skip_idx = len(feats) - 2
        for block in self.decoder_blocks:
            x = block(x, feats[skip_idx])
            skip_idx -= 1

        x = self.final_upsample(x)
        x = self.seg_head(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3UNet(in_channels=4, out_channels=1, config_name="large", backbone=True).to(device)

    dummy_input = torch.randn(1, 4, 112, 112).to(device)
    output = model(dummy_input)
    print(model)
    summary(model, input_size=(4,112,112), device=str(device))
    print("Output shape:", output.shape)
