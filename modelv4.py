import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import mobilenet_v3_large,mobilenet_v3_small
from utils import get_config

class ConvBlock(nn.Module):
    def __init__(self, inchannels:int, outchannels:int, kernelsize:int, stride:int,
                 actvn:nn.Module=nn.ReLU(), groups:int=1, bn:bool=True, bias:bool=False):
        super().__init__()
        padding = kernelsize // 2
        layers = [nn.Conv2d(inchannels, outchannels, kernelsize, stride, padding, groups=groups, bias=bias)]
        if bn:
            layers.append(nn.BatchNorm2d(outchannels))
        if actvn is not None:
            layers.append(actvn)
        self.conv = nn.Sequential(*layers)

    def forward(self, x:Tensor)->Tensor:
        return self.conv(x)

class SEblock(nn.Module):
    def __init__(self, inchannels:int, reduction:int=4):
        super().__init__()
        C = inchannels
        r = max(1, C//reduction)
        self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        self.squeeze = nn.Linear(C, r, bias=True)
        self.excite = nn.Linear(r, C, bias=True)
        self.relu = nn.ReLU()
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self,x:Tensor)->Tensor:
        f = self.globalpool(x)
        f = torch.flatten(f,1)
        f = self.relu(self.squeeze(f))
        f = self.hsigmoid(self.excite(f))
        f = f.view(f.size(0), f.size(1), 1, 1)
        return x * f

class Bottleneck(nn.Module):
    def __init__(self, inchannels:int, outchannels:int, kernelsize:int, stride:int,
                 exp_size:int, se:bool=False, actvn:nn.Module=nn.ReLU()):
        super().__init__()
        self.add = (inchannels == outchannels and stride == 1)
        self.block = nn.Sequential(
            ConvBlock(inchannels, exp_size, 1, 1, actvn=actvn),
            ConvBlock(exp_size, exp_size, kernelsize, stride, actvn=actvn, groups=exp_size),
            SEblock(exp_size) if se else nn.Identity(),
            ConvBlock(exp_size, outchannels, 1, 1, actvn=None)
        )

    def forward(self,x:Tensor)->Tensor:
        res = self.block(x)
        if self.add:
            res += x
        return res

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class MobileNetV3Encoder(nn.Module):
    def __init__(self, config_name="large", inchannels=3, backbone=False):
        super().__init__()
        self.backbone = backbone
        self.config_name = config_name
        
        if self.backbone:
            if config_name=="large":
                # Use torchvision's pretrained mobilenet_v3_large as encoder
                self.backbone_model = mobilenet_v3_large(pretrained=True)
                # We'll extract features from self.backbone_model.features
                # We'll identify downsample points by inspecting the model.
                # For demonstration, let's pick some indices (adjust as needed):
                # self.downsample_indices = [0, 2, 3, 5, len(self.backbone_model.features)-1]
            else:
                self.backbone_model = mobilenet_v3_small(pretrained=True)
            self.downsample_indices = [0, 2, 3, 5, len(self.backbone_model.features)-1]
        else:
            # Use config-based custom implementation
            self.config = get_config(config_name)
            self.initial = ConvBlock(inchannels, 16, 3, 2, nn.Hardswish())
            self.blocks = nn.ModuleList()
            for (k, exp_size, in_c, out_c, se, nl, s) in self.config:
                self.blocks.append(Bottleneck(
                    inchannels=in_c,
                    outchannels=out_c,
                    kernelsize=k,
                    stride=s,
                    exp_size=exp_size,
                    se=se,
                    actvn=nl
                ))

    def forward(self, x: Tensor):
        if self.backbone:
            # Using pretrained backbone
            out = x
            feats = []
            for i, layer in enumerate(self.backbone_model.features):
                out = layer(out)
                if i in self.downsample_indices:
                    feats.append(out)
            # feats should be (f1, f2, f3, f4, f5)
            return tuple(feats)
        else:
            # Using config-based manual construction
            x = self.initial(x)
            f1 = x
            skips = []
            for block, params in zip(self.blocks, self.config):
                s = params[-1]
                x = block(x)
                if s == 2:
                    skips.append(x)
            return (f1, *skips)

class MobileNetV3UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, config_name="large", backbone=False):
        super().__init__()
        self.encoder = MobileNetV3Encoder(config_name=config_name, inchannels=in_channels, backbone=backbone)

        # Run a dummy pass to know skip structures
        dummy_input = torch.randn(1, in_channels, 112, 112)
        with torch.no_grad():
            features = self.encoder(dummy_input)
        # features = (f1, skip1, skip2, ...)
        
        self.n_skips = len(features) - 1
        channels = [f.size(1) for f in features]

        self.decoder_blocks = nn.ModuleList()
        dec_in = channels[-1]
        # Decode up to f1
        for i in range(self.n_skips - 1, 0, -1):
            skip_ch = channels[i]
            out_ch = max(16, skip_ch // 2)
            self.decoder_blocks.append(UpsampleBlock(in_channels=dec_in, skip_channels=skip_ch, out_channels=out_ch))
            dec_in = out_ch

        f1_ch = channels[0]
        out_ch = max(16, f1_ch // 2)
        self.decoder_blocks.append(UpsampleBlock(in_channels=dec_in, skip_channels=f1_ch, out_channels=out_ch))
        dec_in = out_ch

        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.seg_head = nn.Conv2d(dec_in, out_channels, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        f1 = features[0]
        skips = features[1:]

        x = skips[-1]
        idx = len(skips) - 2

        for block in self.decoder_blocks[:-1]:
            x = block(x, skips[idx])
            idx -= 1

        last_block = self.decoder_blocks[-1]
        x = last_block(x, f1)

        x = self.final_upsample(x)
        x = self.seg_head(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Example: backbone=False uses custom config, backbone=True uses pretrained mobilenet_v3_large
    model = MobileNetV3UNet(in_channels=3, out_channels=1, config_name="large", backbone=False).to(device)
    
    dummy_input = torch.randn(1, 3, 112, 112).to(device)
    print(model)
    summary(model, input_size=(3,112,112), device=str(device))
    output = model(dummy_input)
    print("Output shape:", output.shape)
