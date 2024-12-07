import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchsummary import summary

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

class MobileNetV3Encoder(nn.Module):
    def __init__(self, config_name="large", inchannels=3):
        super().__init__()
        config = self.config(config_name)
        self.config_name = config_name

        # Initial conv
        self.initial = ConvBlock(inchannels, 16, 3, 2, nn.Hardswish())

        self.blocks = nn.ModuleList()
        for (k, exp_size, in_c, out_c, se, nl, s) in config:
            self.blocks.append(Bottleneck(inchannels=in_c, outchannels=out_c, kernelsize=k, stride=s, exp_size=exp_size, se=se, actvn=nl))

    def forward(self, x: Tensor):
        # Encoder forward with skip connections
        x = self.initial(x)
        f1 = x  # 16-ch feature

        f2 = f3 = f4 = f5 = None

        for block in self.blocks:
            out_c = block.block[-1].conv[0].out_channels
            x = block(x)
            # Identify downsampling layers by their output channels
            if out_c == 24 and f2 is None:
                f2 = x
            elif out_c == 40 and f3 is None:
                f3 = x
            elif out_c == 80 and f4 is None:
                f4 = x
            elif out_c == 160 and f5 is None:
                f5 = x

        return f1, f2, f3, f4, f5

    def config(self, name:str):
        HE = nn.Hardswish()
        RE = nn.ReLU()
        large = [
            [3, 16, 16, 16, False, RE, 1],
            [3, 64, 16, 24, False, RE, 2],
            [3, 72, 24, 24, False, RE, 1],
            [5, 72, 24, 40, True, RE, 2],
            [5, 120, 40, 40, True, RE, 1],
            [5, 120, 40, 40, True, RE, 1],
            [3, 240, 40, 80, False, HE, 2],
            [3, 200, 80, 80, False, HE, 1],
            [3, 184, 80, 80, False, HE, 1],
            [3, 184, 80, 80, False, HE, 1],
            [3, 480, 80, 112, True, HE, 1],
            [3, 672, 112, 112, True, HE, 1],
            [5, 672, 112, 160, True, HE, 2],
            [5, 960, 160, 160, True, HE, 1],
            [5, 960, 160, 160, True, HE, 1],
        ]
        small = [
            [3, 16, 16, 16, True, RE, 2],
            [3, 72, 16, 24, False, RE, 2],
            [3, 88, 24, 24, False, RE, 1],
            [5, 96, 24, 40, True, HE, 2],
            [5, 240, 40, 40, True, HE, 1],
            [5, 240, 40, 40, True, HE, 1],
            [5, 120, 40, 48, True, HE, 1],
            [5, 144, 48, 48, True, HE, 1],
            [5, 288, 48, 96, True, HE, 2],
            [5, 576, 96, 96, True, HE, 1],
            [5, 576, 96, 96, True, HE, 1],
        ]
        if name == "large":
            return large
        elif name == "small":
            return small
        else:
            raise ValueError("config_name must be 'large' or 'small'.")

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # We won't define a fixed upsample here. We'll interpolate dynamically in forward.
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        # Dynamically resize x to match skip's size
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        return x

class MobileNetV3UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, config_name="large"):
        super().__init__()
        self.encoder = MobileNetV3Encoder(config_name=config_name, inchannels=in_channels)
        
        # Decoder stages
        # Adjusting channels remain the same
        self.up1 = UpsampleBlock(in_channels=160, skip_channels=80, out_channels=128)
        self.up2 = UpsampleBlock(in_channels=128, skip_channels=40, out_channels=64)
        self.up3 = UpsampleBlock(in_channels=64, skip_channels=24, out_channels=32)
        self.up4 = UpsampleBlock(in_channels=32, skip_channels=16, out_channels=16)

        # For the final upsample, also interpolate dynamically to the original size
        self.seg_head = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        f1, f2, f3, f4, f5 = self.encoder(x)

        x = self.up1(f5, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)

        # Now x should be at 56x56 (same as f1). We need final upsample to get back to 112x112.
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.seg_head(x)
        return x

if __name__ == "__main__":
    # Testing the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy input and move it to device
    dummy_input = torch.randn(1, 3, 112, 112).to(device)

    model = MobileNetV3UNet(in_channels=3, out_channels=1, config_name="large").to(device)

    # Print full model structure
    print(model)

    # Print model summary
    summary(model, input_size=(3,112,112), device=str(device))

    # Forward pass with dummy input on the same device as the model
    output = model(dummy_input)
    print("Output shape:", output.shape)

