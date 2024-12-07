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

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        # Match spatial size of skip
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        return x

class MobileNetV3Encoder(nn.Module):
    def __init__(self, config_name="large", inchannels=3):
        super().__init__()
        self.config = self.get_config(config_name)
        self.config_name = config_name
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
        x = self.initial(x)
        f1 = x  # This is at 56x56
        skips = []

        # Collect skip features at each downsampling step
        for block, params in zip(self.blocks, self.config):
            s = params[-1]  # stride
            x = block(x)
            if s == 2:
                # Store skip feature after downsampling
                skips.append(x)

        # (f1, skip1, skip2, ...)
        return (f1, *skips)

    def get_config(self, name:str):
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

class MobileNetV3UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, config_name="large"):
        super().__init__()
        self.encoder = MobileNetV3Encoder(config_name=config_name, inchannels=in_channels)

        # Run a dummy pass to know skip structures
        dummy_input = torch.randn(1, in_channels, 112, 112)
        with torch.no_grad():
            features = self.encoder(dummy_input)
        # features = (f1, skip1, skip2, ...)
        # The last element of features is the deepest skip

        self.n_skips = len(features) - 1  # number of actual skip features (excluding f1)
        channels = [f.size(1) for f in features]

        self.decoder_blocks = nn.ModuleList()
        # We'll decode from deepest skip to the shallower ones, ending at f1 scale.
        # Example: if we have (f1, skip1, skip2, skip3), skip3 is deepest.
        # We'll do up1 with skip3 and skip2, up2 with that result and skip1, etc., until f1.

        dec_in = channels[-1]  # channels of the deepest skip
        for i in range(self.n_skips - 1, 0, -1):
            # i runs backward from last skip to skip1
            skip_ch = channels[i]
            out_ch = max(16, skip_ch // 2)
            self.decoder_blocks.append(UpsampleBlock(in_channels=dec_in, skip_channels=skip_ch, out_channels=out_ch))
            dec_in = out_ch

        # Now decode with f1 (the shallowest "skip")
        f1_ch = channels[0]  # f1's channels
        # final upsample block to merge x with f1
        out_ch = max(16, f1_ch // 2)
        self.decoder_blocks.append(UpsampleBlock(in_channels=dec_in, skip_channels=f1_ch, out_channels=out_ch))
        dec_in = out_ch

        # After merging with f1, we have a feature map at 56x56.
        # We need to get back to 112x112:
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # seg_head must match dec_in channels (after merging with f1) because final block sets dec_in
        self.seg_head = nn.Conv2d(dec_in, out_channels, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        # features = (f1, skip1, skip2, ... skipN)
        # deepest skip is features[-1], shallowest skip is features[1], and f1 is features[0]

        f1 = features[0]
        skips = features[1:]

        # Start from deepest skip:
        x = skips[-1]
        idx = len(skips) - 2

        # Go through decoder blocks except the last one:
        for block in self.decoder_blocks[:-1]:
            x = block(x, skips[idx])
            idx -= 1
        
        # Now the last decoder block merges x with f1:
        last_block = self.decoder_blocks[-1]
        x = last_block(x, f1)

        # x is now at f1 resolution (56x56), upsample to original 112x112
        x = self.final_upsample(x)

        # final segmentation head
        x = self.seg_head(x)
        return x


if __name__ == "__main__":
    # Testing the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dummy_input = torch.randn(1, 3, 112, 112).to(device)
    model = MobileNetV3UNet(in_channels=3, out_channels=1, config_name="large").to(device)

    print(model)
    summary(model, input_size=(3,112,112), device=str(device))
    output = model(dummy_input)
    print("Output shape:", output.shape)
