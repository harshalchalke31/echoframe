import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary 

class ConvBlock(nn.Module):
    def __init__(self, inchannels:int,outchannels:int,kernelsize:int,stride:int,actvn=nn.ReLU(),groups=1,bn:bool=True,bias:bool=False):
        super().__init__()
        # set dynamic padding according to kernel size
        padding = kernelsize//2 # k=1 -> p=0, k=7 -> p=3
        self.conv =  nn.Sequential(
            nn.Conv2d(in_channels=inchannels,
                      out_channels=outchannels,
                      kernel_size=kernelsize,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias),
            nn.BatchNorm2d(outchannels) if bn else nn.Identity(),
            actvn,
        )

    def forward(self,x:Tensor)->Tensor:
        return self.conv(x)
    
class SEblock(nn.Module):
    def __init__(self, inchannels,reduction:int=4):
        super().__init__()
        C = inchannels
        r = inchannels//reduction # reduction factor
        self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        self.squeeze = nn.Linear(in_features=C,out_features=r,bias=True)
        self.excite = nn.Linear(in_features=r,out_features=C,bias=True)
        self.relu = nn.ReLU()
        self.hsigmoid = nn.Hardsigmoid()
    def forward(self,x:Tensor)->Tensor:
        # x = [N,C,H,W]
        f = self.globalpool(x) # [N,C,1,1]
        f = torch.flatten(f,1) # [N,C]
        f = self.relu(self.squeeze(f)) #[N,r]
        f = self.hsigmoid(self.excite(f)) # [N,C]
        f = f.view(f.size(0),f.size(1),1,1) # reshape to f = [N,C,1,1]
        return x*f # channel wise scaling

class Bottleneck(nn.Module):
    def __init__(self,inchannels:int,outchannels:int,kernelsize:int, stride:int,
                 exp_size:int,se:bool=False,actvn=nn.modules.activation):
        super().__init__()
        self.add = inchannels == outchannels and stride==1
        self.block = nn.Sequential(
            ConvBlock(inchannels=inchannels,outchannels=exp_size,kernelsize=1,stride=1,actvn=actvn),
            # depthwise conv
            ConvBlock(inchannels=exp_size,outchannels=exp_size,kernelsize=kernelsize,stride=stride,actvn=actvn,groups=exp_size),
            SEblock(inchannels=exp_size) if se == True else nn.Identity(),
            ConvBlock(inchannels=exp_size,outchannels=outchannels,kernelsize=1,stride=1,actvn=nn.Identity())
        )

    def forward(self,x:Tensor)->Tensor:
        res = self.block(x)
        if self.add:
            res+=x
        return res

""" MobileNetV3 """
class MobileNetV3(nn.Module):
    def __init__(self,config_name : str,inchannels = 3,classes = 1000):
        super().__init__()
        config = self.config(config_name)

        # First convolution(conv2d) layer. 
        self.conv = ConvBlock(inchannels=inchannels, outchannels=16, kernelsize=3, stride=2, actvn=nn.Hardswish())
        # Bneck blocks in a list. 
        self.blocks = nn.ModuleList([])
        for c in config:
            kernel_size, exp_size, in_channels, out_channels, se, nl, s = c
            self.blocks.append(Bottleneck(inchannels=in_channels, outchannels=out_channels, kernelsize=kernel_size,
                                          stride=s,exp_size=exp_size, se=se, actvn=nl))
        
        # Classifier 
        last_outchannel = config[-1][3]
        last_exp = config[-1][1]
        out = 1280 if config_name == "large" else 1024
        self.classifier = nn.Sequential(
            ConvBlock(last_outchannel, last_exp, 1, 1, nn.Hardswish()),
            nn.AdaptiveAvgPool2d((1,1)),
            ConvBlock(last_exp, out, 1, 1, nn.Hardswish(), bn=False, bias=True),
            nn.Dropout(0.8),
            nn.Conv2d(out, classes, 1, 1)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)

        x = self.classifier(x)
        return torch.flatten(x, 1)


    def config(self, name):
        HE, RE = nn.Hardswish(), nn.ReLU()
        # [kernel, exp size, in_channels, out_channels, SEBlock(SE), activation function(NL), stride(s)] 
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
                [5, 960, 160, 160, True, HE, 1]
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
                [5, 576, 96, 96, True, HE, 1]
        ]

        if name == "large": return large
        if name == "small": return small

# testing
if __name__ == "__main__":
    conv = ConvBlock(
        inchannels=3,
        outchannels=16,
        kernelsize=3,
        stride=1,
        actvn=nn.ReLU(),
        bn=True,
        bias=False
    )
    conv_input = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image
    # forward pass
    conv_output = conv(conv_input)
    print(f"Architecture: {conv}\nOutput shape: {conv_output.shape}")

    se = SEblock(inchannels=64)
    se_input = torch.randn(1,64,32,32) # Batch size 1, 64 channels, 32x32 spatial size
    se_output = se(se_input)
    print(f"Architecture: {se}\nInput shape: {se_input.shape}\nOutput shape: {se_output.shape}")

    bottleneck = Bottleneck(inchannels=32,outchannels=32,kernelsize=3,stride=1,exp_size=64,se=True,actvn=nn.ReLU())
    blk_input = torch.randn(1, 32, 64, 64)  # Batch size 1, 32 channels, 64x64 spatial size
    blk_output = bottleneck(blk_input)
    print(f"Architecture: {bottleneck}\nInput shape: {blk_input.shape}\nOutput shape: {blk_output.shape}")

    name = "large"
    rho = 1
    res = int(rho * 224)

    mbnet = MobileNetV3(name)
    mbnet_input = torch.rand(1, 3, res, res)
    mbnet_output = mbnet(mbnet_input)
    print(f'MobileNetV3 Architecture: {mbnet}\nInput shape: {mbnet_input.shape}\nOutput shape: {mbnet_output.shape}')


