import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, inchannels:int, outchannels:int, kernelsize:int, stride:int,
                 actvn:nn.Module=nn.ReLU(), groups:int=1, bn:bool=True, bias:bool=False):
        super().__init__()
        # set dynamic padding according to kernel size
        padding = kernelsize//2
        layers = [
            nn.Conv2d(in_channels=inchannels,
                      out_channels=outchannels,
                      kernel_size=kernelsize,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias)
        ]
        if bn:
            layers.append(nn.BatchNorm2d(outchannels))
        layers.append(actvn)
        self.conv = nn.Sequential(*layers)

    def forward(self,x:Tensor)->Tensor:
        return self.conv(x)
    
class SEblock(nn.Module):
    def __init__(self, inchannels:int, reduction:int=4):
        super().__init__()
        C = inchannels
        r = C//reduction
        self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        self.squeeze = nn.Linear(in_features=C, out_features=r, bias=True)
        self.excite = nn.Linear(in_features=r, out_features=C, bias=True)
        self.relu = nn.ReLU()
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self,x:Tensor)->Tensor:
        # x = [N,C,H,W]
        f = self.globalpool(x) # [N,C,1,1]
        f = torch.flatten(f,1) # [N,C]
        f = self.relu(self.squeeze(f)) #[N,r]
        f = self.hsigmoid(self.excite(f)) # [N,C]
        f = f.view(f.size(0), f.size(1), 1, 1) # reshape to [N,C,1,1]
        return x * f # channel wise scaling

class Bottleneck(nn.Module):
    def __init__(self, inchannels:int, outchannels:int, kernelsize:int, stride:int,
                 exp_size:int, se:bool=False, actvn:nn.Module=nn.ReLU()):
        super().__init__()
        self.add = (inchannels == outchannels and stride == 1)
        self.block = nn.Sequential(
            # 1x1 pointwise
            ConvBlock(inchannels=inchannels, outchannels=exp_size, kernelsize=1, stride=1, actvn=actvn),
            # depthwise
            ConvBlock(inchannels=exp_size, outchannels=exp_size, kernelsize=kernelsize, stride=stride, actvn=actvn, groups=exp_size),
            SEblock(inchannels=exp_size) if se else nn.Identity(),
            # final pointwise with no activation
            ConvBlock(inchannels=exp_size, outchannels=outchannels, kernelsize=1, stride=1, actvn=nn.Identity())
        )

    def forward(self,x:Tensor)->Tensor:
        res = self.block(x)
        if self.add:
            res += x
        return res
    
class Upsample(nn.Module):
    def __init__(self, filters:int):
        super().__init__()
        self.filters = filters
        self.up_samp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(filters * 2, filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x:Tensor, skip_features:Tensor)->Tensor:
        x = self.up_samp(x)
        x = torch.cat([x, skip_features], dim=1)  # Skip connection
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        return x


class MobileNetV3(nn.Module):
    def __init__(self, config_name: str = "large", inchannels=3, classes=1000):
        super().__init__()
        config = self.config(config_name)

        # First convolution layer
        self.conv = ConvBlock(inchannels=inchannels, outchannels=16, kernelsize=3, stride=2, actvn=nn.Hardswish())

        # Bottleneck layers
        self.blocks = nn.ModuleList([])
        for c in config:
            kernel_size, exp_size, in_channels, out_channels, se, nl, s = c
            # nl is either nn.ReLU() or nn.Hardswish() from config
            self.blocks.append(Bottleneck(inchannels=in_channels, 
                                          outchannels=out_channels, 
                                          kernelsize=kernel_size,
                                          stride=s, 
                                          exp_size=exp_size, 
                                          se=se, 
                                          actvn=nl))

        # Classifier
        last_outchannel = config[-1][3]
        last_exp = config[-1][1]
        out = 1280 if config_name == "large" else 1024
        self.classifier = nn.Sequential(
            ConvBlock(last_outchannel, last_exp, 1, 1, nn.Hardswish()),
            nn.AdaptiveAvgPool2d((1, 1)),
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

    def config(self, name:str):
        HE = nn.Hardswish()
        RE = nn.ReLU()
        # [kernel, exp size, in_channels, out_channels, SEBlock, activation, stride]
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

# Example usage:
if __name__ == "__main__":
    model = MobileNetV3(config_name="large", inchannels=3, classes=1000)
    # Print model summary if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.randn(1, 1, 112, 112).to(device)

    summary(model, (1,112,112),device=str(device))
  
    out = model(x)
    print("Output shape:", out.shape)


    # def config(self, name):
    #     HE, RE = nn.Hardswish(), nn.ReLU()
    #     # [kernel, exp size, in_channels, out_channels, SEBlock(SE), activation function(NL), stride(s)] 
    #     large = [
    #             [3, 16, 16, 16, False, RE, 1],
    #             [3, 64, 16, 24, False, RE, 2],
    #             [3, 72, 24, 24, False, RE, 1],
    #             [5, 72, 24, 40, True, RE, 2],
    #             [5, 120, 40, 40, True, RE, 1],
    #             [5, 120, 40, 40, True, RE, 1],
    #             [3, 240, 40, 80, False, HE, 2],
    #             [3, 200, 80, 80, False, HE, 1],
    #             [3, 184, 80, 80, False, HE, 1],
    #             [3, 184, 80, 80, False, HE, 1],
    #             [3, 480, 80, 112, True, HE, 1],
    #             [3, 672, 112, 112, True, HE, 1],
    #             [5, 672, 112, 160, True, HE, 2],
    #             [5, 960, 160, 160, True, HE, 1],
    #             [5, 960, 160, 160, True, HE, 1]
    #     ]

    #     small = [
    #             [3, 16, 16, 16, True, RE, 2],
    #             [3, 72, 16, 24, False, RE, 2],
    #             [3, 88, 24, 24, False, RE, 1],
    #             [5, 96, 24, 40, True, HE, 2],
    #             [5, 240, 40, 40, True, HE, 1],
    #             [5, 240, 40, 40, True, HE, 1],
    #             [5, 120, 40, 48, True, HE, 1],
    #             [5, 144, 48, 48, True, HE, 1],
    #             [5, 288, 48, 96, True, HE, 2],
    #             [5, 576, 96, 96, True, HE, 1],
    #             [5, 576, 96, 96, True, HE, 1]
    #     ]

    #     if name == "large": return large
    #     if name == "small": return small

# # testing
# import torch
# import torch.nn as nn
# from torchsummary import summary
# import matplotlib.pyplot as plt

# # Assuming your MobileNetV3UNet class is defined

# def test_mobilenetv3_unet():
#     # Initialize the architecture
#     print("Initializing MobileNetV3UNet...")
#     model = MobileNetV3UNet(config_name="large", inchannels=3, outchannels=1)
    
#     # Print the architecture
#     print("\nModel Architecture:")
#     print(model)
    
#     # Input data
#     input_shape = (1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 spatial size
#     dummy_input = torch.randn(input_shape)  # Create dummy input
#     print(f"\nInput Shape: {dummy_input.shape}")

#     # Run a forward pass
#     print("\nRunning Forward Pass...")
#     output = model(dummy_input)
#     print(f"Output Shape: {output.shape}")
    
#     # Assert output shape
#     expected_output_shape = (1, 1, 224, 224)  # Should match input spatial dimensions
#     assert output.shape == expected_output_shape, f"Output shape mismatch! Expected {expected_output_shape}, got {output.shape}"

#     # Use torchsummary to display architecture details
#     print("\nSummary:")
#     summary(model, input_size=(3, 224, 224))

#     # Visualize input and output
#     visualize_data(dummy_input, output)

#     print("\nMobileNetV3UNet passed all tests!")


# def visualize_data(input_tensor, output_tensor):
#     """
#     Visualize the input and output tensors.
#     """
#     input_image = input_tensor[0].permute(1, 2, 0).detach().numpy()  # Convert to HWC for visualization
#     output_image = output_tensor[0, 0].detach().numpy()  # Extract first output channel
    
#     # Plot input and output
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Input Image")
#     plt.imshow(input_image)
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.title("Output Segmentation")
#     plt.imshow(output_image, cmap="gray")
#     plt.axis("off")

#     plt.show()


# if __name__ == "__main__":
#     # Test the MobileNetV3UNet architecture
#     test_mobilenetv3_unet()


#     conv = ConvBlock(
#         inchannels=3,
#         outchannels=16,
#         kernelsize=3,
#         stride=1,
#         actvn=nn.ReLU(),
#         bn=True,
#         bias=False
#     )
#     conv_input = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image
#     # forward pass
#     conv_output = conv(conv_input)
#     print(f"Architecture: {conv}\nOutput shape: {conv_output.shape}")

#     se = SEblock(inchannels=64)
#     se_input = torch.randn(1,64,32,32) # Batch size 1, 64 channels, 32x32 spatial size
#     se_output = se(se_input)
#     print(f"Architecture: {se}\nInput shape: {se_input.shape}\nOutput shape: {se_output.shape}")

#     bottleneck = Bottleneck(inchannels=32,outchannels=32,kernelsize=3,stride=1,exp_size=64,se=True,actvn=nn.ReLU())
#     blk_input = torch.randn(1, 32, 64, 64)  # Batch size 1, 32 channels, 64x64 spatial size
#     blk_output = bottleneck(blk_input)
#     print(f"Architecture: {bottleneck}\nInput shape: {blk_input.shape}\nOutput shape: {blk_output.shape}")

#     name = "large"
#     rho = 1
#     res = int(rho * 224)

#     mbnet = MobileNetV3(name)
#     mbnet_input = torch.rand(1, 3, res, res)
#     mbnet_output = mbnet(mbnet_input)
#     print(f'MobileNetV3 Architecture: {mbnet}\nInput shape: {mbnet_input.shape}\nOutput shape: {mbnet_output.shape}')


