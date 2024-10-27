import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se_block(x)
        return x * scale

class MobileUNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileUNetEncoder, self).__init__()
        mobilenet = mobilenet_v3_small(pretrained=pretrained)
        self.features = mobilenet.features
        self.se_blocks = nn.ModuleList([
            SqueezeExcitation(16),
            SqueezeExcitation(24),
            SqueezeExcitation(40),
            SqueezeExcitation(48),
        ])

    def forward(self, x):
        skip_connections = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in {1, 3, 6, 10}:  # layers with skip connections
                se_block = self.se_blocks.pop(0)
                x = se_block(x)
                skip_connections.append(x)
        return x, skip_connections

class UNetDecoder(nn.Module):
    def __init__(self, num_classes, dropout_p=0.5):
        super(UNetDecoder, self).__init__()
        self.upsample1 = nn.ConvTranspose2d(48, 40, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(40, 24, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(80, 40, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(48, 24, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(dropout_p)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x, skip_connections):
        x = self.upsample1(x)
        x = torch.cat([x, skip_connections[3]], dim=1)
        x = F.relu(self.conv1(x))

        x = self.dropout(x)
        x = self.upsample2(x)
        x = torch.cat([x, skip_connections[2]], dim=1)
        x = F.relu(self.conv2(x))

        x = self.dropout(x)
        x = self.upsample3(x)
        x = torch.cat([x, skip_connections[1]], dim=1)
        x = F.relu(self.conv3(x))

        x = self.dropout(x)
        x = self.upsample4(x)
        x = torch.cat([x, skip_connections[0]], dim=1)
        x = F.relu(self.conv4(x))

        x = self.final_conv(x)
        return x

class EchoSegmentationModel(nn.Module):
    def __init__(self, num_classes=1, dropout_p=0.5):
        super(EchoSegmentationModel, self).__init__()
        # Preliminary layer to transform 3-channel input to 48-channel
        self.initial_conv = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        self.encoder = MobileUNetEncoder()
        self.decoder = UNetDecoder(num_classes, dropout_p=dropout_p)

    def forward(self, x):
        x = self.initial_conv(x)  # Transform to 48 channels
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        return x

# Instantiate model for testing
if __name__ == "__main__":
    model = EchoSegmentationModel(num_classes=1)
    x = torch.randn((1, 3, 224, 224))  # Example input tensor
    y = model(x)
    print("Output shape:", y.shape)
