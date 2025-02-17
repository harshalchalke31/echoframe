import torch.nn as nn
import torch
import torch.nn.functional as F


class Conv_block(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,kernel_size:int=3,padding:int=1):
        super(Conv_block,self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)

class Encoder(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,kernel_size:int=3,padding:int=1):
        super(Encoder,self).__init__()

        self.conv = Conv_block(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x:torch.Tensor)->tuple:
        x = self.conv(x)
        pooled_x = self.pool(x)

        return x, pooled_x


class Decoder(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,kernel_size:int=2,stride:int=2):
        super(Decoder,self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride)
        self.conv = Conv_block(in_channels=out_channels*2,out_channels=out_channels)

    def forward(self,x:torch.Tensor,skip_features:torch.Tensor)->torch.Tensor:
        x = self.upconv(x)
        x = torch.cat((x,skip_features),dim=1)
        x = self.conv(x)

        return x
        

class UNet(nn.Module):
    def __init__(self,input_channels:int=3,output_channels:int=1):
        super(UNet,self).__init__()

        self.encoder1 = Encoder(in_channels=input_channels,out_channels=64)
        self.encoder2 = Encoder(in_channels=64,out_channels=128)
        self.encoder3 = Encoder(in_channels=128,out_channels=256)
        self.encoder4 = Encoder(in_channels=256,out_channels=512)

        self.bottleneck = Conv_block(in_channels=512,out_channels=1024)
        self.dropout = nn.Dropout2d(p=0.5)

        self.decoder1 = Decoder(in_channels=1024,out_channels=512)
        self.decoder2 = Decoder(in_channels=512,out_channels=256)
        self.decoder3 = Decoder(in_channels=256,out_channels=128)
        self.decoder4 = Decoder(in_channels=128,out_channels=64)

        self.final_layer = nn.Conv2d(in_channels=64, out_channels=output_channels,kernel_size=1)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        skip1,pooled1 = self.encoder1(x)
        skip2,pooled2 = self.encoder2(pooled1)
        skip3,pooled3 = self.encoder3(pooled2)
        skip4,pooled4 = self.encoder4(pooled3)

        bottleneck = self.bottleneck(pooled4)
        bottleneck = self.dropout(bottleneck)

        decoder_output1 = self.decoder1(bottleneck,skip4)
        decoder_output2 = self.decoder2(decoder_output1,skip3)
        decoder_output3 = self.decoder3(decoder_output2,skip2)
        decoder_output4 = self.decoder4(decoder_output3,skip1)

        output = self.final_layer(decoder_output4)

        return output

if __name__=='__main__':
    model = UNet()
    test_input = torch.randn((1,3,256,256))
    test_output = model(test_input)

    print(model)
    print(test_output)
    print(test_output.shape)

