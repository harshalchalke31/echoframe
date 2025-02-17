import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class PretrainedTransUNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        img_size: int = 224,
        patch_size: int = 16,
        encoder_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        decoder_channels: list = [256, 128, 64, 32]
    ):
        """
        Args:
            num_classes: Number of segmentation classes (or 1 for binary segmentation).
            img_size: Input image resolution (assumed square).
            patch_size: Patch size used in the transformer backbone.
            encoder_name: Name of the pretrained transformer backbone from timm.
            pretrained: Whether to load pretrained weights.
            decoder_channels: List defining the number of channels for each decoder block.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Load a pretrained ViT model from timm with global_pool disabled
        # This ensures the output is (B, num_tokens, embed_dim)
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''  # disable pooling to return all patch tokens
        )
        self.encoder_embed_dim = self.encoder.embed_dim  # e.g., 768 for vit_base_patch16_224

        # Determine if the model outputs a class token (commonly the first token)
        self.use_cls_token = hasattr(self.encoder, "cls_token")
        
        # Calculate the feature map size (e.g., 224/16 = 14)
        self.enc_feat_size = img_size // patch_size
        
        # Decoder: progressive upsampling using ConvTranspose2d layers
        self.decoder_conv1 = nn.ConvTranspose2d(
            in_channels=self.encoder_embed_dim, out_channels=decoder_channels[0],
            kernel_size=2, stride=2
        )  # Upsample: enc_feat_size x enc_feat_size -> (enc_feat_size*2 x enc_feat_size*2)
        
        self.decoder_conv2 = nn.ConvTranspose2d(
            in_channels=decoder_channels[0], out_channels=decoder_channels[1],
            kernel_size=2, stride=2
        )
        
        self.decoder_conv3 = nn.ConvTranspose2d(
            in_channels=decoder_channels[1], out_channels=decoder_channels[2],
            kernel_size=2, stride=2
        )
        
        self.decoder_conv4 = nn.ConvTranspose2d(
            in_channels=decoder_channels[2], out_channels=decoder_channels[3],
            kernel_size=2, stride=2
        )
        
        # Final convolution to produce the segmentation mask
        self.final_conv = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)
        
        self._init_weights()

    def _init_weights(self):
        # Initialize decoder weights
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        B = x.size(0)
        # Pass through the pretrained transformer encoder.
        # With global_pool disabled, expected shape: (B, num_tokens, embed_dim)
        enc_tokens = self.encoder(x)

        # If the output is 2D, raise an error to help with debugging.
        if enc_tokens.dim() == 2:
            raise ValueError("Encoder output is 2D. Make sure to disable global pooling by setting global_pool=''.")

        # Remove the classification token if present (assumed to be the first token)
        if self.use_cls_token:
            enc_tokens = enc_tokens[:, 1:, :]  # Now shape: (B, num_patches, embed_dim)

        # Reshape the tokens to a 2D spatial feature map
        enc_feat = enc_tokens.transpose(1, 2)  # (B, embed_dim, num_patches)
        enc_feat = enc_feat.contiguous().view(B, self.encoder_embed_dim,
                                              self.enc_feat_size, self.enc_feat_size)

        # Decoder: progressively upsample to original resolution
        x = F.relu(self.decoder_conv1(enc_feat))  # e.g., 14x14 -> 28x28
        x = F.relu(self.decoder_conv2(x))           # 28x28 -> 56x56
        x = F.relu(self.decoder_conv3(x))           # 56x56 -> 112x112
        x = F.relu(self.decoder_conv4(x))           # 112x112 -> 224x224
        
        out = self.final_conv(x)  # Final segmentation map
        return out

# Example usage:
if __name__ == "__main__":
    model = PretrainedTransUNet(num_classes=1, img_size=224, patch_size=16)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (2, 1, 224, 224)
