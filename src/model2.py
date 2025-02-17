import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.entity import UNetRTrainerConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim,key_dim,num_heads,dropout_rate=0.0):
        super().__init__()
        self.num_heads=num_heads
        self.key_dim = key_dim #dimension of each head
        self.embed_dim = embed_dim
        self.total_key_dim = self.num_heads*self.key_dim

        # linear projections for query key values
        self.W_q = nn.Linear(in_features=embed_dim,out_features=self.total_key_dim)
        self.W_k = nn.Linear(in_features=embed_dim,out_features=self.total_key_dim)
        self.W_v = nn.Linear(in_features=embed_dim,out_features=self.total_key_dim)

        # final projection back to embed dimension
        self.out_proj = nn.Linear(in_features=self.total_key_dim,out_features=embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,query,key,value):
        # q, k, v are expected to be in shape (B,N,embed_dim)
        B,N, _ = query.shape

        # project inputs to multi heads
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        # reshape and transpose = (B, num_heads, N, key_dim)
        q = q.view(B, self.num_heads, N, self.key_dim).transpose(1,2)
        k = k.view(B, self.num_heads, N, self.key_dim).transpose(1,2)
        v = v.view(B, self.num_heads, N, self.key_dim).transpose(1,2)

        # compute scaled dot product attention
        attn_scores = torch.matmul(q, k.transpose(-2,-1)) / (self.key_dim**0.5)
        attn_weights = torch.softmax(attn_scores,dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights,v) # (B, num_heads, N, key_dim)

        # combine heads
        attn_output = attn_output.transpose(1,2).contiguous().view(B, N, self.total_key_dim) #(B, N, key_dim)
        output = self.out_proj(attn_output)  # (B, N, embed_dim)
        return output

class Convblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size//2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout_rate):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(embed_dim=hidden_dim,num_heads=num_heads,key_dim=hidden_dim,dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim,mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim,hidden_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self,x):
        skip1 = x
        x = self.norm(x)
        attn_out = self.attn(x,x,x)
        x = skip1 + attn_out

        skip2 = x
        x = self.norm(x)
        mlp_out = self.mlp(x)
        x = skip2 + mlp_out

        return x
    


class UNetR2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_patches = (config.image_size ** 2) // (config.patch_size ** 2)

        # Patch Embedding via convolution  
        self.patch_embed = nn.Conv2d(in_channels=config.num_channels, out_channels=config.hidden_dim,
                                     kernel_size=config.patch_size, stride=config.patch_size)
        # Positional embedding  
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.hidden_dim))

        # Transformer Encoder Layers  
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(hidden_dim=config.hidden_dim,
                               num_heads=config.num_heads,
                               mlp_dim=config.mlp_dim,
                               dropout_rate=config.dropout_rate)
            for _ in range(config.num_layers)
        ])
        self.skip_indices = config.skip_indices  # e.g. {2, 4, 6} for a 6-layer encoder

        # (Optional) Spatial adjustment for skip features if needed  
        self.total_upscale_factor = int(math.log2(config.patch_size)) if config.patch_size > 0 else 0
        self.upscale = self.total_upscale_factor - len(self.skip_indices)
        if self.upscale > 2:
            self.skip_deconv = nn.ModuleList([
                nn.ConvTranspose2d(in_channels=config.hidden_dim, out_channels=config.hidden_dim,
                                   kernel_size=2, stride=2 ** self.upscale)
                for _ in range(len(self.skip_indices))
            ])

        # --- New 4-Block Decoder (Standard UNet Style) ---  #new
        # Note: All transformer skips are at the same resolution (H',W'), so we upsample them with different strides
        # to simulate a multi-scale encoder.
        #
        # Let’s assume image_size=256 and patch_size=16, so H'=16.
        # We define:
        #   Block 1: Use deepest skip (from layer 6) upsampled by factor 2 => 16->32.
        #   Block 2: Use middle skip (from layer 4) upsampled by factor 4 => 16->64.
        #   Block 3: Use shallowest skip (from layer 2) upsampled by factor 8 => 16->128.
        #   Block 4: Fuse with original input (256x256) after processing.
        #
        # Block 1:
        self.dec1_deconv = nn.ConvTranspose2d(in_channels=config.hidden_dim, out_channels=64, kernel_size=2, stride=2)  # 16->32  
        # Block 2:
        self.dec2_deconv = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)  # 32->64  
        self.dec2_skip = nn.ConvTranspose2d(in_channels=config.hidden_dim, out_channels=32, kernel_size=4, stride=4)  # 16->64  #new
        self.dec2_conv = nn.Sequential(
            Convblock(64, 32, 3),
            Convblock(32, 32, 3)
        )
        # Block 3:
        self.dec3_deconv = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)  # 64->128  #changed
        self.dec3_skip = nn.ConvTranspose2d(in_channels=config.hidden_dim, out_channels=16, kernel_size=8, stride=8)  # 16->128  #new
        self.dec3_conv = nn.Sequential(
            Convblock(32, 16, 3),
            Convblock(16, 16, 3)
        )
        # Block 4:
        self.dec4_deconv = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2)  # 128->256  #changed
        self.dec4_skip = Convblock(config.num_channels, 16, 3)  # process original input  #new
        self.dec4_conv = nn.Sequential(
            Convblock(32, 16, 3),
            Convblock(16, 16, 3)
        )

        # Final segmentation mask  
        self.out_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        z0 = x  # original input for block 4

        # Patch Embedding  
        x = self.patch_embed(x)  # shape: (B, hidden_dim, H', W')
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # shape: (B, num_patches, hidden_dim)
        x = x + self.pos_embed

        # Transformer Encoder with Skip Connections  
        skip_connections = []
        for i, layer in enumerate(self.transformer_layers, start=1):
            x = layer(x)
            if i in self.skip_indices:
                skip_connections.append(x)
        if len(skip_connections) != len(self.skip_indices):
            raise ValueError(f"Expected {len(self.skip_indices)} skip connections, got {len(skip_connections)}")
        # Reverse order so that deepest skip comes first  
        # Assuming skip_indices were collected in order [z2, z4, z6],
        # we want: z6 (deepest), z4 (middle), z2 (shallowest)
        z6, z4, z2 = skip_connections[::-1]  # #new

        # Reshape each skip to (B, hidden_dim, H', W')
        z6 = z6.transpose(1, 2).view(B, self.config.hidden_dim, H, W)
        z4 = z4.transpose(1, 2).view(B, self.config.hidden_dim, H, W)
        z2 = z2.transpose(1, 2).view(B, self.config.hidden_dim, H, W)

        # (Optional) Adjust spatial resolution if needed  
        if self.upscale >= 2:
            z6 = self.skip_deconv[0](z6)
            z4 = self.skip_deconv[1](z4)
            z2 = self.skip_deconv[2](z2)
        elif self.upscale < 0:
            pool_kernel = 2 ** abs(self.upscale)
            z6 = F.max_pool2d(z6, kernel_size=pool_kernel)
            z4 = F.max_pool2d(z4, kernel_size=pool_kernel)
            z2 = F.max_pool2d(z2, kernel_size=pool_kernel)

        # --- Decoder Block 1: Use z6 only ---  
        # (No fusion; simply upsample the deepest skip to form the first decoder output.)
        x1 = self.dec1_deconv(z6)  # x1 shape: (B, 64, 32, 32)

        # --- Decoder Block 2: Fuse x1 with skip from layer 4 (z4) ---  
        x2_main = self.dec2_deconv(x1)  # upsample x1: (B, 32, 64, 64)
        x2_skip = self.dec2_skip(z4)     # upsample z4 from 16->64: (B, 32, 64, 64)
        x2 = torch.cat([x2_main, x2_skip], dim=1)  # (B, 64, 64, 64)
        x2 = self.dec2_conv(x2)  # reduce channels to 32

        # --- Decoder Block 3: Fuse x2 with skip from layer 2 (z2) ---  
        x3_main = self.dec3_deconv(x2)  # upsample: (B, 16, 128, 128)
        x3_skip = self.dec3_skip(z2)      # upsample z2 from 16->128: (B, 16, 128, 128)
        x3 = torch.cat([x3_main, x3_skip], dim=1)  # (B, 32, 128, 128)
        x3 = self.dec3_conv(x3)  # reduce channels to 16

        # --- Decoder Block 4: Fuse x3 with original input (z0) ---  
        x4_main = self.dec4_deconv(x3)  # upsample: (B, 16, 256, 256)
        # Downsample original input to match spatial size if needed  
        z0_down = F.interpolate(z0, size=x4_main.shape[2:], mode='bilinear', align_corners=False)  # #changed
        x4_skip = self.dec4_skip(z0_down)  # process original input: (B, 16, 256, 256)
        x4 = torch.cat([x4_main, x4_skip], dim=1)  # (B, 32, 256, 256)
        x4 = self.dec4_conv(x4)  # reduce channels to 16

        out = self.out_conv(x4)
        # out = torch.sigmoid(out)
        return out