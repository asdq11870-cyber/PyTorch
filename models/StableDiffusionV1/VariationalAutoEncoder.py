import torch
from torch import nn

class ResNet(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, num_groups:int=32):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels
        )
        self.groupnorm2 = nn.GroupNorm(
            num_groups=num_groups, num_channels=out_channels
        )
        self.silu1 = nn.SiLU()
        self.silu2 = nn.SiLU()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
        self.residual_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(1,1), padding=0, stride=1
        )

    def forward(self, x:torch.Tensor):
        residual = x
        x = self.groupnorm1(x)
        x = self.silu1(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = self.silu2(x)
        x = self.conv2(x)
        x = x + self.residual_conv(residual)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim:int, heads:int):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        assert self.embed_dim % self.heads == 0, "Embedding dimension should be divisible by the number of heads"
        self.head_dim = self.embed_dim // self.heads
        self.qkv = nn.Linear(in_features=embed_dim, out_features=embed_dim*3)
        self.projection = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(self, x:torch.Tensor):
        batch, tokens, _ = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.reshape(batch, tokens, self.heads, self.head_dim).transpose(1,2)
        key = key.reshape(batch, tokens, self.heads, self.head_dim).transpose(1,2)
        value = value.reshape(batch, tokens, self.heads, self.head_dim).transpose(1,2)

        attn_scores = (query @ key.transpose(-1,-2)) / (self.head_dim ** 0.5)
        attn_scores = torch.softmax(attn_scores) @ value
        attn_scores = attn_scores.transpose(1,2).reshape(batch, tokens, self.embed_dim)
        return self.projection(attn_scores)

class Downsample(nn.Module):
    def __init__(self, input_channels:int, output_channels:int):
        super().__init__()
        self.downsampling_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=(3,3),
            stride=2,
            padding=1
        )
    def forward(self, x:torch.Tensor):
        x = self.downsampling_conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, input_channels:int, output_channels:int):
        super().__init__()
        self.downsampling_conv = nn.ConvTranspose2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=(3,3),
            stride=2,
            padding=1
        )
    def forward(self, x:torch.Tensor):
        x = self.upsampling_conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim:int=768, heads:int=12, input_channels:int=128, output_channels:int=8, rgb_channels:int=3):
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels=rgb_channels,
            out_channels=input_channels,
            kernel_size=(3,3),
            stride=1,
            padding=1
        )
        self.down_block1 = nn.ModuleList(
            [
                ResNet(in_channels=input_channels, out_channels=input_channels),
                ResNet(in_channels=input_channels, out_channels=input_channels),
                Downsample(channels=input_channels, iteration=1)
            ]
        )
        self.down_block2 = nn.ModuleList(
            [
                ResNet(in_channels=input_channels*2, out_channels=input_channels*2),
                ResNet(in_channels=input_channels*2, out_channels=input_channels*2),
                Downsample(channels=input_channels, iteration=2)
            ]
        )
        self.down_block3 = nn.ModuleList(
            [
                ResNet(in_channels=input_channels*4, out_channels=input_channels*4),
                ResNet(in_channels=input_channels*4, out_channels=input_channels*4),
                Downsample(channels=input_channels, iteration=3)
            ]
        )
        self.down_block4 = nn.ModuleList(
            [
                ResNet(in_channels=input_channels*4, out_channels=input_channels*4),
                ResNet(in_channels=input_channels*4, out_channels=input_channels*4),
                MultiHeadSelfAttention(embed_dim=embed_dim, heads=heads),
                MultiHeadSelfAttention(embed_dim=embed_dim, heads=heads)
            ]
        )
        self.middle_block = nn.ModuleList(
            [
                ResNet(in_channels=input_channels*4, out_channels=input_channels*4),
                MultiHeadSelfAttention(embed_dim=embed_dim, heads=heads),
                ResNet(in_channels=input_channels*4, out_channels=input_channels*4)
            ]
        )
        self.groupnorm = nn.GroupNorm(num_groups=embed_dim, num_channels=input_channels*4)
        self.silu = nn.SiLU(inplace=True)
        self.output_conv = nn.Conv2d(
            in_channels=input_channels*4, out_channels=output_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
        self.latent_distribution = LatentDistribution()
    def forward(self, x:torch.Tensor):
        x = self.input_conv(x)
        for block1 in self.down_block1:
            x = block1(x)
        for block2 in self.down_block2:
            x = block2(x)
        for block3 in self.down_block3:
            x = block3(x)
        for block4 in self.down_block4:
            x = block4(x)
        for block5 in self.middle_block:
            x = block5(x)
        x = self.groupnorm(x)
        x = self.silu(x)
        x = self.output_conv(x)
        z = self.latent_distribution(x)
        return z


class Decoder(nn.Module):
    def __init__(self, embed_dim:int=768, heads:int=64, rgb_channels:int=3, latent_channels:int=4, input_channels:int=512):
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels=latent_channels, out_channels=input_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
        self.middle_block = nn.ModuleList(
            [
                ResNet(in_channels=input_channels, out_channels=input_channels),
                MultiHeadSelfAttention(embed_dim=embed_dim, heads=heads),
                ResNet(in_channels=input_channels, out_channels=input_channels)
            ]
        )
        self.up_block1 = nn.ModuleList(
            [
                ResNet(in_channels=input_channels, out_channels=input_channels),
                ResNet(in_channels=input_channels, out_channels=input_channels),
                ResNet(in_channels=input_channels, out_channels=input_channels),
                Upsample(channels=input_channels, iteration=1)
            ]
        )
        self.up_block2 = nn.ModuleList(
            [
                ResNet(in_channels=input_channels/2, out_channels=input_channels/2),
                ResNet(in_channels=input_channels/2, out_channels=input_channels/2),
                ResNet(in_channels=input_channels/2, out_channels=input_channels/2),
                Upsample(channels=input_channels, iteration=2)
            ]
        )
        self.up_block3 = nn.ModuleList(
            [
                ResNet(in_channels=input_channels/4, out_channels=input_channels/4),
                ResNet(in_channels=input_channels/4, out_channels=input_channels/4),
                ResNet(in_channels=input_channels/4, out_channels=input_channels/4),
                Upsample(channels=input_channels, iteration=3)
            ]
        )
        self.up_block4 = nn.ModuleList(
            [
                ResNet(in_channels=input_channels/4, out_channels=input_channels/4),
                ResNet(in_channels=input_channels/4, out_channels=input_channels/4),
                ResNet(in_channels=input_channels/4, out_channels=input_channels/4)
            ]
        )
        self.groupnorm = nn.GroupNorm(num_groups=embed_dim, num_channels=input_channels/4)
        self.silu = nn.SiLU(inplace=True)
        self.output_conv = nn.Conv2d(
            in_channels=input_channels/4, out_channels=rgb_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
    def forward(self, x:torch.Tensor):
        x = self.input_conv(x)
        for block1 in self.middle_block:
            x = block1(x)
        for block2 in self.up_block1:
            x = block2(x)
        for block3 in self.up_block2:
            x = block3(x)
        for block4 in self.up_block3:
            x = block4(x)
        for block5 in self.up_block4:
            x = block5(x)
        
        x = self.groupnorm(x)
        x = self.silu(x)
        x = self.output_conv(x)
        return x

class LatentDistribution(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        mean, log_variance = x.chunk(chunks=2, dim=1)
        std = torch.exp(log_variance/2)
        z = (x * std) + mean
        return z

class VAE(nn.Module):
    def __init__(self, encode:bool=False, decode:bool=False):
        super().__init__()
        self.encode = encode
        self.decode = decode
        assert encode == decode, "Cannot encode and decode at once!"
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x:torch.Tensor):
        if self.encode: x = self.encoder(x)
        elif self.decode: x = self.decoder(x)
        return x



