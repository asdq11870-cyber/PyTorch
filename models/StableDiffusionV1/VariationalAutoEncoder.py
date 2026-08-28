import torch
from torch import nn

class ResNet(nn.Module):
    def __init__(self, embed_dim, num_groups, in_channels, out_channels):
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
            in_channels=in_channels, out_channels=out_channels
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels
        )
        self.residual_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads):
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
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class LatentDistribution(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass


