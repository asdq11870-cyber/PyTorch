import torch
from torch import nn
from models.StableDiffusionV1.CLIPTransformer import CLIPTransformer
from transformers import CLIPTokenizer
import yaml

with open("Parameters.yaml","r") as f:
    config = yaml.safe_load(f)["UNET"]

class TimestepEmbedding(nn.Module):
    """
    Converting the timestep to a 1D vector quantity for each image given in a batch

    A 1D vector of zeros is created with a batch number of rows and embedded dimension
    for the number of columns. The code is made device agnostic and converting to an 
    essential datatype. Then apply the frequency equation 2i: sin(t/10000^2j/embed_dim)
    and 2i+1: cos(t/10000^2j/embed_dim). This tensor is then pass through a mlp projection
    or learnable projection to prepare it for the injection into the ResNet blocks

    Args:
        embed_dim: The embedded dimension
        timestep: The scalar we want to convert into a vector
        expanded_channels: The channels used in the mlp projection to add features and non-linearity

    Returns:
        a: The 1D timestep vector for each image

    Example:
        a: (batch_size, embed_dim)
        After for loop: (batch_size, embed_dim)
        After projection: (batch_size, 1280)
    """
    def __init__(self, embed_dim:int, expanded_channels:int):
        super().__init__()
        self.embed_dim = embed_dim
        assert self.embed_dim % 2 == 0, "Embedded Dimension should be even!"
        self.time_mlp_projection = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=expanded_channels),
            nn.SiLU(),
            nn.Linear(in_features=expanded_channels,out_features=expanded_channels)
        )

    def forward(self, timestep:torch.Tensor) -> torch.Tensor:
        a = torch.zeros(timestep.shape[0], self.embed_dim).to(device=timestep.device, dtype="float32")
        j = 0
        for i in range(0,self.embed_dim,2):
            a[:,i] = torch.sin(timestep/torch.pow(10000,(2*j)/self.embed_dim))
            a[:,i+1] = torch.cos(timestep/torch.pow(10000,(2*j)/self.embed_dim))    
            j += 1
        a = self.time_mlp_projection(a)
        return a

class ResNet(nn.Module):
    """
    Adding the timestep vector to a latent representation of the image

    A projection and several convolutions, SiLU activations, and group 
    normalisations are initalised in the constructor. The latent image
    passes through the first group normalisation and activation, then
    pass through a convolution that refines features. The timestep 
    vector is projected from the expanded_channels in the mlp step to 
    the out_channels corresponding to the channels in the U-Net. The 
    vector is then reshaped to add two new dimensions and concatenated 
    to the latent image. The latent images then pass through a second
    group normalisation and activation, then pass through a convolution 
    that does further refinement after timestep injection. The final 
    output of the ResNet block is also added to the initial latent
    image tensors for making the information learn from the original.

    Args:
        in_channels: The input channels of the ResNet block
        out_channels: The output channels of the ResNet block
        expanded_channels: The mlp channels
        num_groups: The amount of groups that pass through the group norm
        x: The original latent image tensor
        timestep_vector: The 2D timestep vector

    Returns:
        x: The latent image tensor which has timestep embeddings and original latent information

    Example:
        x: (batch_size, in_channels, height, width)
        after groupnorm1: (batch_size, in_channels, height, width)
        after silu1: (batch_size, in_channels, height, width)
        after conv1: (batch_size, out_channels, height, width)
        after timestep embedding: (batch_size, out_channels, height, width)
        after groupnorm2: (batch_size, out_channels, height, width)
        after silu2: (batch_size, out_channels, height, width)
        after conv2: (batch_size, out_channels, height, width)
        after residual addition: (batch_size, out_channels, height, width)
    """
    def __init__(self, in_channels:int, out_channels:int, expanded_channels:int, num_groups:int):
        super().__init__()
        self.timestep_projection = nn.Linear(
            in_features=expanded_channels, out_features=out_channels
        )
        self.residual_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(1,1), stride=1, padding=0
        )
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
        self.silu1 = nn.SiLU()
        self.silu2 = nn.SiLU()
        self.groupnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.groupnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.dropout = nn.Dropout2d(p=0.5, inplace=True)

    def forward(self, x:torch.Tensor, timestep_vector:torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.groupnorm1(x)
        x = self.silu1(x)
        x = self.conv1(x)
        t = timestep_vector
        t = self.timestep_projection(t)
        t = t.reshape(t.shape[0],t.shape[1],1,1)
        x = x + t
        x = self.groupnorm2(x)
        x = self.silu2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x + self.residual_conv(residual)
        return x
        
class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.downsampling_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=output_channels,
            kernel_size=(3,3), stride=2, padding=1
        )

    def forward(self, x:torch.Tensor):
        x = self.downsampling_conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, input_channels:int, output_channels:int, scale_factor:int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.upsampling_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=output_channels,
            kernel_size=(3,3), stride=1, padding=0
        )

    def forward(self, x:torch.Tensor):
        x = self.upsample(x)
        x = self.upsampling_conv(x)
        return x

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim:int, heads:int):
        super().__init__()
        assert embed_dim % heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // heads
        self.qkv_projection = nn.Linear(in_features=embed_dim, out_features=embed_dim * 3)
        self.projection = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.embed_dim = embed_dim
        self.heads = heads

    def forward(self, x:torch.Tensor):
        batch, tokens, _ = x.shape
        qkv = self.qkv_projection(x)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.reshape(batch, tokens, self.heads, self.head_dim).permute(0,2,1,3)
        key = key.reshape(batch, tokens, self.heads, self.head_dim).permute(0,2,1,3)
        value = value.reshape(batch, tokens, self.heads, self.head_dim).permute(0,2,1,3)

        attn_scores = query @ key.transpose(-1,-2)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1) @ value
        return self.projection(attn_scores)

class MultiHeadCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim:int, heads:int, channels:int, max_seq_len:int):
        super().__init__()
        assert channels % heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = channels // heads
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.q_projection = nn.Linear(in_features=channels, out_features=channels)
        self.k_projection = nn.Linear(in_features=embed_dim, out_features=channels)
        self.v_projection = nn.Linear(in_features=embed_dim, out_features=channels)
        self.out_projection = nn.Conv2d(
            in_channels=channels, out_channels=channels,
            kernel_size=(1,1), stride=1, padding=0
        )

    def forward(self, x:torch.Tensor, context:torch.Tensor):
        batch, channels, height, width = x.shape
        query = x.flatten(start_dim=2, end_dim=3).reshape(batch, self.heads, self.head_dim, height*width).permute(0,1,3,2)
        key = self.k_projection(context)
        key = key.reshape(batch, self.max_seq_len, self.heads, self.head_dim).permute(0,2,1,3)
        value = self.v_projection(context)
        value = value.reshape(batch, self.max_seq_len, self.heads, self.head_dim).permute(0,2,1,3)

        attn_scores = query @ key.transpose(-1,-2)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1) @ value
        attn_output = attn_scores.transpose(2,3).reshape(batch, channels, height, width)
        return self.out_projection(attn_output)
        
class DownBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class MidBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class UpBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    def forward(self, x:torch.Tensor):
        pass

    def load_pretrained(self):
        pass