import torch
from torch import nn
from models.StableDiffusionV1.CLIPTransformer import CLIPTransformer
from transformers import CLIPTokenizer
from diffusers import UNet2DConditionModel
import yaml

with open("Parameters.yaml","r") as f:
    unet_config = yaml.safe_load(f)["UNET"]
    clip_config = yaml.safe_load(f)["CLIP"]

class TimestepEmbedding(nn.Module):
    """
    Converting the timestep to a 1D vector quantity for each image given in a batch

    A 1D vector of zeros is created with a batch number of rows and embedded dimension
    for the number of columns. The code is made device agnostic and converting to an 
    essential datatype. Then apply the frequency equation 2i: sin(t/10000^2j/temb_channels)
    and 2i+1: cos(t/10000^2j/temb_channels). This tensor is then pass through a mlp projection
    or learnable projection to prepare it for the injection into the ResNet blocks

    Args:
        temb_channels: The embedded dimension
        timestep: The scalar we want to convert into a vector
        expanded_channels: The channels used in the mlp projection to add features and non-linearity

    Returns:
        a: The 1D timestep vector for each image

    Example:
        a: (batch_size, temb_channels)
        After for loop: (batch_size, temb_channels)
        After projection: (batch_size, 1280)
    """
    def __init__(self, temb_channels:int, out_channels:int):
        super().__init__()
        self.temb_channels = temb_channels
        assert self.temb_channels % 2 == 0, "Embedded Dimension should be even!"
        self.time_mlp_projection = nn.Sequential(
            nn.Linear(in_features=temb_channels, out_features=out_channels),
            nn.SiLU(),
            nn.Linear(in_features=out_channels,out_features=out_channels)
        )

    def vectorize(self, timestep:torch.Tensor):
        a = torch.zeros(timestep.shape[0], self.temb_channels).to(device=timestep.device, dtype="float32")
        indices = torch.arange(0, self.temb_channels//2, device=a.device, dtype=a.dtype)
        a[:,0::2] = torch.sin(timestep/torch.pow(10000,(2*indices)/self.temb_channels))
        a[:,1::2] = torch.cos(timestep/torch.pow(10000,(2*indices)/self.temb_channels))
        return a

    def forward(self, timestep:torch.Tensor) -> torch.Tensor:
        timestep.unsqueeze(dim=1)
        temb = self.vectorize(timestep=timestep)
        temb = self.time_mlp_projection(temb)
        return temb

class ResNet(nn.Module):
    """
    Adding the timestep vector to a latent representation of the image

    A projection and several convolutions, SiLU activations, and group 
    normalisations are initalised in the constructor. The latent image
    passes through the first group normalisation and activation, then
    pass through a convolution that refines features. The timestep 
    vector is projected from the expanded_channels in the mlp step to 
    the out_channels corresponding to the channels in the U-Net. The 
    vector is then reshaped to add two new dimensions and added 
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
    def __init__(self, in_channels:int, out_channels:int, expanded_channels:int, num_groups:int, resnet_eps:float):
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.silu1 = nn.SiLU()
        self.silu2 = nn.SiLU()
        self.groupnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=resnet_eps)
        self.groupnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=resnet_eps)
        self.dropout = nn.Dropout2d(p=0.5)

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
        if self.in_channels != self.out_channels:
            x = x + self.residual_conv(residual)
        else:
            x = x + residual
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

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, temb_channels:int, heads:int):
        super().__init__()
        assert temb_channels % heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = temb_channels // heads
        self.qkv_projection = nn.Linear(in_features=temb_channels, out_features=temb_channels * 3)
        self.projection = nn.Linear(in_features=temb_channels, out_features=temb_channels)
        self.temb_channels = temb_channels
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

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, temb_channels:int, heads:int, channels:int, max_seq_len:int):
        super().__init__()
        assert channels % heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = channels // heads
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.q_projection = nn.Linear(in_features=channels, out_features=channels)
        self.k_projection = nn.Linear(in_features=temb_channels, out_features=channels)
        self.v_projection = nn.Linear(in_features=temb_channels, out_features=channels)
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

class FeedForward(nn.Module):
    def __init__(self, channels:int, ff_expansion:int):
        super().__init__()
        self.linear_projection_1 = nn.Linear(in_features=channels, out_features=channels*2*ff_expansion)
        self.gelu = nn.GELU()
        self.linear_projection_2 = nn.Linear(in_features=channels*ff_expansion, out_features=channels)

    def forward(self, x:torch.Tensor):
        x = self.linear_projection_1(x)
        path1, path2 = x.chunk(2, dim=-1)
        path1 = self.gelu(path1)
        output = path1 * path2
        return self.linear_projection_2(output)

class BasicTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class Transformer2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class CrossAttnDownBlock2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass
        
class DownBlock2D(nn.Module):
    def __init__(self, input_channels:int, output_channels:int):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class CrossAttnUpBlock2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class UpBlock2D(nn.Module):
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