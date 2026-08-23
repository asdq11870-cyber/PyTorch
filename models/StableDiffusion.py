import torch
from torch import nn

# 1. TimestepEmbedding       # nn.Module
# 2. ResNet                   # nn.Module
# 3. Downsample               # nn.Module
# 4. Upsample                 # nn.Module
# 5. SelfAttentionBlock       # nn.Module
# 6. CLIPTransformer          # nn.Module
# 7. CrossAttentionBlock      # nn.Module
# 8. VAE                      # nn.Module
# 9. UNet                     # nn.Module
# 10. Noise/Sampling Scheduler # does NOT need to inherit from nn.Module
# 11. StableDiffusion         # nn.Module

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
    pass through a convolution that does god know what. The timestep 
    vector is projected from the expanded_channels in the mlp step to 
    the out_channels corresponding to the channels in the U-Net. The 
    vector is then reshaped to add two new dimensions and concatenated 
    to the latent image. The latent images then pass through a second
    group normalisation and activation, then pass through a convolution 
    that does god know what. The final output of the ResNet block is 
    also concatenated to the initial latent image tensors for god knows
    what.

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
        x = self.conv2(x)
        x = x + self.residual_conv(residual)
        return x
        

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

class SelfAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class CLIPTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class CrossAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class DiffusionSchedular():
    def __init__(self):
        pass

class StableDiffusion(nn.Module):
    def __init__(self):
            super().__init__()
    
    def forward(self, x:torch.Tensor):
        pass


