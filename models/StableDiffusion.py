import torch
from torch import nn
from transformers import CLIPTokenizer

# CLIP CLASSES -----------------------------------------------------------------

class CLIPEmbedding(nn.Module):
    """
    Turns each token into a learnable 1D vector

    A tensor of tokens is passed through the forward pass. Another
    tensor is created to house the positional information of each
    token. Tensor x is passed through the token_embedding object
    to create the 1D vectors. Tensor a is passed through the 
    positional_embedding object to create a 1D vector of the position.
    The two tensors are then added and x is returned.

    Args:
        embed_dim: The embedding dimension
        vocab_size: Total number of tokens
        max_seq_len: Maximum number of consecutive tokens positional embedding can represent
        x: Tensor containing all tokens

    Returns:
        x: Tensor that contains all token and positional information

    Example:
        x: (batch_size, tokens)
        a: (tokens)
        After token embedding, x:(batch_size, tokens, embed_dim)
        After positional embedding, a:(tokens, embed_dim)
        After addition, x:(batch_size, tokens, embed_dim)
        Output, x:(batch_size, tokens, embed_dim)
    """
    def __init__(self, embed_dim, vocab_size, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(embedding_dim=embed_dim, num_embeddings=vocab_size)
        self.positional_embedding = nn.Embedding(embedding_dim=embed_dim, num_embeddings=max_seq_len)

    def forward(self, x:torch.Tensor):
        a = torch.arange(end=x.shape[1], device=x.device, dtype=torch.long)
        x = self.token_embedding(x)
        a = self.positional_embedding(a)
        x = x + a
        return x

class CLIPMultiHeadCasualSelfAttention(nn.Module):
    """
      Applies multi-head self-attention to the input tokens.
    
      Each token is projected into Query, Key and Value vectors. Attention
      scores are calculated between every pair of tokens, allowing information
      to be exchanged across the entire image. The outputs from all attention
      heads are concatenated and projected back to the original embedding
      dimension. We mask the tokens after the token we are focusing on to
      prevent cheating. This is done my creating a tensor of ones and
      covering part of it and 
    
      Args:
          embed_dim: Dimension of each token embedding.
          heads: Number of parallel attention heads.
    
      Returns:
          Tensor of shape (batch_size, num_tokens, embed_dim).
    
      Example:
          Input:                    (1, 197, 768)
          Query, Key, Value:        (1, 197, 768)
          Split into heads:         (1, 12, 197, 64)
          Attention scores:         (1, 12, 197, 197)
          Attention output:         (1, 12, 197, 64)
          Concatenated output:      (1, 197, 768)
          Final output:             (1, 197, 768)
      """
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        assert self.embed_dim % self.heads == 0, "Embedding dimension should the divisible by the number of heads"
        self.head_dim = self.embed_dim // self.heads
        self.qkv = nn.Linear(
            in_features=embed_dim, out_features=embed_dim * 3
        )
        self.projection = nn.Linear(
            in_features=embed_dim, out_features=embed_dim
        )

    def forward(self, x:torch.Tensor):
        batch, tokens, _ = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.reshape(batch, tokens, self.heads, self.head_dim).transpose(1,2)
        key = key.reshape(batch, tokens, self.heads, self.head_dim).transpose(1,2)
        value = value.reshape(batch, tokens, self.heads, self.head_dim).transpose(1,2)

        attn_scores = (query @ key.transpose(-1,-2)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(tokens, tokens, device=x.device))
        attn_scores = attn_scores.mask_filled(mask == 0, float("-inf"))
        attn_scores = torch.softmax(attn_scores, dim=-1) @ value
        attn_scores = attn_scores.transpose(1,2).reshape(batch, tokens, self.embed_dim)
        return self.projection(attn_scores)


class QuickGELU(nn.Module):
    """
    QuickGELU class

    Args:
        x: The input tensor

    Returns:
        quickgelu: The output tensor with the calculation done on it
    """
    def forward(self, x:torch.Tensor):
        quickgelu = x * torch.sigmoid(1.702 * x)
        return quickgelu
    
class CLIPMLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.quickgelu = QuickGELU()
        self.mlp_projection = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim * 4),
            self.quickgelu,
            nn.Linear(in_features=embed_dim * 4, out_features=embed_dim)
        )

    def forward(self, x:torch.Tensor):
        return self.mlp_projection(x)

class CLIPEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class CLIPTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# VAE CLASSES ------------------------------------------------------------------

class VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class VAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class VAEResNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class VAEAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class VAEDownsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class VAEUpsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class LatentDistribution(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# U-NET CLASSES ----------------------------------------------------------------

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

class UNETResNet(nn.Module):
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
        
class UNETDownsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class UNETUpsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class UNETSelfAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class UNETDownBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class UNETUpBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class UNETMidBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class UNETCrossAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# StableDiffusion CLASSES ------------------------------------------------------

class DiffusionSchedular():
    def __init__(self):
        pass

class DiffusionSampler():
    def __init__(self):
        pass

class StableDiffusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

