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
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        assert self.embed_dim % 2 == 0, "Embedded Dimension should be even!"

    def forward(self, timestep:torch.Tensor):
        x = torch.zeros(timestep.shape[0], self.embed_dim).to(device=timestep.device, dtype="float32")
        j = 0
        for i in range(0,self.embed_dim,2):
            x[:,i] = torch.sin(timestep/torch.pow(10000,(2*j)/self.embed_dim))
            x[:,i+1] = torch.cos(timestep/torch.pow(10000,(2*j)/self.embed_dim))    
            j += 1 
        return x

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

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


