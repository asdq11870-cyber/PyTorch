import torch
from torch import nn
import os
import yaml

class MellingerController(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass