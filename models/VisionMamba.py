import torch
from torch import nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
  """
  Splits an image into non-overlapping patches and projects each patch into
  an embedding vector using a Conv2D layer.

  The resulting patch embeddings are flattened and transposed into the
  format expected by a Vision Transformer.

  Args:
      patch_size: Height and width of each square image patch.
      in_channels: Number of input image channels (e.g. 3 for RGB).
      embed_dim: Dimension of the embedding vector produced for each patch.

  Returns:
      Tensor of shape (batch_size, num_patches, embed_dim).

  Example:
      Input:             (1, 3, 224, 224)
      After Conv2D:      (1, 768, 14, 14)
      After flatten():   (1, 768, 196)
      After transpose(): (1, 196, 768)
  """
  def __init__(self, patch_size, in_channels, embed_dim):
      super().__init__()
      self.create_patches = nn.Conv2d(
          in_channels=in_channels,
          out_channels=embed_dim,
          kernel_size=(patch_size, patch_size),
          stride=patch_size
      )

  def forward(self, x):
      x = self.create_patches(x)
      x = x.flatten(start_dim=2)
      x = x.transpose(1, 2)
      return x

class SelectiveSSM(nn.Module):
  def __init__(self, embed_dim, ssm_dim):
    super().__init__()
    self.A = nn.Parameter(torch.rand((ssm_dim,ssm_dim)))
    self.B_projection = nn.Linear(embed_dim,ssm_dim)
    self.C_projection = nn.Linear(embed_dim,ssm_dim)
    self.delta_projection = nn.Linear(embed_dim,ssm_dim)
    self.ssm_dim = ssm_dim
    self.embed_dim = embed_dim

  def discretization(self, delta, B):
    _, tokens, _ = delta.shape
    A_bar = []
    B_bar = []
    I = torch.eye(self.ssm_dim, device=delta.device)
    for t in range(tokens):
      delta_A = delta[:,t,:].unsqueeze(-1) * self.A
      A_t = torch.matrix_exp(delta_A)
      B_t = torch.linalg.solve(delta_A, torch.matmul(A_t - I, B[:,t,:].unsqueeze(-1))).squeeze(-1)
      A_bar.append(A_t)
      B_bar.append(B_t)
    return torch.stack(A_bar,dim=1), torch.stack(B_bar,dim=1)

  def scan_forward(self,x):
    batch,tokens,_ = x.shape
    B = self.B_projection(x)
    C = self.C_projection(x)
    delta = F.softplus(self.delta_projection(x))
    A_bar,B_bar = self.discretization(delta,B)
    h = torch.zeros(batch,self.ssm_dim,device=x.device)
    outputs = []
    for i in range(tokens):
      h = torch.matmul(h.unsqueeze(1),A_bar[:,i]).squeeze(1) + B_bar[:,i]
      y = C[:,i] * h
      outputs.append(y)
    return torch.stack(outputs,dim=1)

  def scan_backward(self,x):
    batch,tokens,_ = x.shape
    B = self.B_projection(x)
    C = self.C_projection(x)
    delta = F.softplus(self.delta_projection(x))
    A_bar,B_bar = self.discretization(delta,B)
    h = torch.zeros(batch,self.ssm_dim,device=x.device)
    outputs = []
    for i in reversed(range(tokens)):
      h = torch.matmul(h.unsqueeze(1),A_bar[:,i]).squeeze(1) + B_bar[:,i]
      y = C[:,i] * h
      outputs.append(y)
    outputs.reverse()
    return torch.stack(outputs,dim=1)

  def forward(self,x):
    forward = self.scan_forward(x)
    backward = self.scan_backward(x)
    return forward,backward


class VisionMambaEncoder(nn.Module):
  def __init__(self, embed_dim, ssm_dim, expand_dim):
    super().__init__()
    self.norm = nn.LayerNorm(embed_dim)
    self.X_projection = nn.Linear(in_features=embed_dim,out_features=expand_dim)
    self.Z_projection = nn.Linear(in_features=embed_dim,out_features=expand_dim)
    self.Y_projection = nn.Linear(in_features=expand_dim, out_features=embed_dim)

    self.conv_layer_1 = nn.Conv1d(in_channels=expand_dim, out_channels=expand_dim,kernel=3,padding=1)
    self.conv_layer_2 = nn.Conv1d(in_channels=expand_dim, out_channels=expand_dim,kernel=3,padding=1)

    self.forward_ssm = SelectiveSSM(
      embed_dim=expand_dim,
      ssm_dim=ssm_dim
    )

    self.backward_ssm = SelectiveSSM(
      embed_dim=expand_dim,
      ssm_dim=ssm_dim
    )

    self.ssm_output = nn.Linear(in_features=ssm_dim, out_features=expand_dim)

    self.embed_dim = embed_dim
    self.expand_dim = expand_dim
    self.ssm_dim = ssm_dim

  def forward(self, y):
    residual = y
    y = self.norm(y)
    x = self.X_projection(y)
    z = self.Z_projection(y)
    x_forward = self.conv_layer_1(x.transpose(1,2))
    x_forward = x_forward.transpose(1,2)
    x_backward = torch.flip(x,dims=[1])
    x_backward = self.conv_layer_2(x_backward.transpose(1,2))
    x_backward = x_backward.transpose(1,2)
    y_forward,_ = self.forward_ssm(x_forward)
    _,y_backward = self.backward_ssm(x_backward)
    y_backward = torch.flip(y_backward, dims=[1])
    y_forward, y_backward = self.ssm_output(y_forward), self.ssm_output(y_backward)
    y = y_forward * F.silu(z) + y_backward * F.silu(z)
    y = self.Y_projection(y)
    y = y + residual
    return y


class VisionMamba(nn.Module):
  def __init__(self, image_size, patch_size, in_channels,
                embed_dim, num_classes, num_encoder_layers,
                expand_dim, ssm_dim):
    super().__init__()

    assert image_size % patch_size == 0
    self.num_patch = (image_size ** 2) // (patch_size ** 2)

    self.patch_embedding = PatchEmbedding(
      patch_size=patch_size,
      in_channels=in_channels,
      embed_dim=embed_dim
    )

    self.cls_token = nn.Parameter(
      torch.randn(1,1,embed_dim)
    )

    self.position_embedding = nn.Parameter(
      torch.randn(1, self.num_patch+1, embed_dim)
    )

    self.encoder_blocks = nn.ModuleList(
      [
        VisionMambaEncoder(
          embed_dim=embed_dim,
          ssm_dim=ssm_dim,
          expand_dim=expand_dim
        )
        for _ in range(num_encoder_layers)
      ]
    )

    self.mlp_head = nn.Linear(
      in_features=embed_dim,
      out_features=num_classes
    )

  def forward(self, x):
    x = self.patch_embedding(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = x + self.position_embedding
    for block in self.encoder_blocks:
      x = block(x)
    x = x.mean(dim=1)
    x = self.mlp_head(x)
    return x