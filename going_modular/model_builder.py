import torch
from torch import nn
import torch.nn.functional as F

class TinyVGG(nn.Module):
  """
  Creates a TinyVGG architecture
  Replicates the architecture of the CNN Explainer website

  Args:
    input: The shape of the input size
    hidden: The amount of hidden units there are
    output: The shape of the output size

  Returns:
    A tensor that has gone through the convulution layers and the classifier layer
  """
  def __init__(self,input:int, hidden:int, output:int):
    super().__init__()
    self.relu = nn.ReLU()
    self.conv_layer_1 = nn.Sequential(
        nn.Conv2d(in_channels=input, out_channels=hidden, kernel_size=3, stride=1, padding=1),
        self.relu,
        nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3, stride=1, padding=1),
        self.relu,
        nn.MaxPool2d(kernel_size=3, stride=2)
    )
    self.conv_layer_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3, stride=1, padding=1),
        self.relu,
        nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3, stride=1, padding=1),
        self.relu,
        nn.MaxPool2d(kernel_size=3, stride=2)
    )
    self.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=1),
        nn.Flatten(),
        nn.Linear(hidden, output)
    )

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.classifier(self.conv_layer_2(self.conv_layer_1(x)))

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

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


class MultiHeadSelfAttention(nn.Module):
  """
  Applies multi-head self-attention to the input tokens.

  Each token is projected into Query, Key and Value vectors. Attention
  scores are calculated between every pair of tokens, allowing information
  to be exchanged across the entire image. The outputs from all attention
  heads are concatenated and projected back to the original embedding
  dimension.

  Args:
      embed_dim: Dimension of each token embedding.
      heads: Number of parallel attention heads.
      attn_dropout: Dropout probability applied to the attention weights.

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
  def __init__(self, embed_dim, heads, attn_dropout):
      super().__init__()
      self.heads = heads
      self.embed_dim = embed_dim
      self.query_key_value = nn.Linear(embed_dim, embed_dim * 3)
      self.projection = nn.Linear(embed_dim, embed_dim)
      assert embed_dim % heads == 0
      self.head_dim = embed_dim // heads
      self.attention_dropout = nn.Dropout(attn_dropout)

  def forward(self, x):
      batch, tokens, _ = x.shape
      qkv = self.query_key_value(x)
      query, key, value = qkv.chunk(3, dim=-1)

      query = query.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
      key = key.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
      value = value.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

      scores = torch.matmul(query, key.transpose(-2, -1))
      scores = scores / (self.head_dim ** 0.5)
      attention = self.attention_dropout(torch.softmax(scores, dim=-1))
      out = torch.matmul(attention, value)

      out = out.transpose(1, 2).reshape(batch, tokens, self.embed_dim)
      return self.projection(out)


class FeedForward(nn.Module):
  """
  Implements the feed-forward network used within a Transformer encoder.

  Args:
      embed_dim: Dimension of each token embedding.
      mlp_dim: Hidden dimension of the feed-forward network.
      mlp_dropout: Dropout probability applied after each linear layer.

  Returns:
      Tensor of shape (batch_size, num_tokens, embed_dim).

  Example:
      Input:                    (1, 197, 768)
      First Linear Layer:       (1, 197, 3072)
      GELU + Dropout:           (1, 197, 3072)
      Second Linear Layer:      (1, 197, 768)
      Output:                   (1, 197, 768)
  """
  def __init__(self, embed_dim, mlp_dim, mlp_dropout):
      super().__init__()
      self.layer = nn.Sequential(
          nn.Linear(embed_dim, mlp_dim),
          nn.GELU(),
          nn.Dropout(mlp_dropout),
          nn.Linear(mlp_dim, embed_dim),
          nn.Dropout(mlp_dropout),
      )

  def forward(self, x):
      return self.layer(x)


class TransformEncoderBlock(nn.Module):
  """
  Implements a single Vision Transformer encoder block.

  Args:
      embed_dim: Dimension of each token embedding.
      heads: Number of attention heads.
      mlp_dim: Hidden dimension of the feed-forward network.
      mlp_dropout: Dropout probability within the feed-forward network.
      attn_dropout: Dropout probability applied to the attention weights.

  Returns:
      Tensor of shape (batch_size, num_tokens, embed_dim).

  Example:
      Input:                    (1, 197, 768)
      LayerNorm:                (1, 197, 768)
      Self-Attention:           (1, 197, 768)
      First Residual Add:       (1, 197, 768)
      LayerNorm:                (1, 197, 768)
      Feed Forward Network:     (1, 197, 768)
      Second Residual Add:      (1, 197, 768)
      Output:                   (1, 197, 768)
  """
  def __init__(self, embed_dim, heads, mlp_dim, mlp_dropout, attn_dropout):
    super().__init__()

    self.norm1 = nn.LayerNorm(embed_dim)
    self.attention = MultiHeadSelfAttention(
      embed_dim=embed_dim,
      heads=heads,
      attn_dropout=attn_dropout
    )
    self.norm2 = nn.LayerNorm(embed_dim)
    self.mlp = FeedForward(
      embed_dim=embed_dim,
      mlp_dim=mlp_dim,
      mlp_dropout=mlp_dropout
    )

  def forward(self, x):
    x = x + self.attention(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x

class ViT(nn.Module):
  def __init__(self, image_size, patch_size, in_channels,
                embed_dim, heads, mlp_dim, num_classes, num_encoder_layers,
                attn_dropout, mlp_dropout, emb_dropout):
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
        TransformEncoderBlock(
          embed_dim=embed_dim,
          heads=heads,
          mlp_dim=mlp_dim,
          mlp_dropout=mlp_dropout,
          attn_dropout=attn_dropout
        )
        for _ in range(num_encoder_layers)
      ]
    )

    self.mlp_head = nn.Linear(
      in_features=embed_dim,
      out_features=num_classes
    )
    self.norm = nn.LayerNorm(embed_dim)
    self.emb_dropout = nn.Dropout(p=emb_dropout)

  def forward(self, x):
    x = self.patch_embedding(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = x + self.position_embedding
    x = self.emb_dropout(x)
    for block in self.encoder_blocks:
      x = block(x)
    x = self.norm(x)
    x = x[:,0]
    x = self.mlp_head(x)
    return x
  
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------


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
  
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

class TokenEmbedding(nn.Module):
  def __init__(self):
    super().__init__()

  def forward():
    pass

class MultiHeadCasualSelfAttention(nn.Module):
  def __init__(self):
    super().__init__()

  def forward():
    pass

class GPT(nn.Module):
  def __init__(self):
    super().__init__()

  def forward():
    pass








