import torch #type: ignore
from torch import nn #type: ignore

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