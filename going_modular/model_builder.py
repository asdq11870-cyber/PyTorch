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
  This class if for splitting an image into batches
  This is done using the conv2d module to split our image and create a tensor
  then the last two dimensions are combined into one and flattened and then the 
  tensor is transposed to swap the dimensions 1 and 2

  Args:
    patch_size: The pixel dimensions of each patch
    in_channels: The number of colour channels like red, green, blue
    embed_dim: The number of features or pixels that include the colour channels

  Returns:
    A tensor of (batch, tokens, features)

  Examples:
    input tensor: (1, 3, 224, 224) (batch, channels, height, width)
    patched tensor: (1, 768, 14, 14) (batch, emb_dim, patch_rows, patch_cols)
    flattened tensor: (1, 768, 196) (batch, emb_dim, tokens)
    transposed tensor: (1, 196, 768) (batch, tokens, emb_dim)

  """
  def __init__(self, patch_size,
                in_channels,
                  embed_dim
                  ):
    
    super().__init__()
    self.create_patches = nn.Conv2d(
      in_channels=in_channels,
      out_channels=embed_dim,
      kernel_size=(patch_size,patch_size),
      stride=patch_size
    )
  
  def forward(self, x):
    x = self.create_patches(x)
    x = x.flatten(start_dim=2)
    x = x.transpose(1,2)
    return x

class MultiHeadSelfAttention(nn.Module):

  def __init__(self, embed_dim, heads, attn_dropout):
    super().__init__()
    self.heads = heads
    self.embed_dim = embed_dim
    self.query_key_value = nn.Linear(
      in_features=embed_dim,
      out_features=embed_dim*3
    )
    self.projection = nn.Linear(
      in_features=embed_dim,
      out_features=embed_dim
    )
    assert embed_dim % heads == 0
    self.head_dim = embed_dim // self.heads # Calculates the number of features per head
    self.attention_dropout = nn.Dropout(p=attn_dropout)

  def forward(self, x):
    batch, tokens, _ = x.shape # Aquired these values when this class is called 
    query_key_value = self.query_key_value(x) # Creating a vector 3 times the size
    query, key, value = query_key_value.chunk(3, dim=-1) # Splits the vector into 3 vectors with 768 features
    # (batch, tokens, features)
    
    query = query.reshape(batch, tokens, self.heads, self.head_dim)
    key = key.reshape(batch, tokens, self.heads, self.head_dim)
    value = value.reshape(batch, tokens, self.heads, self.head_dim) # The information is currently (batch, tokens, heads, features per head)

    query = query.transpose(1,2)
    key = key.transpose(1,2)
    value = value.transpose(1,2) # The information now is (batch, heads, tokens, features per head)

    scores = torch.matmul(query, key.transpose(-2,-1))
    scores = scores / (self.head_dim ** 0.5)
    attention = torch.softmax(scores, dim=-1)
    attention = self.attention_dropout(attention)
    out = torch.matmul(attention, value)

    out = out.transpose(1,2)
    out = out.reshape(batch, tokens, self.embed_dim)
    out = self.projection(out)

    return out

class FeedForward(nn.Module):
  def __init__(self, embed_dim, mlp_dim, mlp_dropout):
    super().__init__()

    self.gelu_activation = nn.GELU() # add non linearity
    self.layer = nn.Sequential(
      nn.Linear(in_features=embed_dim, out_features=mlp_dim), # increases size of tokens 768 -> 3072 to create more complex feature relationships
      self.gelu_activation,
      nn.Dropout(p=mlp_dropout),
      nn.Linear(in_features=mlp_dim, out_features=embed_dim), # Compressing the size back to prepare for another encoding
      nn.Dropout(p=mlp_dropout)
    )

  def forward(self, x):
    return self.layer(x)

class TransformEncoderBlock(nn.Module):
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