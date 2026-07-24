import torch
from torch import nn

class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size, embed_dim):
    super().__init__()
    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

  def forward(self, x):
    return self.embedding(x)

class MultiHeadCasualSelfAttention(nn.Module):
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
      mask = torch.tril(torch.ones(tokens,tokens,device=x.device))
      scores = scores.masked_fill(mask == 0, float("-inf"))
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

class GPTTransformBlock(nn.Module):
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
    self.attention = MultiHeadCasualSelfAttention(
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


class GPT(nn.Module):
  def __init__(self, vocab_size,
                embed_dim, heads, mlp_dim,
                  mlp_dropout, attn_dropout,
                    num_encoder_layers, context_length):
    
    super().__init__()
    self.token_embedding = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)
    self.positional_embedding = nn.Parameter(torch.randn((1,context_length,embed_dim)))
    self.dropout = nn.Dropout(p=0.1)
    self.norm = nn.LayerNorm(embed_dim)
    self.encoder_blocks = nn.ModuleList(
      [
        GPTTransformBlock(
          embed_dim=embed_dim, heads=heads, mlp_dim=mlp_dim,
          mlp_dropout=mlp_dropout,attn_dropout=attn_dropout
        )
        for _ in range(num_encoder_layers)
      ]
    )
    self.LLM_head = nn.Linear(
      in_features=embed_dim, out_features=vocab_size
    )

  def forward(self, x):
    x = self.token_embedding(x)
    x = x + self.positional_embedding
    x = self.dropout(x)
    for block in self.encoder_blocks:
      x = block(x)
    x = self.norm(x)
    x = self.LLM_head(x)
    return x