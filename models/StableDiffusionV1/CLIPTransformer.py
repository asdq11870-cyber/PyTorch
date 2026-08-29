# Text Conditioning Class used in StableDiffusionV1
import torch
from torch import nn

class Embedding(nn.Module):
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

    Example Tensor:
        x: (batch_size, tokens)
        a: (tokens)
        After token embedding, x:(batch_size, tokens, embed_dim)
        After positional embedding, a:(tokens, embed_dim)
        After addition, x:(batch_size, tokens, embed_dim)
        Output, x:(batch_size, tokens, embed_dim)
    """
    def __init__(self, embed_dim:int, vocab_size:int, max_seq_len:int):
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

class MultiHeadCausalSelfAttention(nn.Module):
    """
      Applies multi-head self-attention to the input tokens.
    
      Each token is projected into Query, Key and Value vectors. Attention
      scores are calculated between every pair of tokens, allowing information
      to be exchanged across the entire token sequence. The outputs from all attention
      heads are concatenated and projected back to the original embedding
      dimension. We mask the tokens after the token we are focusing on to
      prevent cheating. This is done by creating a tensor of ones and
      covering part of it. The attention score is made negative infinity if
      a particular token is covered. Softmax and matrix multiplication with the 
      value tensor. The final output matrix is transposed, reshaped and projected. 
    
      Args:
          embed_dim: Dimension of each token embedding.
          heads: Number of parallel attention heads.
    
      Returns:
          Tensor of shape (batch_size, tokens, embed_dim).
    
      Example Tensor:
          x: (batch_size, tokens, embed_dim)
          query, key, value: (batch_size, tokens, embed_dim)
          After reshape: (batch_size, tokens, self.heads, self.head_dim)
          After transpose: (batch_size, self.heads, tokens, self.head_dim)
          Attention Scores: (batch_size, self.heads, tokens, tokens)
          After projection: (batch_size, tokens, self.embed_dim)
      """
    def __init__(self, embed_dim:int, heads:int):
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
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_scores = torch.softmax(attn_scores, dim=-1) @ value
        attn_scores = attn_scores.transpose(1,2).reshape(batch, tokens, self.embed_dim)
        return self.projection(attn_scores)


class QuickGELU(nn.Module):
    """
    QuickGELU class
    
    Applies non-linearity to the feed forward class which allows the model
    to understand complex relationships instead of the model decaying into
    a linear system.

    Args:
        x: The input tensor

    Returns:
        quickgelu: The output tensor with the calculation done on it
    """
    def forward(self, x:torch.Tensor):
        quickgelu = x * torch.sigmoid(1.702 * x)
        return quickgelu
    
class FeedForward(nn.Module):
    """
    Implements the multilayer preceptron into the  transformer.

    This class is used to add non-linearity by expanding features and
    transform feature dimensions before adding non-linearity to the system.
    If non-linearity is not applied then the model collapses to a linear 
    system.

    Args:
        embed_dim: The embedding dimension
        x: The input tensor

    Returns:
        x: Output tensor passed through the mlp_projection to add non-linearity

    Example Tensor:
        x: (batch_size, tokens, self.embed_dim)
        After linear projection: (batch_size, tokens, self.embed_dim * 4)
        After QuickGELU: (batch_size, tokens, self.embed_dim * 4)
        After 2nd linear projection: (batch_size, tokens, self.embed_dim)
        x: (batch_size, tokens, self.embed_dim)
    """
    def __init__(self, embed_dim:int):
        super().__init__()
        self.quickgelu = QuickGELU()
        self.mlp_projection = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim * 4),
            self.quickgelu,
            nn.Linear(in_features=embed_dim * 4, out_features=embed_dim)
        )

    def forward(self, x:torch.Tensor):
        return self.mlp_projection(x)

class EncoderLayer(nn.Module):
    """
    A single encoder block of the CLIP Transformer
    
    he input token representations are first normalized
    and passed through causal multi-head self-attention.
    The attention output is added to the original input through
    a residual connection. The result is then normalized again 
    and passed through a feed-forward network using QuickGELU
    for non-linearity. A second residual connection adds the feed-forward
    output to the previous representation.

    Args:
        embed_dim: The embedding dimension passed to feed forward and self attention classes
        heads: The number of heads in the self attention class
        x: Input tensor

    Return:
        x: Output tensor from one encoder block

    Example Tensor:
        x: (batch_size, tokens, embed_dim)
        After layernorm1, x: (batch_size, tokens, embed_dim)
        After self_attention, x: (batch_size, tokens, embed_dim)
        After residual1, x: (batch_size, tokens, embed_dim)
        After layernorm2, x: (batch_size, tokens, embed_dim)
        After mlp, x: (batch_size, tokens, embed_dim)
        After residual2, x: (batch_size, tokens, embed_dim)
        Output, x: (batch_size, tokens, embed_dim)
    """
    def __init__(self, embed_dim:int, heads:int):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.self_attention = MultiHeadCausalSelfAttention(
            embed_dim=embed_dim, heads=heads
        )
        self.layernorm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.mlp = FeedForward(
            embed_dim=embed_dim
        )

    def forward(self, x:torch.Tensor):
        residual1 = x
        x = self.layernorm1(x)
        x = self.self_attention(x)
        x = x + residual1
        residual2 = x
        x = self.layernorm2(x)
        x = self.mlp(x)
        x = x + residual2
        return x

class CLIPTransformer(nn.Module):
    """
    The complete CLIP text encoder used to convert a sequence
    of token IDs into contextualized text representations for Stable Diffusion.

		The input token IDs are first converted into token and positional embeddings.
    These embeddings are then passed sequentially through multiple CLIP Transformer
    encoder layers, allowing each token to incorporate contextual information from 
    the surrounding text while respecting the causal attention mask. A final LayerNorm
    is applied to the resulting representations and the normalised shape is configured
    to the embedding dimension.

		The output contains a contextualized embedding for every token in the input sequence.
    These representations are subsequently used by the Stable Diffusion U-Net as text 
    conditioning through cross-attention.

    Args:
        embed_dim: The embedding dimension
        heads: The amount of heads in the self attention class
        vocab_size: Total number of tokens
        max_seq_len: Maximum number of consecutive tokens positional embedding can represent 
        num_encoder_layers: The number of encoder layers in the module list

    Returns:
        x: The text conditioned tensor

    Example Tensor:
        x: (batch_size, tokens)
        After embedding, x: (batch_size, tokens, embed_dim)
        After encoder_layers, x: (batch_size, tokens, embed_dim)
        After layernorm, x: (batch_size, tokens, embed_dim)
        Output, x: (batch_size, tokens, embed_dim)
    """
    def __init__(self, embed_dim, heads, vocab_size, max_seq_len, num_encoder_layers):
        super().__init__()
        self.embedding = Embedding(
            embed_dim=embed_dim, vocab_size=vocab_size, max_seq_len=max_seq_len
        )
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(embed_dim=embed_dim, heads=heads)
                for _ in range(num_encoder_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x:torch.Tensor):
        x = self.embedding(x)
        for block in self.encoder_layers:
            x = block(x)
        x = self.layernorm(x)
        return x
