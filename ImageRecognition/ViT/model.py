import torch
import torch.nn as nn
import sys
sys.path.append("/Users/tora/Desktop/DL/DLModels/Modules")
from Attention.Attention import MultiHeadAttention

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class SinusoidPostionalEmbedding(nn.Module):
    def __init__(
        self,
        L,
        dim
    ):
        super().__init__()
        self.L = L
        self.dim_model = dim
        theta_even = torch.arange(L)[:, None] / (10000 ** (2 * torch.arange(dim // 2)[None, :] / dim))
        theta_odd = torch.arange(L)[:, None] / (10000 ** (2 * torch.arange((dim + 1) // 2)[None, :] / dim))
        
        pos_encoding_even = torch.sin(theta_even)
        pos_encoding_odd  = torch.cos(theta_odd)
        
        self.pos_encoding = torch.empty((L, dim))
        self.pos_encoding[:, 0::2] = pos_encoding_even
        self.pos_encoding[:, 1::2] = pos_encoding_odd
        self.pos_encoding = self.pos_encoding[:, None, :]
        
    def forward(
        self,
        inputs
    ):
        L, _, dim = inputs.shape  # (L, n_batch, dim)
        assert L == self.L and dim == self.dim_model
        self.pos_encoding = self.pos_encoding.to(inputs.device)
        x = inputs + self.pos_encoding
        return x

class FeedForward(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out
    ):
        super().__init__()
        self.fc_in = nn.Linear(
            in_features=dim_in,
            out_features=dim_hidden
        )
        self.fc_out = nn.Linear(
            in_features=dim_hidden,
            out_features=dim_out
        )
        self.activation = nn.GELU()
        
    def forward(
        self,
        x
    ):
        x = self.fc_in(x)
        x = self.activation(x)
        x = self.fc_out(x)
        
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim_model,
        dim_head,
        dim_mlp,
        n_head
    ):
        super().__init__()
        self.dim_model = dim_model
        self.n_head = n_head
        self.layer_norm = nn.LayerNorm(dim_model)
        self.attention = MultiHeadAttention(dim_model, dim_head, dim_head, dim_head, n_head)
        self.mlp = FeedForward(dim_model, dim_mlp, dim_model)
    
    def forward(
        self,
        x
    ):
        L, n_batch, dim = x.shape
        assert dim == self.dim_model
        
        identity = x
        x = self.layer_norm(x)
        x = self.attention(x, x, x)
        x += identity

        identity = x
        x = self.layer_norm(x)
        x = self.mlp(x)
        x += identity
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim_model,
        dim_head,
        dim_mlp,
        n_head,
        depth
    ):
        super().__init__()
        self.depth = depth
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(dim_model, dim_head, dim_mlp, n_head)
        for _ in range(depth)])
    
    def forward(
        self,
        x
    ):
        for i in range(self.depth):
            x = self.encoder[i](x)
        return x

class ViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim_model,
        depth,
        n_head,
        dim_mlp,
        channels=3,
        dim_head=64
    ):
        super().__init__()
        self.H, self.W = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)
        assert self.H % self.patch_height == 0 and self.W % self.patch_width == 0, "Impossible value pairs to make patch"
        self.h, self.w = self.H // self.patch_height, self.W // self.patch_width
        
        self.channels = channels
        self.patch_num = self.h * self.w
        self.patch_dim = self.patch_height * self.patch_width * self.channels
        self.dim_model = dim_model
        
        # Embeddings
        self.pos_embedding = SinusoidPostionalEmbedding(
            self.patch_num,
            self.patch_dim)
        self.embedding = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim_model),
            nn.LayerNorm(dim_model)
        )
        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(dim_model, dim_head, dim_mlp, n_head, depth)
        # Classification Head
        self.classification_head = nn.Sequential(
            nn.LayerNorm(self.patch_num * self.dim_model),
            nn.Linear(
                in_features=self.patch_num * self.dim_model,
                out_features=num_classes
            )
        )
        
    def forward(
        self,
        image
    ):
        n_batch, H, W, C = image.shape
        assert self.H == H and self.W == W and self.channels == C, f"Invalid shape of the images: the shape must be ({self.H}, {self.W}, {self.channels}), and yours are ({H}, {W}, {C})"
        x = image.reshape(n_batch, self.h, self.patch_height, self.w, self.patch_width, C)\
                 .permute(0, 1, 3, 2, 4, 5)\
                 .reshape(n_batch, self.patch_num, self.patch_dim)\
                 .permute(1, 0, 2)
        # Embedding
        x = self.pos_embedding(x)
        # x = x.reshape(-1, self.dim_model)
        x = self.embedding(x)
        # x = x.reshape(self.patch_num, n_batch, self.dim_model)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Classification
        x = x.permute(1, 0, 2).reshape(n_batch, self.patch_num * self.dim_model)
        x = self.classification_head(x)
        
        return x

