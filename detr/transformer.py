import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, encode_depth=3, decode_depth=3):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.encode_depth = encode_depth
        self.decode_depth = decode_depth

        self.encoder = TransformerEncoderLayer(d_model=d_model, num_heads=num_heads)
        self.decoder = TransformerDecoderLayer(d_model=d_model, num_heads=num_heads)
    def forward(self, input, pos, query_embed):
        b, c, h, w = input.shape
        x = rearrange(input, 'b c h w -> b (h w) c')
        pos_embed = rearrange(pos, 'b c h w -> b (h w) c')
        query_embed = repeat(query_embed, 'n c -> b n c', b=b)
        tgt = torch.zeros_like(query_embed)

        for _ in range(self.encode_depth):
            x = self.encoder(x, pos_embed)
        memory = x

        for _ in range(self.decode_depth):
            tgt = self.decoder(tgt, memory, pos_embed, query_embed)
        return tgt

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim=2048, dropout=0.):
        super(TransformerEncoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.post_norm1 = nn.LayerNorm(d_model)
        self.post_norm2 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, num_heads)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.relu = nn.ReLU()

    def forward(self, input, pos):
        x0 = input
        x = self.attn(input, pos)
        x = x0 + self.dropout(x)
        x1 = self.post_norm1(x)

        x = self.linear2(self.relu(self.linear1(x1)))
        x = x1 + self.dropout(x)
        x = self.post_norm2(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.multi_head_attn = MultiHeadAttention(d_model, num_heads)

    def forward(self, input, pos):
        q = k = input + pos
        v = input
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        out = self.multi_head_attn(q, k, v)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.softmax = nn.Softmax(-1)

    def forward(self, q, k, v):
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** -0.5)
        attn = self.softmax(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim=2048, dropout=0.):
        super(TransformerDecoderLayer, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.q_proj_self = nn.Linear(d_model, d_model, bias=False)
        self.k_proj_self = nn.Linear(d_model, d_model, bias=False)
        self.v_proj_self = nn.Linear(d_model, d_model, bias=False)
        self.q_proj_cross = nn.Linear(d_model, d_model, bias=False)
        self.k_proj_cross = nn.Linear(d_model, d_model, bias=False)
        self.v_proj_cross = nn.Linear(d_model, d_model, bias=False)

        self.multi_head_attn = MultiHeadAttention(d_model, num_heads)
        self.post_norm1 = nn.LayerNorm(d_model)
        self.post_norm2 = nn.LayerNorm(d_model)
        self.post_norm3= nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.relu = nn.ReLU()


    def forward(self, tgt, memory, pos_embed, query_embed):
        src = tgt

        # self attention
        q = k = tgt + query_embed
        v = tgt
        q = self.q_proj_self(q)
        k = self.k_proj_self(k)
        v = self.v_proj_self(v)

        out = self.multi_head_attn(q, k, v)
        tgt1 = self.post_norm1(src + self.dropout(out))

        # cross attention
        q = tgt1 + query_embed
        k = memory + pos_embed
        v = memory
        q = self.q_proj_cross(q)
        k = self.k_proj_cross(k)
        v = self.v_proj_cross(v)
        out = self.multi_head_attn(q, k, v)
        tgt2 = self.post_norm2(tgt1 + self.dropout(out))

        out = self.linear2(self.dropout(self.relu(self.linear1(tgt2))))
        tgt3 = self.post_norm3(tgt2 + self.dropout(out))
        return tgt3

