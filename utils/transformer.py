import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0., ):
        super(MultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = self.embed_size // self.num_heads
        self.dropout = dropout
        assert self.embed_size == self.num_heads * self.head_size

        self.project_q = nn.Linear(embed_size, embed_size, bias=False)
        self.project_k = nn.Linear(embed_size, embed_size, bias=False)
        self.project_v = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, query, key, value):
        # B L C
        q = self.project_q(query)
        k = self.project_q(key)
        v = self.project_q(value)

        q = rearrange(q, 'b l (n h) -> b n l h', n=self.num_heads)
        k = rearrange(k, 'b l (n h) -> b n l h', n=self.num_heads)
        v = rearrange(v, 'b l (n h) -> b n l h', n=self.num_heads)

        attn = torch.matmul(q, k.transpose(-1, -2)) / self.head_size**0.5
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout)

        output = torch.matmul(attn, v)
        output = rearrange(output, 'b n l d -> b l (n d)')
        return output

class FedForwardNet(nn.Module):
    def __init__(self, embed_size, hidden_size, post_norm=True, dropout=0.):
        super(FedForwardNet, self).__init__()

        self.post_norm = post_norm
        self.dropout = dropout

        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, input):
        if self.post_norm:
            x = self.forward_post(input)
        else:
            x = self.forward_pre(input)
        return x

    def forward_pre(self, input):
        x = self.layer_norm(input)
        x = self.fc2(F.gelu(self.fc1(x)))
        x = F.dropout(x, self.dropout)
        x = input + x
        return x

    def forward_post(self, input):
        x = self.fc2(F.gelu(self.fc1(input)))
        x = F.dropout(x, self.dropout)
        x = input + x
        x = self.layer_norm(x)
        return x




class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, encoder_depth, post_norm=True, dropout=0.):
        super(TransformerEncoder, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.encoder_depth = encoder_depth
        self.post_norm = post_norm
        self.dropout = dropout


    def forward(self, input):
        x = input
        for _ in range(self.encoder_depth):




if __name__ == "__main__":
    b = 3
    embed_size = 256
    query = torch.rand(b, 14 * 14, 256)
    key = torch.rand(b, 1000, 256)
    value = torch.rand(b, 1000, 256)

    mh_attn = MultiHeadAttention(embed_size=embed_size, num_heads=8)
    result = mh_attn(query=query, key=key, value=value)