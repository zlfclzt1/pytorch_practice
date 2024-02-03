import torch
import torch.nn as nn

from einops import rearrange, repeat


class Image2Patch(nn.Module):
    def __init__(self, image_size, patch_size, patch_dim=2048):
        super(Image2Patch, self).__init__()
        num_patch = (image_size // patch_size)**2

        self.patch_size = patch_size

        self.conv = nn.Conv2d(3, patch_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.rand(1, 1, patch_dim))
        self.pos_embedding = nn.Parameter(torch.rand(num_patch+1, patch_dim))


    def forward(self, images):
        # images = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
        #                    p1=self.patch_size, p2=self.patch_size)
        b, c, h, w = images.shape

        x = self.conv(images)
        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_token = self.cls_token.repeat(b, 1, 1)
        # cls_token = repeat(self.cls_token, '() n c -> b n c', b=b)
        pos_embedding = self.pos_embedding.repeat(b, 1, 1)
        # pos_embedding = repeat(self.pos_embedding[None, ...], '() n c -> b n c', b=b)

        x = torch.cat([x, cls_token], dim=1)
        x = x + pos_embedding
        return x


class FedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super(FedForward, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),   ##nn.GELU()
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



class Attention(nn.Module):
    def __init__(self, dim, num_heads, dim_head, dropout):
        super(Attention, self).__init__()

        inter_dim = num_heads * dim_head
        self.num_heads = num_heads
        self.dim_head = dim_head

        self.qkv = nn.Linear(dim, inter_dim*3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Linear(inter_dim, dim)

    def forward(self, input):
        h, d = self.num_heads, self.dim_head

        qkv = self.qkv(input)
        qkv = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', h=h, d=d)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(2, 3)) * (d**-0.5)
        attn = self.softmax(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.mlp(out)
        return out



class Transformer(nn.Module):
    def __init__(self, dim, depth, num_heads, dim_head, hidden_dim, dropout=0.):
        super(Transformer, self).__init__()

        self.layers = nn.ModuleList()
        self.pre_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([Attention(dim=dim, num_heads=num_heads,
                                         dim_head=dim_head, dropout=dropout),
                                            FedForward(dim=dim, hidden_dim=hidden_dim,
                                                      dropout=dropout)]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = self.dropout(attn(self.pre_norm(x))) + x
            x = self.dropout(ff(self.pre_norm(x))) + x
        return x

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, patch_dim, depth, num_heads, dim_head, hidden_dim, dropout=0., embd_dropout=0.):
        super(ViT, self).__init__()

        self.image2patch = Image2Patch(image_size, patch_size, patch_dim)
        self.dropout = nn.Dropout(embd_dropout)
        self.transformer = Transformer(dim=patch_dim, depth=depth,
                                       num_heads=num_heads, dim_head=dim_head,
                                       hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, images):
        x = self.image2patch(images)
        x = self.dropout(x)

        x = self.transformer(x)

        return x


if __name__ == "__main__":
    images = torch.randn((2, 3, 224, 224))
    vit = ViT(image_size=224, patch_size=16, patch_dim=1024, depth=3, num_heads=8, dim_head=64, hidden_dim=2048)
    x = vit(images)