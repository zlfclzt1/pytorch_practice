import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import Backbone
from position_embedding import PositionEnbeddingSine
from transformer import Transformer

class Detr(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Detr, self).__init__()

        self.backbone = Backbone()
        self.pos_embed = PositionEnbeddingSine(num_pos_feats=d_model//2)
        self.input_proj = nn.Conv2d(2048, d_model, kernel_size=1)
        self.transformer = Transformer(d_model, num_heads)

        self.query_embed = nn.Embedding(num_queries, d_model)

    def forward(self, input):
        x = self.backbone(input) # N * 2048 * 8 * 22
        x = self.input_proj(x)
        pos = self.pos_embed(x)
        # N * hidden_dim * 8 * 22
        hs = self.transformer(x, pos, self.query_embed.weight)


        return x


if __name__ == "__main__":

    images = torch.randn((2, 3, 256, 704))
    d_model = 256
    ff_dim = 2048
    num_heads = 8
    num_queries = 1000

    detr = Detr(d_model=d_model, num_heads=num_heads)
    result = detr(images)