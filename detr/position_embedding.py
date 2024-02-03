import torch
import torch.nn as nn


class PositionEnbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000):
        super(PositionEnbeddingSine, self).__init__()

        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    # def forward(self, input):
    #     #input shape: B C H W
    #     B, C, H, W = input.shape
    #
    #     x_embed = torch.arange(W, dtype=torch.float32).repeat(B, H, 1)
    #     y_embed = torch.arange(H, dtype=torch.float32)[None, :, None].repeat(B, 1, W)
    #
    #     dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
    #     dim_t = self.temperature ** (2. * (dim_t // 2) / self.num_pos_feats)
    #
    #     pos_x = x_embed[:, :, :, None] / dim_t
    #     pos_y = y_embed[:, :, :, None] / dim_t
    #     pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
    #     pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
    #
    #     pos = torch.cat([pos_y, pos_x], dim=-1).permute(0, 3, 1, 2)
    #     return pos

    def forward(self, x):
        B, C, H, W = x.shape
        mask = torch.ones((B, H, W), dtype=bool)
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num_pos_feats=256):
        super(PositionEmbeddingLearned, self).__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)

    def forward(self, input):
        B, C, H, W = input.shape

        i = torch.arange(W, device=input.device)
        j = torch.arange(H, device=input.device)
        x_embed = self.col_embed(i)
        y_embed = self.row_embed(j)

        pos = torch.cat([
            x_embed.unsqueeze(0).repeat(H, 1, 1),
            y_embed.unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
        return pos


if __name__ == '__main__':
    input = torch.randn(2, 64, 12, 16)
    pos_embed_sine = PositionEnbeddingSine()
    pos_embed_learned = PositionEmbeddingLearned()
    result0 = pos_embed_sine(input)
    result1 = pos_embed_learned(input)

    print(result1)

