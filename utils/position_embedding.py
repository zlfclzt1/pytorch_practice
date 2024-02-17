import torch
import torch.nn as nn





class PositionEmbeddingSine(nn.Module):
    def __init__(self, model_d):
        super(PositionEmbeddingSine, self).__init__()

        self.model_d = model_d

    def forward(self, input):
        # sin(k/1000**(2*i/d) cos
        b, c, h, w = input.shape

        dim_t = 1000**(torch.arange(self.model_d) // 2 * 2.0 / self.model_d)

        x_embed = (torch.arange(w).reshape(-1, 1) / dim_t.reshape(1, -1)).unsqueeze(0).repeat(h, 1, 1)
        y_embed = (torch.arange(h).reshape(-1, 1) / dim_t.reshape(1, -1)).unsqueeze(1).repeat(1, w, 1)

        x_embed = torch.stack([x_embed[:, :, ::2].sin(), x_embed[:, :, 1::2].cos()], dim=-1).flatten(2)
        y_embed = torch.stack([y_embed[:, :, ::2].sin(), y_embed[:, :, 1::2].cos()], dim=-1).flatten(2)

        pos_embed = torch.cat([x_embed, y_embed], dim=2).unsqueeze(0).repeat(b, 1, 1, 1)
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        return pos_embed




class PositionEmbeddingLearnable(nn.Module):
    def __init__(self, max_l, model_d):
        super(PositionEmbeddingLearnable, self).__init__()

        self.col_embed = nn.Embedding(max_l, model_d)
        self.row_embed = nn.Embedding(max_l, model_d)

    def forward(self, input):
        b, c, h, w = input.shape

        x_embed = self.col_embed(torch.arange(w)).unsqueeze(0).repeat(h, 1, 1)
        y_embed = self.row_embed(torch.arange(h)).unsqueeze(1).repeat(1, w, 1)

        pos_embed = torch.cat([x_embed, y_embed], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        return pos_embed




if __name__ == "__main__":
    b, c, h, w = 3, 256, 14, 16
    pos_embed_learn = PositionEmbeddingLearnable(max_l=100, model_d=c//2)
    pos_embed_sine = PositionEmbeddingSine(model_d=c//2)

    input = torch.randn((b, c, h, w))
    pos_embed1 = pos_embed_learn(input)
    pos_embed2 = pos_embed_sine(input)



