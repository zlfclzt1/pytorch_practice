import numpy as np
import torch

num_pos_feats = 128
temperature = 10000
B = 50
L = 1024 ## 1D
H, W = 704, 256     ## 2D

## 1D
feat = torch.zeros((B, L, num_pos_feats))

embed = (torch.arange(num_pos_feats, dtype=torch.float32) // 2) * 2 / num_pos_feats
embed = 1 / (temperature ** embed)

pos_embed = torch.arange(L, dtype=torch.float32).reshape(-1, 1).repeat(1, num_pos_feats)
pos_embed = pos_embed * embed
pos_embed = torch.stack([pos_embed[:, 0::2].sin(), pos_embed[:, 1::2].cos()], dim = 2).flatten(1)


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

inv_freq = 1.0 / (10000 ** (torch.arange(0, num_pos_feats, 2).float() / num_pos_feats))
pos_x = torch.arange(L).type(inv_freq.type())
sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
emb_x = get_emb(sin_inp_x)

# #
# ## 2D
# feat = torch.zeros((B, H, W, num_pos_feats))
# embed = (torch.arange(num_pos_feats, dtype=torch.float32) // 2) * 2 / num_pos_feats
#
# x_embed = torch.arange(H, dtype=torch.float32).reshape(-1, 1).repeat(1, num_pos_feats)
# y_embed = torch.arange(W, dtype=torch.float32).reshape(-1, 1).repeat(1, num_pos_feats)
#
# x_embed = x_embed * embed
# y_embed = y_embed * embed
#
# x_embed = torch.stack([x_embed[:, 0::2].sin(), x_embed[:, 1::2].cos()], dim = 2).flatten(1)
# y_embed = torch.stack([y_embed[:, 0::2].sin(), y_embed[:, 1::2].cos()], dim = 2).flatten(1)







# mask = torch.zeros(50, 704, 256, dtype=torch.bool)
# not_mask = ~mask
# # 计算此时竖直方向上的坐标
# y_embed = not_mask.cumsum(1, dtype=torch.float32)
# # 计算此时水平方向上的坐标
# x_embed = not_mask.cumsum(2, dtype=torch.float32)
#
#
# # self.num_pos_feats = d/2
# dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
# # 计算编码函数中的分母，temperature=10000，2*(dim_t//2)就是公式中的2i，无论此时为2i还是2i+1都不改变分母的值
# dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
#
#
# # 计算水平、竖直方向的位置编码
# pos_x = x_embed[:, :, :, None] / dim_t
# pos_y = y_embed[:, :, :, None] / dim_t
#
# # 将编码分奇偶取出，并交叉合并成一维
# pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
# pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
# # 最后将竖直、水平编码合并
# pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)