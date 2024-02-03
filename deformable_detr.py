import torch

H_ , W_ = 25, 34
reference_points_list = []

# 从0.5到H-0.5采样H个点，W同理 这个操作的目的也就是为了特征图的对齐
ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32),
                                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32))
ref_y = ref_y.reshape(-1)[None] / ( H_)
ref_x = ref_x.reshape(-1)[None] / ( W_)
ref = torch.stack((ref_x, ref_y), -1)
reference_points_list.append(ref)

reference_points = torch.cat(reference_points_list, 1)