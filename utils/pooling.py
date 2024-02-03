import torch
import torch.nn as nn


class MaxPooling2d(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        super(MaxPooling2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        assert len(x.shape) == 4
        N, C, H, W = x.shape
        device = x.device

        if self.padding > 0:
            padded_x = torch.zeros((N, C, H + 2 * self.padding, W + 2 * self.padding), device=device)
            padded_x[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
        else:
            padded_x = x

        output_H = ((H + 2 * self.padding - self.kernel_size) // self.stride) + 1
        output_W = ((W + 2 * self.padding - self.kernel_size) // self.stride) + 1

        output = torch.zeros((N, C, output_H, output_W), device=device)

        for i in range(output_H):
            for j in range(output_W):
                i_start = i * self.stride
                i_end = i * self.stride + self.kernel_size
                j_start = j * self.stride
                j_end = j * self.stride + self.kernel_size
                cur_feat = padded_x[:, :, i_start:i_end, j_start:j_end]
                cur_feat = cur_feat.reshape(N, C, -1)
                output[:, :, i, j] = torch.max(cur_feat, dim=2)[0]
        return output


class AvgPooling2d(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(AvgPooling2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        assert len(x.shape) == 4
        N, C, H, W = x.shape
        device = x.device

        if self.padding > 0:
            padded_x = torch.zeros((N, C, H + 2 * self.padding, W + 2 * self.padding), device=device)
            padded_x[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
        else:
            padded_x = x

        output_H = ((H + 2 * self.padding - self.kernel_size) // self.stride) + 1
        output_W = ((W + 2 * self.padding - self.kernel_size) // self.stride) + 1

        output = torch.zeros((N, C, output_H, output_W), device=device)

        for i in range(output_H):
            for j in range(output_W):
                i_start = i * self.stride
                i_end = i * self.stride + self.kernel_size
                j_start = j * self.stride
                j_end = j * self.stride + self.kernel_size
                cur_feat = padded_x[:, :, i_start:i_end, j_start:j_end]
                cur_feat = cur_feat.reshape(N, C, -1)
                output[:, :, i, j] = torch.mean(cur_feat, dim=2)[0]
        return output


# class AdaptiveAvgPool2d(nn.Module):
#     def __init__(self, output_size=[1, 1]):
#         super(AdaptiveAvgPool2d, self).__init__()
#
#         assert isinstance(output_size, int) or isinstance(output_size, list)
#         if isinstance(output_size, int):
#             output_size = [output_size, output_size]
#         self.output_size = output_size
#
#     def forward(self, x):
#         assert len(x.shape) == 4
#         N, C, H, W = x.shape
#
#         padding = 0
#         stride = [H // self.output_size[0], W // self.output_size[1]]
#         kernel_size =


if __name__ == "__main__":
    input = torch.randn((32, 3, 64, 64))
    max_pooling = MaxPooling2d(kernel_size=3, stride=2, padding=1)
    y = max_pooling(input)
    print("input.shape: ", input.shape)
    print(f"output.shape: {y.shape}")
