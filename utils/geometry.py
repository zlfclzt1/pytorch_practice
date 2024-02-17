import numpy as np
import torch

from einops import rearrange, repeat


def get_angle_of_three_points(p1, p2, p3):
    v12 = p2[:2] - p1[:2]
    v13 = p3[:2] - p1[:2]

    dot = np.dot(v12, v13)
    cross = np.cross(v12, v13)
    theta = np.arctan2(cross, dot)
    return theta/np.pi * 180

def sort_convex_hull(convex_hull):
    center = convex_hull.mean(axis=0)
    center_left = center + np.array([-1.0, 0., 0.])

    angles = np.zeros(convex_hull.shape[0])
    for i, point in enumerate(convex_hull):
        angle = get_angle_of_three_points(center, center_left, point)
        angles[i] = angle
    sorted_idx = np.argsort(angles)
    result = convex_hull[sorted_idx, :]
    return result

def create_frustum(shape, image_size, cam_nums):
    h, w, d = shape          # 3, 8, 22, 64
    imaghe_h, imaghe_w = image_size     # 256, 704
    pixel_size = imaghe_h // h      #32

    height = (torch.arange(h) + 0.5) * pixel_size
    width = (torch.arange(w) + 0.5) * pixel_size
    depth = torch.arange(d) + 1

    height = repeat(height, 'h -> c h w d', c=cam_nums, w=w, d=d)
    width = repeat(width, 'w -> c h w d', c=cam_nums, h=h, d=d)
    depth = repeat(depth, 'd -> c h w d', c=cam_nums, w=w, h=h)

    frustum = torch.stack([height, width, depth], dim=-1)
    return frustum





if __name__ == "__main__":
     # p1 = np.array([2.0, 3.0])
     # p2 = np.array([2.0, 6.0])
     # p3 = np.array([5.0, 3.0])
     #
     # result = get_angle_of_three_points(p2, p1, p3)
    # box: [x, y, z, l, w, h, yaw]
    #  convex_hull = np.array([[2.0, 3.0, 0.0],
    #                          [-3.0, 3.2, 0.0],
    #                          [0.8, 2.5, 0.0],
    #                          [2.3, 2.8, 0.0],
    #                          [3.5, 1.0, 0.0],
    #                          [3.0, -2.0, 0.0],
    #                          [1.0, -1.0, 0.0],
    #                          [-0.6, -4.0, 0.0],
    #                          [-2.6, -2.0, 0.0],
    #                          [-3.6, 1.0, 0.0],])
    #  convex_hull = sort_convex_hull(convex_hull)
     shape = (8, 22, 64)
     image_size = (256, 704)
     cam_nums = 6
     frustum = create_frustum(shape, image_size, cam_nums)
