# This file is used in the docker build process to initialize nerfacc library and
# download pretrained models
import lpips
import torch
import nerfacc

# download pretrained models
lpips.LPIPS(net='vgg')

# compile nerfacc library using example code from
# https://www.nerfacc.com/apis/rendering.html
t_starts = torch.tensor([0.1, 0.2, 0.1, 0.2, 0.3], device="cuda:0")
t_ends = torch.tensor([0.2, 0.3, 0.2, 0.3, 0.4], device="cuda:0")
ray_indices = torch.tensor([0, 0, 1, 1, 1], device="cuda:0")


def rgb_sigma_fn(t_starts, t_ends, ray_indices):
    # This is a dummy function that returns random values.
    rgbs = torch.rand((t_starts.shape[0], 3), device="cuda:0")
    sigmas = torch.rand((t_starts.shape[0],), device="cuda:0")
    return rgbs, sigmas


colors, opacities, depths, extras = nerfacc.rendering(
    t_starts, t_ends, ray_indices, n_rays=2, rgb_sigma_fn=rgb_sigma_fn
)
