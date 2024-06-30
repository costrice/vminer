"""This file contains global variables to refer to across different files."""

import argparse
from pathlib import Path

import torch
from tensorboardX import SummaryWriter

from internal import metrics

cur_iter: int = 0
args: argparse.Namespace = None
training: bool = False
log_dir: Path = None
ckpt_dir: Path = None
visual_dir: Path = None
mesh_dir: Path = None
device: torch.device = None
pbr_enabled: bool = False
need_jittered: bool = False
metric_fn: metrics.MetricsGatherer = None
writer: SummaryWriter = None

all_attr_names = [
    # neural rendered attributes
    "depth",
    "opacity",
    "normal",
    "sample_sdf",
    "sample_sdf_grad",
    "sample_weights",
    "sample_ray_indices",
    "sample_intervals",
    "sample_normal",
    "sample_jittered_normal",
    "nr_rgb_far",
    "nr_rgb_near",
    # material
    "base_color",
    "sample_base_color",
    "sample_jittered_base_color",
    "roughness",
    "sample_roughness",
    "sample_jittered_roughness",
    "specular",  # Disney
    "sample_specular",  # Disney
    "sample_jittered_specular",  # Disney
    "fresnel",  # GGX
    "sample_fresnel",  # GGX
    "sample_jittered_fresnel",  # GGX
    # PBR attributes
    "pbr_rgb_diff_far",
    "pbr_rgb_spec_far",
    "direct_shading_far",
    "visibility_far",
    "indirect_rgb_far",
    "pbr_rgb_diff_near",
    "pbr_rgb_spec_near",
    "direct_shading_near",
    "visibility_near",
    "indirect_rgb_near",
    # aggregated attributes by renderer
    "nr_rgb",  # = ray_nr_rgb_far + ray_nr_rgb_near
    "pbr_rgb_far",  # = ray_pbr_rgb_diff_far + ray_pbr_rgb_spec_far
    "pbr_rgb_near",  # = ray_pbr_rgb_diff_near + ray_pbr_rgb_spec_near
    "pbr_rgb_spec",  # = ray_pbr_rgb_spec_far + ray_pbr_rgb_spec_near
    "pbr_rgb_diff",  # = ray_pbr_rgb_diff_far + ray_pbr_rgb_diff_near
    "pbr_rgb",  # = ray_pbr_rgb_spec + ray_pbr_rgb_diff
]
