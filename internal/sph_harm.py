"""This file contains functions to evaluate spherical harmonic bases and evaluate spherical harmonics."""

import torch

from internal.rsh import (
    rsh_cart_0,
    rsh_cart_1,
    rsh_cart_2,
    rsh_cart_3,
    rsh_cart_4,
    rsh_cart_5,
)

real_sh_cart_fn = [  # protect from import optimization
    rsh_cart_0,
    rsh_cart_1,
    rsh_cart_2,
    rsh_cart_3,
    rsh_cart_4,
    rsh_cart_5,
]


def eval_sh_bases(deg: int, dirs: torch.Tensor):
    """
    Evaluate the spherical harmonic bases for a given degree.

    Args:
        deg (int): The degree of the spherical harmonic bases. Must be between 0 and 5 (inclusive).
        dirs (torch.Tensor): The tensor containing the directions at which to evaluate the bases.

    Returns:
        torch.Tensor: The evaluated spherical harmonic bases.

    Raises:
        AssertionError: If the degree is not within the supported range.

    """
    assert deg <= 5 and deg >= 0, f"Unsupported SH degree: {deg}"
    sh_bases = real_sh_cart_fn[deg](dirs)
    return sh_bases


def eval_sh(
    deg: int,
    dirs: torch.Tensor,
    sh_weights: torch.Tensor,
):
    """
    Evaluate spherical harmonics bases at unit directions, without taking linear
    combination. At each point, the final result may be obtained through simple
    multiplication.

    Args:
        deg (int): SH max degree. Currently support up to 5.
        dirs (torch.Tensor): (..., 3) unit directions.
        sh_weights (torch.Tensor): (..., c, (deg+1) ** 2) SH weights.

    Returns:
        torch.Tensor: (..., c) combined SH values.

    Raises:
        AssertionError: If the shape of `dirs` or `sh_weights` is invalid.
    """
    assert (
        len(dirs.shape) >= 2 and dirs.shape[-1] == 3
    ), f"Invalid dirs shape: {dirs.shape}"
    assert (
        len(sh_weights.shape) == len(dirs.shape) + 1
        and sh_weights.shape[-1] == (deg + 1) ** 2
        and sh_weights.shape[:-2] == dirs.shape[:-1]
    ), (
        f"Invalid sh_weights shape: {sh_weights.shape}, the shape should "
        f"match ({dirs.shape[:-1]}, c, {(deg + 1) ** 2})."
    )

    sh_bases = eval_sh_bases(deg, dirs).unsqueeze(-2)
    return torch.sum(sh_weights * sh_bases, dim=-1)


if __name__ == "__main__":
    shape = (12, 11, 13)

    # Generate random points on sphere.
    theta = torch.rand(shape) * torch.pi
    phi = torch.rand(shape) * 2 * torch.pi

    # Convert to Cartesian coordinates.
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    xyz = torch.stack([x, y, z], dim=-1)

    view_sh_deg = 4
    sh_weights = torch.rand(shape + (1, (view_sh_deg + 1) ** 2))
    sh_evals = eval_sh(view_sh_deg, xyz, sh_weights)
