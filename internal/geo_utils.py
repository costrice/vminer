"""
This file contains modules for geometry transformation.
The transforming matrix used in this code are assumed to be based on row-major
order. That is, the matrix is used to transform a row vector on the left side
of the matrix.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

eps = 1e-6


def coordinate_system(
        normal: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build an orthonormal basis from the given normals.

    References:
        https://graphics.pixar.com/library/OrthonormalB/paper.pdf

    Args:
        normal: (..., 3), the surface normals

    Returns:
        A tuple containing

        - s (torch.Tensor): (..., 3), the first basis vector
        - t (torch.Tensor): (..., 3), the second basis vector
    """
    x, y, z = normal.unbind(dim=-1)
    # deal with z == 0
    z = z + eps
    sign = torch.sign(z)
    a = -torch.reciprocal(sign + z)
    b = x * y * a
    s = torch.stack(
        [
            1 + sign * x * x * a,
            sign * b,
            -sign * x], dim=-1
    )
    t = torch.stack(
        [
            b,
            sign + y * y * a,
            -y], dim=-1
    )
    
    del x, y, z, sign, a, b
    
    return s, t


class Frame(object):
    """
    Stores a 3D orthonormal coordinate system.
    """
    
    def __init__(
            self,
            normal: torch.Tensor,
            device: torch.device = None,
    ):
        """Build an orthonormal basis from the given normals.

        Args:
            normal: (n_pts, 3), the surface normals
            device: the device to operate on
        """
        if device is None:
            device = normal.device
        normal = F.normalize(normal.to(device), p=2, dim=-1, eps=eps)
        self.n_pts = normal.shape[0]
        s, t = coordinate_system(normal)
        self.s = s.view(self.n_pts, 1, 3)
        self.t = t.view(self.n_pts, 1, 3)
        self.n = normal.view(self.n_pts, 1, 3)
    
    def to_local(self, vec: torch.Tensor) -> torch.Tensor:
        """Convert a vector from world coordinate to local coordinate.

        Args:
            vec: (n_pts, n_dirs, 3), the vector in world coordinate system

        Returns:
            torch.Tensor: (n_pts, n_dirs, 3), the vector in local coordinate system
        """
        # since the basis is orthonormal, we can simply project the vector
        # onto the basis vectors
        assert vec.shape[0] == self.n_pts
        n_dirs = vec.shape[1]
        vec_local = torch.stack(
            [
                torch.sum(self.s * vec, dim=-1),
                torch.sum(self.t * vec, dim=-1),
                torch.sum(self.n * vec, dim=-1),
            ], dim=-1
        )
        return vec_local
    
    def to_world(self, vec_local: torch.Tensor) -> torch.Tensor:
        """
        Convert a vector from local coordinate system to global coordinate
        system.
        Args:
            vec_local: (n_pts, n_dirs, 3), the vector in local coordinate system

        Returns:
            torch.Tensor: (n_pts, n_dirs, 3), the vector in world coordinate
                system
        """
        x, y, z = vec_local.unbind(dim=-1)
        vec = (self.s * x[..., None]
               + self.t * y[..., None]
               + self.n * z[..., None])
        return vec


def look_at(
        origin: torch.Tensor,
        target: torch.Tensor,
        up: torch.Tensor
) -> torch.Tensor:
    """Create a look-at camera transformation in blender coordinate system.

    Args:
        origin: (3, ), the camera origin
        target: (3, ), the target position
        up: (3, ), the up direction

    Returns:
        torch.Tensor: (4, 4), the camera(blender)-to-world transformation
    """
    backward = F.normalize(origin - target, p=2, dim=-1, eps=eps)
    right = F.normalize(torch.cross(up, backward), p=2, dim=-1, eps=eps)
    new_up = torch.cross(backward, right)
    # left = F.normalize(torch.cross(up, dir), p=2, dim=-1, eps=eps)
    # new_up = torch.cross(dir, left)
    
    # 3 x 4
    result = torch.stack([right, new_up, backward, origin], dim=0)
    # 4 x 4
    result = torch.cat(
        [result,
         torch.tensor(
             [0, 0, 0, 1],
             device=result.device
         ).view(4, 1)],
        dim=-1
    )
    
    del backward, right, new_up
    
    return result


def fibonacci_sphere(
        n_samples: int,
        device: torch.device,
):
    """
    Uniformly distribute points on a sphere using the Fibonacci spiral method.

    References:
        https://github.com/Kai-46/PhySG/blob/master/code/model/sg_envmap_material.py
        https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    Args:
        n_samples: the number of samples
        device: the device to put the points on

    Returns:
        torch.Tensor: (n_samples, 3) the sampled points
    """
    points = []
    # phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    # for i in range(n_samples):
    #     z = 1 - (i / float(n_samples - 1)) * 2  # y goes from 1 to -1
    #     radius = np.sqrt(1 - z * z)  # radius at y
    #
    #     theta = phi * i  # golden angle increment
    #
    #     x = np.cos(theta) * radius
    #     y = np.sin(theta) * radius
    #
    #     points.append([x, y, z])
    # points = np.array(points)
    
    # torch version
    # golden angle in radians
    phi = torch.pi * (3. - torch.sqrt(torch.tensor(5., device=device)))
    # z goes from 1 to -1
    z = 1 - (torch.arange(n_samples, device=device) / float(n_samples - 1)) * 2
    radius = torch.sqrt(1 - z ** 2)
    # golden angle increment
    theta = phi * torch.arange(n_samples, device=device)
    x = torch.cos(theta) * radius
    y = torch.sin(theta) * radius
    points = torch.stack([x, y, z], dim=-1)
    
    return points


# # camera: z: forward; y: up; x: right
# # world: z: up; y: left; x: forward
# transform_mat_c2w = torch.tensor(
#     [[0, -1, 0],
#      [0, 0, 1],
#      [1, 0, 0]],
#     dtype=torch.float32,
#     device=torch.device('cuda:0'))
# transform_mat_w2c = torch.linalg.inv(transform_mat_c2w)


def image_coord2view_vec(coord: torch.Tensor):
    """Convert 2D coordinates on a panorama image to a 3D unit view vector in
    world coordinate.

    Coordinate system:
        3D (blender): x: backward, y: right, z: up.
        2D (opencv): theta: down, phi: right. 0, 0 is at the center of the
            image, corresponding to the view direction (1, 0, 0)

    Args:
        coord: (..., 2), the [h, w] coordinates on the image
            in [-1, 1]

    Returns:
        view (torch.Tensor): (..., 3), the unit view vector.
    """
    # convert to polar coordinates
    # theta: [-pi/2, pi/2], phi: [-pi, pi]
    theta = coord[..., 0] * np.pi / 2
    phi = coord[..., 1] * np.pi
    # convert to view vector
    x = torch.cos(theta) * torch.cos(phi)
    y = -torch.cos(theta) * torch.sin(phi)  # the envmap is horizontal flipped
    z = -torch.sin(theta)
    
    view = torch.stack([x, y, z], dim=-1)
    return view


def view_vec2image_coord(view_w: torch.Tensor):
    """Convert a 3D unit view vector in world coordinate to 2D coordinates on a
    panorama image.

    Coordinate system:
        See image_coord2view_vec().

    Args:
        view_w: (..., 3), the view vector in world coordinate

    Returns:
        coord (torch.Tensor): (..., 2), the [h, w] coordinates on the image
        in [-1, 1]
    """
    x, y, z = view_w[..., 0], view_w[..., 1], view_w[..., 2]
    # convert to polar coordinates
    # theta: [-pi/2, pi/2], phi: [-pi, pi]
    theta = torch.asin(-z)
    phi = torch.atan2(-y, x)
    # convert to image coordinates
    coord = torch.stack(
        [
            theta / np.pi * 2,
            phi / np.pi], dim=-1
    )
    
    return coord


def get_intrinsic_matrix(
        img_h: int,
        img_w: int,
        focal: float,
        device: torch.device,
):
    """Compute camera intrinsic matrix from 3D opencv coordinate to 2D image
    coordinate. For row-major vector.

    Opencv coordinate system:
        x: right, y: down, z: forward

    Args:
        img_h: Image height.
        img_w: Image width.
        focal: Focal length of the camera (in pixels).
        device: Device to put the matrix on.

    Returns:
        torch.Tensor: (3, 3) Intrinsic matrix from 3D points in camera space
        (opencv coordinate system) to 2D homogeneous image coordinates.
    """
    # matrix in row-major order
    int_mat_c2i = torch.tensor(
        [[focal, 0, 0],
         [0, focal, 0],
         [img_w / 2, img_h / 2, 1]],
        dtype=torch.float32, device=device
    )
    return int_mat_c2i


def get_homo_image_coord(
        img_h: int,
        img_w: int,
        device: torch.device
) -> torch.Tensor:
    """Get homogeneous image coordinates (x, y, 1) for all pixels.

    Args:
        img_h: Image height
        img_w: Image width
        device: The device to put the output on

    Returns:
        torch.Tensor: (H, W, 3) containing the homogeneous image coordinates:
            (x, y, 1)
    """
    x, y = torch.meshgrid(
        [
            torch.linspace(0.5, img_w - 0.5, img_w, device=device),
            torch.linspace(0.5, img_h - 0.5, img_h, device=device)
        ], indexing='xy'
    )
    z = torch.ones_like(x)
    
    return torch.stack([x, y, z], dim=-1)


def get_camera_ray_directions(
        H: int,
        W: int,
        focal: float,
        device: torch.device
) -> torch.Tensor:
    """
    Get ray directions for all pixels in camera coordinate.

    Args:
        H: image height
        W: image width
        focal: focal length in pixels (assuming pixel is square)
        device: the device to put the output on

    Returns:
        torch.Tensor: (H, W, 3), the direction of the rays in camera space,
        in opencv coordinate system:
            x: right, y: down, z: forward
    """
    # (H, W, 3) containing [x, y, 1]
    img_coord_homo = get_homo_image_coord(
        H, W, device
    )
    
    # get intrinsic matrix from 3D camera space (opencv coordinate) to 2D image
    # opencv coordinate, x: right, y: down, z: forward
    int_mat_c2i = get_intrinsic_matrix(H, W, focal, device)
    int_mat_i2c = torch.inverse(int_mat_c2i)
    directions = img_coord_homo @ int_mat_i2c
    
    return directions


def camera_ray_to_world(
        directions: torch.Tensor,
        ext_c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get ray origin and normalized directions in world coordinate for input
    camera coordinate rays.

    References:
        https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/standard-coordinate-systems

    Args:
        directions: (H, W, 3) precomputed ray directions in camera space,
            opencv coord system.
        ext_c2w: (4, 4) the inverse extrinsic camera matrix, from camera
            space (opencv coord) to world coordinate. The matrix should be:
            [R | 0
             T | 1]

    Returns:
        Tuple containing

        - rays_o (torch.Tensor): (H * W, 3) ray origins in world coordinate
        - rays_d (torch.Tensor): (H * W, 3) normalized ray directions in world
            coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ ext_c2w[:3, :3]  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = ext_c2w[3, :3].view(1, 1, 3).expand(rays_d.shape)  # (H, W, 3)
    
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    rays_d = F.normalize(rays_d, p=2, dim=-1, eps=eps)
    
    return rays_o, rays_d
