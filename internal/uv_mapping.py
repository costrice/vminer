"""
This file contains function for uv unwrapping and texture mapping.
References:
    https://github.com/za-cheng/WildLight/blob/main/models/uv_mapping.py
"""
import math
import os
from typing import Tuple, Union

import cv2
import numpy as np
import scipy
import scipy.interpolate
import xatlas
from tqdm import tqdm

# enable openexr
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


def parametrize(
        vertices: np.ndarray,
        faces: np.ndarray,
        padding: int,
):
    """Use xatlas to parametrize a mesh.

    Args:
        vertices: (n_verts, 3), the coordinates of the vertices of the mesh
        faces: (n_faces, 3), the index of the vertices of the faces of the mesh
        padding: the padding of the atlas

    Returns:
        A tuple of (verts, triangles, uv, resolution):

        - verts (np.ndarray): (n_verts, 3), the coordinates of the vertices of
            the new mesh
        - triangles (np.ndarray): (n_faces, 3), the index of the vertices of the
            faces of the new mesh
        - uv (np.ndarray): (n_verts, 2), the uv coordinates of the vertices of
            the new mesh
        - resolution (tuple): (height, width), the resolution of the atlas
    """
    print('Parametrizing...')
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    pack_options.padding = padding
    pack_options.create_image = True
    atlas.generate(chart_options, pack_options, False)
    vmapping, triangles, uv = atlas[0]
    resolution = atlas.chart_image.shape[:2]
    return vertices[vmapping], triangles, uv, resolution


def ij_to_uv(
        ij: np.ndarray,
        height: float,
        width: float
) -> np.ndarray:
    """Map ij coordinates to uv coordinates.

    ij coordinates are in [0, height-1] x [0, width-1]. uv coordinates are
    in [0, 1] x [0, 1]. The origin of the uv coordinates is in the bottom left
    corner, and u is the horizontal axis and v is the vertical axis.

    Args:
        ij: (..., 2), the ij coordinates
        height: the height of the atlas
        width: the width of the atlas

    Returns:
        np.ndarray: (..., 2), the uv coordinates
    """
    i, j = ij[..., 0], ij[..., 1]
    u = j / (width - 1)
    v = 1 - i / (height - 1)
    return np.stack([u, v], axis=-1)


def uv_to_ij(
        uv: np.ndarray,
        height: float,
        width: float
) -> np.ndarray:
    """Map uv coordinates to ij coordinates."""
    u, v = uv[..., 0] % 1, uv[..., 1] % 1
    j = u * (width - 1)
    i = (1 - v) * (height - 1)
    return np.stack([i, j], axis=-1)


def break_up_triangles(
        verts: np.ndarray
):
    """Break up a triangle into triangles with horizontal top or
    bottom edges.

    Args:
        verts: (3, 2), the ij coordinates of the vertices of the triangle

    Returns:
        A tuple of 0, 1, or 2 tuples of (is_upper, up, down, x, left, right):

        - is_upper (bool): whether the triangle is an upper triangle, i.e. the
        bottom is a straight line instead of a vertex
        - up (float): the upper bound of the triangle
        - down (float): the lower bound of the triangle
        - x (float): the x coordinate of the top vertex or the bottom vertex
        - left (float): the left bound of the triangle
        - right (float): the right bound of the triangle
    """
    a, b, c = verts[np.argsort(verts[:, 0])]
    if a[0] < b[0] < c[0]:
        d = (b[0],
             (c[1] * (b[0] - a[0]) + a[1] * (c[0] - b[0])) / (c[0] - a[0]))
        return (break_up_upper_triangles(a, b, d),
                break_up_lower_triangles(c, b, d))
    elif a[0] < b[0]:
        return (break_up_upper_triangles(a, b, c),)
    elif b[0] < c[0]:
        return (break_up_lower_triangles(c, a, b),)
    else:
        return ()


def break_up_upper_triangles(
        top: np.ndarray,
        left: np.ndarray,
        right: Union[np.ndarray, Tuple[float, float]],
):
    """Get attributes of an upper triangle (see rasterize_triangles).

    Args:
        top: (2, ) the ij coordinates of the top vertex
        left: (2, ) the ij coordinates of the bottom left vertex
        right: (2, ) the ij coordinates of the bottom right vertex

    Returns:
        A tuple of (is_upper, up, down, x, left, right). See
        rasterize_triangles for the definition of these attributes.
    """
    assert left[0] == right[0]
    assert top[0] < left[0]
    if left[1] > right[1]:
        tmp = left
        left = right
        right = tmp
    return True, top[0], left[0], top[1], left[1], right[1]


def break_up_lower_triangles(
        down: np.ndarray,
        left: np.ndarray,
        right: Union[np.ndarray, Tuple[float, float]],
):
    """Get attributes of a lower triangle (see rasterize_triangles).

    See Also:
        break_up_upper_triangles
    """
    assert left[0] == right[0]
    assert down[0] > left[0]
    if left[1] > right[1]:
        tmp = left
        left = right
        right = tmp
    return False, left[0], down[0], down[1], left[1], right[1]


def rasterize_triangles(
        atlas_size: Tuple[int, int],
        triangles: np.ndarray,
        uv: np.ndarray
):
    """Rasterize triangles onto an atlas.

    Args:
        atlas_size: the height and width of the atlas
        triangles: (n_triangles, 3), the index of the vertices of the triangles
        uv: (n_verts, 2), the uv coordinates of the vertices of the mesh

    Returns:
        np.ndarray: (height, width), the triangle id at each pixel
    """
    height, width = atlas_size
    atlas = np.zeros(atlas_size, dtype=np.int64) - 1
    ij = uv_to_ij(uv, height, width)
    for triangle_id, triangle in enumerate(
            tqdm(
                    triangles, desc='Rasterize'
            )
    ):
        verts_ij = ij[triangle]
        for is_upper, up, down, x, left, right in break_up_triangles(verts_ij):
            if is_upper:
                rasterize_upper_triangle(
                    atlas, up, down, x, left, right,
                    triangle_id
                )
            else:
                rasterize_lower_triangle(
                    atlas, up, down, x, left, right,
                    triangle_id
                )
    return atlas


def rasterize_upper_triangle(
        atlas: np.ndarray,
        up: float,
        down: float,
        x: float,
        left: float,
        right: float,
        triangle_id: int
):
    """Rasterize an upper triangle onto an atlas.

    Args:
        atlas: (height, width), the atlas to rasterize the triangle onto
        up: the upper bound of the triangle
        down: the lower bound of the triangle
        x: the x coordinate of the top vertex
        left: the left bound of the triangle
        right: the right bound of the triangle
        triangle_id: the id of the triangle

    Returns:
        None
    """
    i_up = math.ceil(up)
    i_down = math.floor(down)
    width = right - left
    for i in range(i_up, i_down + 1):
        ratio = (i - up) / (down - up)
        i_left = x + (left - x) * ratio
        i_right = x + (right - x) * ratio
        for j in range(math.ceil(i_left), math.floor(i_right) + 1):
            atlas[i, j] = triangle_id


def rasterize_lower_triangle(
        atlas,
        up: float,
        down: float,
        x: float,
        left: float,
        right: float,
        triangle_id: int
):
    """Rasterize a lower triangle onto an atlas.

    See Also:
        rasterize_upper_triangle
    """
    i_up = math.ceil(up)
    i_down = math.floor(down)
    width = right - left
    for i in range(i_up, i_down + 1):
        ratio = (down - i) / (down - up)
        i_left = x + (left - x) * ratio
        i_right = x + (right - x) * ratio
        for j in range(math.ceil(i_left), math.floor(i_right) + 1):
            atlas[i, j] = triangle_id


def cartesian_2_barycentric(
        p: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray
) -> np.ndarray:
    """Convert cartesian coordinates to barycentric coordinates.

    Args:
        p: (..., ?), a point in the plane in cartesian coordinates
        a: (..., ?), the first vertex of the triangle in cartesian coordinates
        b: (..., ?), the second vertex of the triangle in cartesian coordinates
        c: (..., ?), the third vertex of the triangle in cartesian coordinates

    Returns:
        np.ndarray: (..., 3), the barycentric coordinates of the point
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = (v0 * v0).sum(axis=-1)
    d01 = (v0 * v1).sum(axis=-1)
    d11 = (v1 * v1).sum(axis=-1)
    d20 = (v2 * v0).sum(axis=-1)
    d21 = (v2 * v1).sum(axis=-1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    
    bary = np.stack([u, v, w], axis=-1)
    return bary


def barycentric2_cartesian(
        p: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray
):
    """Convert barycentric coordinates to cartesian coordinates.

    Args:
        p: (..., 3), a point in the plane in barycentric coordinates
        a: (..., ?), the first vertex of the triangle in cartesian coordinates
        b: (..., ?), the second vertex of the triangle in cartesian coordinates
        c: (..., ?), the third vertex of the triangle in cartesian coordinates

    Returns:
        np.ndarray: (..., ?), the cartesian coordinates of the point
    """
    u, v, w = p[..., 0], p[..., 1], p[..., 2]
    return u[..., None] * a + v[..., None] * b + w[..., None] * c


def mask_edge_out(
        mask,
        padding=1
):
    """Extract the edge of a mask.

    Args:
        mask: (height, width), the mask
        padding: the padding of the dilation

    Returns:
        np.ndarray: (height, width), the edge of the mask
    """
    kernel = np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8)
    dilated_mask = cv2.dilate(
        mask.astype(np.uint8), kernel, iterations=1
    ).astype(np.bool8)
    return np.logical_xor(mask, dilated_mask)


def mask_edge_in(
        mask,
        padding=1
):
    """Extract the edge of a mask within the mask.

    Args:
        mask: (height, width), the mask
        padding: the padding of the dilation

    Returns:
        np.ndarray: (height, width), the edge of the mask
    """
    kernel = np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8)
    eroded_mask = cv2.erode(
        mask.astype(np.uint8), kernel, iterations=1
    ).astype(np.bool8)
    return np.logical_xor(mask, eroded_mask)


def generate_uv_map(
        vertices: np.ndarray,
        faces: np.ndarray,
        min_resolution: int,
        padding: int = 1,
):
    """Generate a UV map for a mesh.

    Args:
        vertices: (n_verts, 3), the coordinates of the vertices of the mesh
        faces: (n_faces, 3), the index of the vertices of the faces of the mesh
        min_resolution: the minimum resolution of the atlas
        padding: the padding of the atlas

    Returns:
        A tuple of (coordinates_atlas, verts, triangles, uvs):

        - coordinates_atlas (np.ndarray): (height, width, 3), the coordinates of
            the vertices of the mesh in the atlas
        - verts (np.ndarray): (n_verts, 3), the coordinates of the vertices of
            the mesh
        - triangles (np.ndarray): (n_faces, 3), the index of the vertices of the
            faces of the mesh
        - uvs (np.ndarray): (n_verts, 2), the uv coordinates of the vertices of
            the mesh
    """
    # parametrize the mesh using xatlas library
    verts, triangles, uvs, resolution = parametrize(
        vertices, faces, padding * 2
    )
    # resize the atlas to the minimum resolution
    resize_ratio = max(min_resolution, min(resolution)) / min(resolution)
    resolution = math.ceil(resolution[0] * resize_ratio), math.ceil(
        resolution[1] * resize_ratio
    )
    
    coordinates_atlas = np.zeros(resolution + (3,))
    ij = np.stack(
        np.meshgrid(
            np.arange(resolution[0]),
            np.arange(resolution[1]),
            indexing='ij'
        ),
        axis=-1
    )
    uv_atlas = ij_to_uv(
        ij.astype(np.float64), resolution[0], resolution[1]
    )
    
    triangle_id_atlas = rasterize_triangles(
        resolution,
        triangles,
        uvs
    )
    atlas_mask = triangle_id_atlas >= 0
    
    tri_ids = triangle_id_atlas[atlas_mask]
    verts_ids = triangles[tri_ids]
    bary_coords = cartesian_2_barycentric(
        uv_atlas[atlas_mask],
        uvs[verts_ids[:, 0]],
        uvs[verts_ids[:, 1]],
        uvs[verts_ids[:, 2]]
    )
    
    coordinates_atlas[atlas_mask] = barycentric2_cartesian(
        bary_coords,
        verts[verts_ids[:, 0]],
        verts[verts_ids[:, 1]],
        verts[verts_ids[:, 2]]
    )
    
    # extrapolate the coordinates at the edge of the mask
    print('Extrapolating mask edge. This may take a while...')
    interp_mask = mask_edge_out(atlas_mask, padding)
    edge_within = mask_edge_in(atlas_mask, padding)
    interp = scipy.interpolate.LinearNDInterpolator(
        ij[edge_within], coordinates_atlas[edge_within], 0
    )
    # interp = scipy.interpolate.LinearNDInterpolator(
    #     ij[atlas_mask], coordinates_atlas[atlas_mask], 0)
    coordinates_atlas[interp_mask] = interp(ij[interp_mask])
    
    coordinates_atlas[np.logical_not(interp_mask | atlas_mask)] = np.nan
    
    return coordinates_atlas, verts, triangles, uvs
