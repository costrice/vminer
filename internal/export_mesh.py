"""
This file contains the function to export a mesh from a scene.
References:
    https://github.com/za-cheng/WildLight/blob/main/models/renderer.py
"""

import mcubes
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import trimesh
import xatlas
from tqdm import tqdm

from internal import global_vars, scenes, utils, uv_mapping


def extract_geometry(
    volume_3d: torch.Tensor,
    bbox: torch.Tensor,
    level: float,
):
    """Extract the geometry from a 3D volume.

    Args:
        volume_3d: (x, y, z), the volume samples
        bbox: scene axis-aligned bounding box
        level: the level of the iso-surface to extract

    Returns:
        A tuple of (vertices, faces):

        - vertices: (n_verts, 3), the coordinates of the vertices
            of the mesh in world space
        - faces: (n_faces, 3), the index of the vertices of the
            faces of the mesh
    """
    volume_3d = volume_3d.cpu()
    bbox = bbox.cpu()
    volume_3d_np = volume_3d.numpy()
    voxel_size = list((bbox[1] - bbox[0]) / (np.array(volume_3d.shape) - 1))

    vertices, triangles = mcubes.marching_cubes(volume_3d_np, level)
    # move into aabb
    vertices = bbox[0].numpy() + vertices * voxel_size
    # inverse face orientation
    triangles = triangles[:, ::-1]

    n_verts = vertices.shape[0]
    n_faces = triangles.shape[0]
    print("Number of vertices and faces:", n_verts, n_faces)
    print("vertices min and max:", vertices.min(axis=0), vertices.max(axis=0))

    return vertices, triangles


@torch.no_grad()
def export_mesh(
    scene: scenes.SceneModel,
):
    """Export a mesh from a scene object.

    Args:
        scene: the scene object

    Returns:
        None
    """
    args = global_vars.args
    mesh_dir = global_vars.mesh_dir
    device = global_vars.device

    grid_size = utils.total_num_to_resolution(args.mesh_grid_reso**3, scene.aabb)
    grid_size = tuple(grid_size)
    print(f"Extracting mesh from marching cubes at resolution {grid_size}...")
    scene.eval()
    _, sdf = scene.get_dense_sdf(grid_size)
    vertices, triangles = extract_geometry(
        sdf,
        scene.aabb,
        level=0,
    )

    mesh = trimesh.Trimesh(vertices, triangles)
    vertices, triangles = mesh.vertices, mesh.faces

    print("Is mesh watertight?", mesh.is_watertight)
    print("Mesh euler number:", mesh.euler_number)

    # TODO: make this optional, in case of scenes with multiple objects.
    # mesh_parts = mesh.split()
    # retain only the largest 5 parts
    # mesh_parts = sorted(mesh_parts, key=lambda x: x.vertices.shape[0], reverse=True)
    # mesh_parts.sort(key=lambda x: x.vertices.shape[0], reverse=True)
    # largest 5 parts
    # mesh = trimesh.util.concatenate(mesh_parts[:2])
    # mesh = max(mesh_parts, key=lambda x: x.vertices.shape[0])
    # mesh.fill_holes()

    vertices, triangles = mesh.vertices, mesh.faces

    print("Is mesh watertight?", mesh.is_watertight)
    print("Mesh euler number:", mesh.euler_number)

    orig_save_path = mesh_dir / f"{global_vars.cur_iter}_orig.ply"
    mesh.export(orig_save_path)
    print(f"Original mesh saved to {orig_save_path}.")

    if args.simplify_mesh:
        print("Simplifying mesh...")
        mesh = mesh.as_open3d.simplify_quadric_decimation(
            args.simplify_mesh_target_num_faces,
            (0.5 / args.mesh_grid_reso) ** 2,
        )
        print(
            f"Simplified mesh: {vertices.shape[0]} verts, "
            f"{triangles.shape[0]} faces -> "
            f"{len(mesh.vertices)} verts, "
            f"{len(mesh.triangles)} faces."
        )
        vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        mesh = trimesh.Trimesh(vertices, triangles)

        simplified_save_path = mesh_dir / f"{global_vars.cur_iter}_simplified.ply"
        mesh.export(simplified_save_path)
        print(f"Simplified mesh saved to {simplified_save_path}.")

    if args.mesh_bake_texture:
        print(f"Running UV unwraping (this can take a few minutes) ...")

        vert_xyz_map, vertices, triangles, uv = uv_mapping.generate_uv_map(
            vertices, triangles, min_resolution=args.mesh_texture_reso
        )

        print(
            f"Baking textures at resolution "
            f"{vert_xyz_map.shape[0]} x {vert_xyz_map.shape[1]}."
        )

        utils.write_image(
            mesh_dir / "vert_xyz.exr",
            vert_xyz_map.reshape(vert_xyz_map.shape[0], vert_xyz_map.shape[1], -1),
        )

        uv_mesh_save_path = mesh_dir / f"{global_vars.cur_iter}_uv.obj"
        xatlas.export(str(uv_mesh_save_path), vertices, triangles, uv)
        print(f"Mesh with UV saved to {uv_mesh_save_path}.")

        # bake material
        map_shape = vert_xyz_map.shape[:2]
        n_pixels = map_shape[0] * map_shape[1]
        vert_xyz_map = (
            torch.tensor(vert_xyz_map, dtype=torch.float32).to(device).view(-1, 3)
        )
        material_dims = scene.brdf.material_dims
        materials = {
            k: torch.zeros(n_pixels, v).to(device) for k, v in material_dims.items()
        }
        materials["normal"] = torch.zeros(n_pixels, 3, device=device)

        chunk_size = 2**18  # 262144
        # mask: is not nan
        valid_pixel = ~torch.isnan(vert_xyz_map).any(dim=-1)
        valid_pixel = torch.where(valid_pixel)[0]
        chunk_pixels = torch.split(valid_pixel, chunk_size)

        for pixel_chunk in tqdm(chunk_pixels, desc="Baking materials"):
            pts_xyz = vert_xyz_map[pixel_chunk]  # (n_pts, 3)
            pts_attrs_chunk = scene.field.query_field(
                pts_xyz,
                return_material=True,
                return_sdf_grad=True,
            )
            # normalize normal
            pts_attrs_chunk["normal"] = F.normalize(
                pts_attrs_chunk["sdf_grad"], dim=-1, p=2, eps=1e-8
            )
            materials["normal"][pixel_chunk] = pts_attrs_chunk["normal"]
            # unpack material
            unpacked_mat = scene.unpack_brdf(pts_attrs_chunk["material"])
            for k in unpacked_mat.keys():
                materials[k][pixel_chunk] = unpacked_mat[k]

        # reshape and to numpy
        for k, v in materials.items():
            # fill nan with 0
            v = torch.nan_to_num(v)
            if k in ["albedo", "base_color"]:
                v = utils.linrgb2srgb(v)
            if k == "normal":
                v = v / v.norm(dim=-1, keepdim=True)
                v = (v + 1) / 2
            v = v.view(map_shape[0], map_shape[1], -1).cpu().numpy()
            utils.write_image(mesh_dir / f"{k}.png", v, depth=16)
