"""
This file is used to relight the images in the test dataset using
Blender.
It contains the blender script to render images from all view points
read from the metadata file in a test dataset. The scene should be
set up in blender before running this script, using the recovered
mesh and textures by the method. The code should be run in blender.
"""
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

project_dir = r'E:\Codes\VMINer'
sys.path.append(project_dir)

import numpy as np
import json
from tqdm import tqdm
# from internal import utils

# python API for blender
import bpy
from mathutils import Matrix

D = bpy.data
C = bpy.context


def parse_pose_from_meta(
        meta: Dict[str, Any],
):
    """Parse the metadata dictionary to get the transformation matrices and
    light information.

    The transformation matrices are in the row-major order (i.e. should be put
    on the right of a row vector in multiplication). They transform between 3D
    homogeneous point in world space, in camera space (opencv coordinate), and
    2D homogeneous pixel location.

    Args:
        meta: the metadata dictionary for an image.

    Returns:
        A tuple of transformation matrices:

        - int_mat_i2c (torch.Tensor): 3x3 intrinsic matrix from 2D homogeneous
            pixel location to 3D point in camera space (opencv
            coordinate).
        - ext_mat_c2w (torch.Tensor): 4x4 extrinsic matrix from 3D homogeneous
            point in camera space (opencv coordinate) to 3D point homogeneous
            in world space.
    """
    img_hw = (int(meta['imh']), int(meta['imw']))
    
    # Get ray directions for all pixels, same for all images
    # (with same H, W, focal)
    focal = 0.5 * int(meta['imw']) / np.tan(
        0.5 * meta['cam_angle_x']
    )  # fov -> focal length
    focal *= img_hw[1] / meta['imw']  # account for possible downsample
    
    # matrix in row-major order
    # int_mat_c2i = torch.tensor(
    #     [[focal, 0, 0],
    #      [0, focal, 0],
    #      [img_w / 2, img_h / 2, 1]],
    #     dtype=torch.float32, device=device)
    # numpy version
    int_mat_c2i = np.array(
        [[focal, 0, 0],
         [0, focal, 0],
         [img_hw[1] / 2, img_hw[0] / 2, 1]],
        dtype=np.float32
    )
    
    # read extrinsic matrix
    ext_mat_c2w = np.array(meta["cam_transform_mat"]).reshape(4, 4)
    ext_mat_c2w = ext_mat_c2w.T  # from column-major to row-major
    
    return int_mat_c2i, ext_mat_c2w


if __name__ == '__main__':
    test_img_dir = Path(
        r'F:\Datasets\VMINer_Synthetic\hotdog_test\test'
    )
    split = 'test'
    scene_name = 'hotdog'
    base_res_dir = Path(
        r'E:\Codes\VMINer\comparison_results'
    )
    scene_res_dir = base_res_dir / scene_name
    method_name = 'nearlight_long'
    out_dir = scene_res_dir / method_name
    out_dir.mkdir(exist_ok=True, parents=True)
    tmp_dir = out_dir / 'tmp'
    
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(exist_ok=True, parents=True)
    
    # get blender objects
    camera = D.objects['Camera']
    obj = D.objects['object']
    # set composite nodes
    for node in C.scene.node_tree.nodes:
        if node.type == 'OUTPUT_FILE':
            node.base_path = str(tmp_dir)
    
    # read metadata from the test dataset
    img_folders = [x for x in test_img_dir.iterdir() if
                   x.stem.startswith(split + '_')]
    img_folders.sort()
    # img_folders = img_folders[:10]
    
    for img_id, img_folder in tqdm(
            enumerate(img_folders),
            total=len(img_folders)
    ):
        # read metadata
        meta_path = img_folder / 'metadata.json'
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        int_mat_i2c, ext_mat_c2w = parse_pose_from_meta(meta)
        # row-major to column-major
        int_mat_i2c, ext_mat_c2w = int_mat_i2c.T, ext_mat_c2w.T
        
        print(
            f'Rendering image {img_id} with intrinsic matrix:\n{int_mat_i2c}'
            f'\nand extrinsic matrix:\n{ext_mat_c2w}'
        )
        
        # set camera intrinsic according to int_mat_i2c
        focal_pixel_x = int_mat_i2c[0, 0]
        img_width = round(int_mat_i2c[0, 2] * 2)
        img_height = round(int_mat_i2c[1, 2] * 2)
        focal_length_mm = focal_pixel_x / img_width * camera.data.sensor_width
        focal_length_mm = round(focal_length_mm)
        
        C.scene.render.resolution_x = img_width
        C.scene.render.resolution_y = img_height
        camera.data.lens = focal_length_mm
        print(f'Image size set to {img_width} x {img_height} pixels')
        print(f'Camera focal length set to {focal_length_mm} mm')
        
        # set camera extrinsic according to ext_mat_c2w
        
        # opencv2blender = np.array(
        #     [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        #     dtype=np.float32)
        camera.matrix_world = Matrix(ext_mat_c2w)
        C.view_layer.update()
        
        rendered_img_path = out_dir / f'rgb_{img_id:03d}.png'
        
        # render image
        C.scene.use_nodes = True
        C.scene.render.filepath = str(rendered_img_path)
        C.scene.view_settings.view_transform = 'Standard'
        C.scene.render.film_transparent = True
        bpy.ops.render.render(write_still=True)
        
        # move images in the tmp dir:
        # DiffCol_, NormalW_, Diffuse_, Specular_
        shutil.copy(
            tmp_dir / 'DiffCol_0000.png',
            out_dir / f'albedo_{img_id:03d}.png'
        )
        shutil.copy(
            tmp_dir / 'NormalW_0000.png',
            out_dir / f'normal_{img_id:03d}.png'
        )
        shutil.copy(
            tmp_dir / 'Diffuse_0000.png',
            out_dir / f'rgb_diff_{img_id:03d}.png'
        )
        shutil.copy(
            tmp_dir / 'Specular_0000.png',
            out_dir / f'rgb_spec_{img_id:03d}.png'
        )
    
    print('Done!')
    # clean up
    shutil.rmtree(tmp_dir)
