"""Convert dataset convention from NeRF (and Ours) to NeuS (and WildLight)."""
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from internal import datasets, utils

all_dataset_dir = Path(r'F:\Datasets')
vminer_syn_dir = all_dataset_dir / 'VMINer_Synthetic'
vminer_real_dir = all_dataset_dir / 'VMINer_Real'
wildlight_syn_dir = all_dataset_dir / 'WildLight_Synthetic'
wildlight_real_dir = all_dataset_dir / 'WildLight_Real'
tensoir_dir = all_dataset_dir / 'TensoIR_Synthetic'
nerf_syn_dir = all_dataset_dir / 'NeRF_Synthetic'
nerf_real_dir = all_dataset_dir / 'NeRF_Real'
real_dir = all_dataset_dir / 'Unformatted_Real'

# our_wildlight_meta_path = \
#     r"F:\Datasets\WildLight_Synthetic\hotdog\cameras_sphere.npz"
# our_wildlight_meta = np.load(our_wildlight_meta_path)
#
# example_wildlight_meta_path = \
#     r"E:\Codes\FromGithub\WildLight\datasets\synthetic\bunny\cameras_sphere.npz"
# example_wildlight_meta = np.load(example_wildlight_meta_path)

scale_mat = np.array(
    [[2.5, 0.0, 0.0, 0.0],
     [0.0, 2.5, 0.0, 0.0],
     [0.0, 0.0, 2.5, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32
)


def convert_nearlight_to_wildlight(
        dataset_name: str,
        split: str,
        nearlight_dir: Path = vminer_syn_dir,
        wildlight_dir: Path = wildlight_syn_dir,
):
    source_dataset_dir = nearlight_dir / dataset_name / split
    img_folders = [x for x in source_dataset_dir.iterdir() if
                   x.stem.startswith(split + '_')]
    img_folders.sort()
    # img_folders = img_folders[::10]
    
    dataset_stem = dataset_name.split('_')[0]
    target_dataset_dir = wildlight_dir / dataset_stem
    if split == 'test':
        target_dataset_dir = target_dataset_dir / 'val'
    target_image_dir = target_dataset_dir / 'image'
    target_mask_dir = target_dataset_dir / 'mask'
    target_image_dir.mkdir(parents=True, exist_ok=True)
    target_mask_dir.mkdir(parents=True, exist_ok=True)
    
    total_dict = {}
    
    # source format is 'nearlight'
    for img_id, img_folder in tqdm(
            enumerate(img_folders),
            total=len(img_folders),
            desc=f'Converting {dataset_name} {split}'
    ):
        # read image, split it into RGB and alpha. Save as .EXR linear image
        img_path = img_folder / 'rgba.png'
        img = utils.read_image(img_path)
        alpha = img[..., -1]
        rgb = img[..., :3] * alpha[..., None]
        # to linear
        rgb = utils.srgb2linrgb(rgb)
        # save
        rgb_path = target_image_dir / f'{img_id:03d}.png'
        alpha_path = target_mask_dir / f'{img_id:03d}.png'
        utils.write_image(rgb_path, rgb, depth=16)
        utils.write_image(alpha_path, alpha)
        
        # convert metadata
        meta_path = img_folder / 'metadata.json'
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        int_mat_i2c, ext_mat_c2w = datasets.parse_pose_from_meta(
            meta, 'nearlight', device=torch.device('cpu')
        )
        
        # construct target format: 'wildlight'
        wildlight_meta_this = datasets.pose_to_meta(
            int_mat_i2c, ext_mat_c2w,
            convention='wildlight'
        )
        
        # check if the conversion is correct
        int_mat_i2c_2, ext_mat_c2w_2 = datasets.parse_pose_from_meta(
            wildlight_meta_this, 'wildlight', device=torch.device('cpu')
        )
        assert torch.allclose(int_mat_i2c, int_mat_i2c_2, atol=1e-5)
        assert torch.allclose(ext_mat_c2w, ext_mat_c2w_2, atol=1e-5)
        
        # read near light info
        near_light_state = meta['near_light_status']
        assert len(near_light_state) == 1
        near_light_state = near_light_state * 3  # to [R, G, B]
        light_energy = np.array(near_light_state, dtype=np.float32)
        
        # add to total dict
        total_dict[f'world_mat_{img_id}'] = wildlight_meta_this['world_mat']
        total_dict[f'scale_mat_{img_id}'] = scale_mat
        total_dict[f'light_energy_{img_id}'] = light_energy
    
    # save total dict
    total_dict_path = target_dataset_dir / 'cameras_sphere.npz'
    np.savez(total_dict_path, **total_dict)


def convert_nearlight_to_tensoir(
        dataset_name: str,
        split: str,
        nearlight_dir: Path = vminer_syn_dir,
        tensoir_dir: Path = tensoir_dir,
):
    source_dataset_dir = nearlight_dir / dataset_name / split
    img_folders = [x for x in source_dataset_dir.iterdir() if
                   x.stem.startswith(split + '_')]
    img_folders.sort()
    # img_folders = img_folders[:10]
    
    target_dataset_dir = tensoir_dir / dataset_name
    target_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # read light metadata
    light_meta_path = source_dataset_dir / f'light_metadata_{split}.json'
    with open(light_meta_path, 'r') as f:
        light_meta = json.load(f)
    far_light_names = light_meta['far_lights']['name']
    # copy light metadata
    shutil.copy(
        light_meta_path,
        target_dataset_dir / f'light_metadata_{split}.json'
        )
    actual_id = 0
    
    for img_id, img_folder in tqdm(
            enumerate(img_folders),
            total=len(img_folders)
    ):
        # convert metadata
        meta_path = img_folder / 'metadata.json'
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        int_mat_i2c, ext_mat_c2w = datasets.parse_pose_from_meta(
            meta, 'nearlight', device=torch.device('cpu')
        )
        
        if not all(state == 0 for state in meta['near_light_status']):
            continue
        
        target_img_folder = target_dataset_dir / f'{split}_{actual_id:03d}'
        target_img_folder.mkdir(parents=True, exist_ok=True)
        actual_id += 1
        
        # construct target format: 'wildlight'
        tensoir_meta_this = datasets.pose_to_meta(
            int_mat_i2c, ext_mat_c2w,
            convention='tensoir'
        )
        # add light info
        envmap_name = meta['far_light']
        tensoir_meta_this['envmap'] = envmap_name
        tensoir_meta_this['envmap_inten'] = 1.0
        
        # save meta
        meta_path = target_img_folder / 'metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(tensoir_meta_this, f, indent=4)
        
        # copy images
        img_names = [
            'rgba.png',
            'albedo.png',
            'normal.png',
            'rgb_spec.png',
            'rgb_diff.png'
        ]
        for img_name in img_names:
            if (img_folder / img_name).exists():
                shutil.copy(
                    img_folder / img_name,
                    target_img_folder / img_name,
                )


def convert_nearlight_to_nerf(
        dataset_name: str,
        split: str,
        nearlight_dir: Path = vminer_syn_dir,
        nerf_dir: Path = nerf_syn_dir,
):
    src_dataset_dir = nearlight_dir / dataset_name / split
    img_folders = [x for x in src_dataset_dir.iterdir() if
                   x.stem.startswith(split + '_')]
    img_folders.sort()
    
    tgt_dataset_dir = nerf_dir / dataset_name
    tgt_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # read metadata
    with open(src_dataset_dir / f'transforms_{split}.json', 'r') as f:
        transform_meta = json.load(f)
    new_transform_meta = {
        'camera_angle_x': transform_meta['camera_angle_x'],
        'frames': []
    }
    meta_save_path = tgt_dataset_dir / f'transforms_{split}.json'
    actual_id = 0
    
    tgt_img_folder = tgt_dataset_dir / split
    tgt_img_folder.mkdir(parents=True, exist_ok=True)
    
    for img_id, img_folder in tqdm(
            enumerate(img_folders),
            total=len(img_folders)
    ):
        # convert metadata
        meta_path = img_folder / 'metadata.json'
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        if not all(state == 0 for state in meta['near_light_status']):
            continue
        
        # copy image
        target_img_path = tgt_img_folder / f'{actual_id:03d}.png'
        actual_id += 1
        shutil.copy(
            img_folder / 'rgba.png',
            target_img_path
        )
        # copy metadata
        meta_frame = transform_meta['frames'][img_id]
        transform_matrix = np.array(meta_frame['transform_matrix'])
        # change to nerf convention: negate 2nd and 3rd column
        transform_matrix[:, 1] *= -1
        transform_matrix[:, 2] *= -1
        meta_frame['transform_matrix'] = transform_matrix.tolist()
        # store relative path
        meta_frame['file_path'] = target_img_path.relative_to(
            meta_save_path.parent
        ).as_posix()
        # save
        new_transform_meta['frames'].append(meta_frame)
    
    # save metadata
    with open(meta_save_path, 'w') as f:
        json.dump(new_transform_meta, f, indent=4)


def convert_real_to_nearlight(
        scene_name: str
):
    src_scene_dir = real_dir / scene_name
    # img_folder = scene_dir / 'images'
    meta_all = src_scene_dir / 'transforms_F1N1.json'
    meta_f1n0 = src_scene_dir / 'transforms_F1N0.json'
    # read metadata
    with open(meta_all, 'r') as f:
        meta_all = json.load(f)
    # read frame meta
    frame_metas_all = meta_all['frames']
    # read metadata for F1N0
    with open(meta_f1n0, 'r') as f:
        meta_f1n0 = json.load(f)
    frames_f1n0 = meta_f1n0['frames']
    filepaths_f1n0 = [x['file_path'] for x in frames_f1n0]
    
    frame_metas_val = frame_metas_all[::10]
    frame_metas_train = [x for x in frame_metas_all if x not in frame_metas_val]
    
    scene_name = f'{scene_name}_F1N1'
    
    for split, frame_metas in zip(
            ['train', 'test'],
            [frame_metas_train, frame_metas_val]
    ):
        tgt_folder = vminer_real_dir / scene_name / split
        tgt_folder.mkdir(parents=True, exist_ok=True)
        tgt_meta = {
            # 'object_aabb': [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],
            'object_aabb': [[-0.75, -0.75, -0.75], [0.75, 0.75, 0.75]],
            'camera_angle_x': meta_all['camera_angle_x'],
            'frames': []
        }
        for img_id, frame_meta in tqdm(
                enumerate(frame_metas),
                total=len(frame_metas)
        ):
            filepath = frame_meta['file_path']
            # if filepath in F1N0_filepaths:
            #     frame['file_path'] = filepath.replace('F1N0', 'F1N1Hard')
            # else:
            #     frame['file_path'] = filepath.replace('F1N0', 'F2N1')
            transform_mat = np.array(frame_meta['transform_matrix'])
            # negate 2nd and 3rd column
            transform_mat[:, 1] *= -1
            transform_mat[:, 2] *= -1
            
            # read and process image
            rgba_path = src_scene_dir / filepath
            rgba = utils.read_image(rgba_path)
            # rgb = utils.linrgb2srgb_np(rgba[..., :3])
            rgb = rgba[..., :3]
            rgba = np.concatenate([rgb, rgba[..., -1:]], axis=-1)
            
            tgt_img_folder = tgt_folder / f'{split}_{img_id:03d}'
            tgt_img_folder.mkdir(parents=True, exist_ok=True)
            tgt_rgba_path = tgt_img_folder / 'rgba.png'
            utils.write_image(tgt_rgba_path, rgba, depth=8)
            
            tgt_meta_this = {
                'cam_angle_x': meta_all['camera_angle_x'],
                'cam_transform_mat': transform_mat.tolist(),
                'imh': meta_all['h'],
                'imw': meta_all['w'],
                'scene': scene_name,
                'spp': -1,
                'far_light': 'far_light_0',
                'near_light_status': [0] if filepath in filepaths_f1n0 else [1],
            }
            with open(tgt_img_folder / 'metadata.json', 'w') as f:
                json.dump(tgt_meta_this, f, indent=4)
            tgt_meta['frames'].append(
                {
                    'file_path': tgt_rgba_path.relative_to(tgt_folder).as_posix(),
                    'transform_matrix': tgt_meta_this['cam_transform_mat'],
                }
            )
        
        with open(tgt_folder / f'transforms_{split}.json', 'w') as f:
            json.dump(tgt_meta, f, indent=4)
        
        # save light meta
        tgt_meta_light = {
            "far_lights": {
                "amount": 1,
                "name": [
                    "far_light_0"
                ]
            },
            "near_lights": {
                "amount": 1,
                "pos_type": [
                    "collocated"
                ]
            }
        }
        with open(tgt_folder / f'light_metadata_{split}.json', 'w') as f:
            json.dump(tgt_meta_light, f, indent=4)


def convert_atom_to_nearlight():
    """Convert the captured data of "Atom" to nearlight format.
    
    Returns:

    """
    src_scene_dir = real_dir / "atom1200"
    img_folder = src_scene_dir / 'images'
    meta_all = src_scene_dir / 'transforms_110.json'
    # read metadata
    with open(meta_all, 'r') as f:
        meta_all = json.load(f)
    # read frame meta
    frame_metas_all = meta_all['frames']
    
    frame_metas_val = frame_metas_all[::10]
    frame_metas_train = [x for x in frame_metas_all if x not in frame_metas_val]
    
    scene_name = "atom_F2N2"
    
    for split, frame_metas in zip(
            ['train', 'test'],
            [frame_metas_train, frame_metas_val]
    ):
        tgt_folder = vminer_real_dir / scene_name / split
        tgt_folder.mkdir(parents=True, exist_ok=True)
        # save light metadata
        yellow_light_name = "yellow_light"
        white_light_name = "white_light"
        tgt_meta_light = {
            "far_lights": {
                "amount": 2,
                "name": [
                    yellow_light_name,
                    white_light_name,
                ]
            },
            "near_lights": {
                "amount": 2,
                "pos_type": [
                    "collocated",
                    "fixed",
                ]
            }
        }
        with open(tgt_folder / f'light_metadata_{split}.json', 'w') as f:
            json.dump(tgt_meta_light, f, indent=4)
        # process each frame
        tgt_meta = {
            'object_aabb': [[-1, -1, -1], [1, 1, 1]],
            'camera_angle_x': meta_all['camera_angle_x'],
            'frames': []
        }
        for img_id, frame_meta in tqdm(
                enumerate(frame_metas),
                total=len(frame_metas)
        ):
            filepath = src_scene_dir / frame_meta['file_path']
            file_id = int(filepath.stem.split('_')[1])
            
            tgt_img_folder = tgt_folder / f'{split}_{img_id:03d}'
            tgt_img_folder.mkdir(parents=True, exist_ok=True)
            tgt_rgba_path = tgt_img_folder / 'rgba.png'
            
            # build meta
            transform_mat = np.array(frame_meta['transform_matrix'])
            transform_mat[:, 1] *= -1
            transform_mat[:, 2] *= -1
            if 8818 <= file_id <= 8873:
                far_light_name = yellow_light_name
                exposure = 1.12
            else:
                far_light_name = white_light_name
                exposure = 5
            near_light_status = [0, 0]
            if 8818 <= file_id <= 8845:
                near_light_status[1] = 1
            if 8903 <= file_id <= 8930:
                near_light_status[0] = 1
            tgt_meta_this = {
                'cam_angle_x': meta_all['camera_angle_x'],
                'cam_transform_mat': transform_mat.tolist(),
                'imh': meta_all['h'],
                'imw': meta_all['w'],
                'scene': scene_name,
                'spp': -1,
                'far_light': far_light_name,
                'near_light_status': near_light_status,
            }
            with open(tgt_img_folder / 'metadata.json', 'w') as f:
                json.dump(tgt_meta_this, f, indent=4)
            tgt_meta['frames'].append(
                {
                    'file_path': tgt_rgba_path.relative_to(tgt_folder).as_posix(),
                    'transform_matrix': tgt_meta_this['cam_transform_mat'],
                }
            )
            
            # copy image
            if exposure == 1:
                shutil.copy(filepath, tgt_rgba_path)
            else:
                rgba = utils.read_image(filepath)
                rgb = utils.linrgb2srgb_np(utils.srgb2linrgb(rgba[..., :3]) * exposure)
                rgba = np.concatenate([rgb, rgba[..., -1:]], axis=-1)
                utils.write_image(tgt_rgba_path, rgba)
        
        with open(tgt_folder / f'transforms_{split}.json', 'w') as f:
            json.dump(tgt_meta, f, indent=4)


if __name__ == '__main__':
    # dataset_name = 'bulldemon1200_F1N1'
    # split = 'train'
    # convert_real_to_nearlight(dataset_name)
    convert_atom_to_nearlight()
    # convert_nearlight_to_wildlight(
    #     dataset_name,
    #     split,
    #     nearlight_real_dir,
    #     wildlight_real_dir,
    # )
    # convert_nearlight_to_wildlight(dataset_name, 'test')
    # convert_nearlight_to_nerf(
    #     dataset_name,
    #     split,
    #     nearlight_real_dir,
    #     nerf_real_dir
    # )
    # convert_nearlight_to_wildlight('mic_F1N1Hard')
    # convert_nearlight_to_tensoir(
    #     dataset_name,
    #     split,
    #     nearlight_real_dir
    # )
