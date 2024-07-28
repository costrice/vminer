"""
This file contains the dataset reader and sampler.
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from internal import geo_utils, lighting, utils


class TrainingRaySampler:
    """Sampler that samples ray chunks with the same lighting condition."""
    
    def __init__(
            self,
            all_cond_rays: List[Dict[str, Any]],
            batch_size: int,
            device: torch.device,
            drop_last: bool = False,
    ):
        self.all_cond_rays = all_cond_rays
        self.batch_size = batch_size
        self.n_imgs = len(all_cond_rays)
        self.device = device
        
        # initialize random permute within each image
        if drop_last:
            self.total_for_each = [
                x['n_rays'] - x['n_rays'] % batch_size for x in all_cond_rays]
        else:
            self.total_for_each = [x['n_rays'] for x in all_cond_rays]
        self.curr_for_each = None
        self.left_for_each = None
        self.permutations = None
        self.reset()
    
    def reset(self):
        self.curr_for_each = [0] * self.n_imgs
        self.left_for_each = self.total_for_each.copy()
        self.permutations = [torch.randperm(x['n_rays'], device=self.device)
                             for x in self.all_cond_rays]
    
    def __next__(self):
        # choose an image according to the number of rays left
        img_idx = random.choices(
            range(len(self.all_cond_rays)),
            weights=self.left_for_each
        )[0]
        # get the permutation for the chosen image
        perm = self.permutations[img_idx]
        # get the current index for the chosen image
        curr = self.curr_for_each[img_idx]
        # get the batch size for the chosen image
        batch_size = min(self.batch_size, self.left_for_each[img_idx])
        # get the indices for the chosen image
        indices = perm[curr:curr + batch_size]
        # update the current and left index for the chosen image
        self.curr_for_each[img_idx] += batch_size
        self.left_for_each[img_idx] -= batch_size
        # if all rays in all images are used, reset the sampler
        if all(x == 0 for x in self.left_for_each):
            self.reset()
        
        # get the rays for the chosen image
        img_rays = self.all_cond_rays[img_idx]
        img_rays_batch = {}
        for key, value in img_rays.items():
            if isinstance(value, torch.Tensor):
                if value.shape[0] == img_rays['n_rays']:
                    value = value[indices]
                value = value.to(self.device)
            img_rays_batch[key] = value
        img_rays_batch['n_rays'] = batch_size
        return img_rays_batch


def read_gt_lighting(
        light_dir: Union[str, Path],
        light_names: List[str],
        device: torch.device,
):
    """
    Read ground truth HDR lighting in the given directory.
    Args:
        light_dir: Path to the directory containing ground truth HDR lighting.
        light_names: Names of the lights to read, without extension.
        device: Device to put the data on.

    Returns:
        lights_gt (Dict[str, torch.Tensor]): Ground truth HDR lighting.
        gt_light_h (int): Height of the ground truth HDR lighting.
        gt_light_w (int): Width of the ground truth HDR lighting.
    """
    # initialize ground truth lighting instance
    far_lights_gt: lighting.PixelFarLight = lighting.create_light(
        which='far',
        light_type='pixel',
        n_lights=len(light_names),
        device=device,
        requires_grad=False,
        envmap_h=32,
        envmap_w=64,
    )
    # read GT envmaps
    gt_envmaps = []
    for light_name in light_names:
        light_path = light_dir / f'{light_name}.hdr'
        assert light_path.exists(), f'{light_path} does not exist.'
        gt_envmap = utils.read_image(light_path)
        gt_envmap = torch.from_numpy(gt_envmap)
        gt_envmaps.append(gt_envmap)
    gt_envmaps = torch.stack(gt_envmaps, dim=0).to(device)
    far_lights_gt.override_actual_params(gt_envmaps, mutable=False)
    
    return far_lights_gt


def read_light_meta(meta_path: Union[str, Path]):
    """
    Read metadata of lights in the scene.
    Args:
        meta_path (Union[str, Path]): Path to the json light metadata file.
    Returns:
        n_far_lights (int): Amount of far lights in the scene.
        far_lights_meta (Dict[str, Any]): Metadata of far lights.
        n_near_lights (int): Amount of near lights in the scene.
        near_lights_meta (Dict[str, Any]): Metadata of near lights.
    """
    with open(meta_path, 'r') as f:
        lights_info = json.load(f)
        near_lights_meta = lights_info['near_lights']
        n_near_lights = near_lights_meta['amount']
        
        far_lights_meta = lights_info['far_lights']
        n_far_lights = far_lights_meta['amount']
        
        assert len(near_lights_meta['pos_type']) == n_near_lights
        assert len(far_lights_meta['name']) == n_far_lights
        assert all(
            x in ['fixed', 'collocated'] for x in
            near_lights_meta['pos_type']
        )
    
    return n_far_lights, far_lights_meta, n_near_lights, near_lights_meta


def load_K_Rt_from_P(P: np.ndarray):
    """Decompose the projection matrix into intrinsic matrix and extrinsic
    matrix. Used by WildLight.

    Args:
        P: 3x4 projection matrix from 3D homogeneous point in world space to 2D
            homogeneous pixel location. For column vector.

    Returns:

    """
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]
    
    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K
    
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    
    return intrinsics, pose


def pose_to_meta(
        int_mat_i2c: torch.Tensor,
        ext_mat_c2w: torch.Tensor,
        convention: str,
):
    """Convert the pose to a metadata dictionary.

    All input matrices should follow the row-major order (i.e. should be put on
    the right of a row vector in multiplication).

    Args:
        int_mat_i2c: 3x3 intrinsic matrix from 2D homogeneous pixel location to
            3D point in camera space (opencv coordinate).
        ext_mat_c2w: 4x4 extrinsic matrix from 3D homogeneous point in camera
            space (opencv coordinate) to 3D point homogeneous in world space.

    Returns:
        A dictionary containing the metadata for the pose according to the
        convention.
    """
    if convention == 'wildlight':
        # construct projection matrix from world space to image coordinate,
        # in column-major order
        ext_mat_w2c = ext_mat_c2w.inverse()
        int_mat_c2i = int_mat_i2c.inverse()
        P = ext_mat_w2c[:, :3] @ int_mat_c2i
        P = P.T  # from row-major to column-major
        return {
            'world_mat': P.cpu().numpy(),
        }
    elif convention == 'tensoir':
        # cam_transform_mat is c2w extrinsic, in column major order, and
        # camera coord system is in blender coord
        cam_transorm_mat = ext_mat_c2w.T
        # opencv to blender
        cam_transorm_mat[:, 1] = -cam_transorm_mat[:, 1]
        cam_transorm_mat[:, 2] = -cam_transorm_mat[:, 2]
        cam_transorm_mat = cam_transorm_mat.view(-1).tolist()
        cam_transorm_mat = [str(x) for x in cam_transorm_mat]
        cam_transorm_mat = ','.join(cam_transorm_mat)
        # compute camera view angle
        int_mat_c2i = int_mat_i2c.inverse().T
        imw = round(int_mat_c2i[0, 2].item() * 2)
        imh = round(int_mat_c2i[1, 2].item() * 2)
        focal = int_mat_c2i[0, 0].item()  # in pixel
        fov = 2 * np.arctan(imw / 2 / focal)
        return {
            'cam_angle_x': fov,
            'cam_transform_mat': cam_transorm_mat,
            'imh': imh,
            'imw': imw,
        }
    else:
        raise NotImplementedError(
            f'Convention {convention} is not implemented.'
        )


def parse_pose_from_meta(
        meta: Dict[str, Any],
        convention: str,
        device: torch.device,
):
    """Parse the metadata dictionary to get the transformation matrices and
    light information.

    The transformation matrices are in the row-major order (i.e. should be put
    on the right of a row vector in multiplication). They transform between 3D
    homogeneous point in world space, in camera space (opencv coordinate), and
    2D homogeneous pixel location.

    Args:
        meta: the metadata dictionary for an image.
        convention: the convention of the metadata dictionary. Can be
            'nearlight', 'tensoir', 'wildlight'.
        device: the device to put the transformation matrix on.

    Returns:
        A tuple of transformation matrices:

        - int_mat_i2c (torch.Tensor): 3x3 intrinsic matrix from 2D homogeneous
            pixel location to 3D point in camera space (opencv
            coordinate).
        - ext_mat_c2w (torch.Tensor): 4x4 extrinsic matrix from 3D homogeneous
            point in camera space (opencv coordinate) to 3D point homogeneous
            in world space.
    """
    if convention == 'nearlight':
        img_hw = (int(meta['imh']), int(meta['imw']))
        
        # Get ray directions for all pixels, same for all images
        # (with same H, W, focal)
        focal = 0.5 * int(meta['imw']) / np.tan(
            0.5 * meta['cam_angle_x']
        )  # fov -> focal length
        focal *= img_hw[1] / meta['imw']  # account for possible downsample
        
        h, w = img_hw
        
        # get intrinsic matrix from 3D camera space (opencv coordinate) to 2D
        # image opencv coordinate, x: right, y: down, z: forward
        int_mat_c2i = geo_utils.get_intrinsic_matrix(
            h, w, focal, device
        )
        int_mat_i2c = torch.inverse(int_mat_c2i)
        
        # get camera extrinsic matrix from camera space (opencv coord) to world
        # space
        ext_mat_c2w = torch.tensor(
            meta["cam_transform_mat"],
            dtype=torch.float32, device=device
        )
        ext_mat_c2w = ext_mat_c2w.T  # from column-major to row-major
        
        return int_mat_i2c, ext_mat_c2w
    
    elif convention == 'wildlight':
        world_mat = meta['world_mat']
        intrinsics_c2i, pose_c2w = load_K_Rt_from_P(world_mat)
        # from column-major to row-major
        intrinsics_c2i = torch.from_numpy(intrinsics_c2i).T.float()
        pose_c2w = torch.from_numpy(pose_c2w).T.float()
        
        int_mat_i2c = intrinsics_c2i.inverse()[:3, :3]
        ext_mat_c2w = pose_c2w
        
        return int_mat_i2c, ext_mat_c2w
    else:
        raise NotImplementedError(
            f'Convention {convention} is not implemented.'
        )


class VMINerDataset(Dataset):
    def __init__(
            self,
            img_dir: Union[str, Path],
            split: str = 'train',
            device: torch.device = torch.device('cpu'),
            img_hw: Tuple[int, int] = None,
            read_extra: bool = False,
            read_gt_far_lights: bool = False,
            remove_ignored: bool = False,
    ):
        """
        Dataset for scenes including both near-field lights and far-field
        lights.
        Args:
            img_dir: Path to the scene image folder
            split: Typically 'train', 'test', or 'val'
            device: Device to put the data on
            img_hw: Height and width of the images. If the sizes are varied across
                images, set to None.
            read_extra: whether to read extra information including albedo,
                normal, specular, and diffuse, for computing metrics
            read_gt_far_lights: whether to read ground truth far lights
            remove_ignored: remove rays specified in ignored.png (if given)
        """
        # check input arguments
        assert split in ['train', 'test', 'val'], \
            f'Invalid split {split}, must be one of [train, test, val].'
        
        # load image paths
        self.device = device
        self.img_dir = Path(img_dir)
        self.scene = self.img_dir.stem  # Scene name e.g. 'lego', 'hotdog'
        self.split = split
        self.img_folders = [x for x in self.img_dir.iterdir() if
                            x.stem.startswith(self.split + '_')]
        self.img_folders.sort()
        
        # read light data
        (self.n_far_lights, self.far_lights_meta,
         self.n_near_lights, self.near_lights_meta) = (
            read_light_meta(self.img_dir / f'light_metadata_{split}.json'))
        
        # read ground truth lighting if given
        if read_gt_far_lights:
            light_dir = self.img_dir / 'gt_far_lights'
            if not light_dir.exists():
                raise FileNotFoundError(
                    f'Ground truth far lights not found in {light_dir}.'
                )
            
            self.far_lights_gt: lighting.PixelFarLight = read_gt_lighting(
                light_dir,
                self.far_lights_meta['name'],
                device=self.device
            )
        
        # reading settings
        self.img_hw = img_hw
        self.transform = self.define_transforms()
        
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        # inverse is the same
        self.opencv2blender = self.blender2opencv.copy()
        
        self.read_extra = read_extra
        self.remove_ignored = remove_ignored
        
        # when training, load all the rays and RGB values
        # when testing, load them on the fly
        if self.split == 'train':
            self.all_cond_rays = self.read_all_frames()
        
        print(f'\n{self.split.capitalize()} dataset size: {len(self)}')
        print(self.get_light_info_str())
        
        # try to read the first image to test if the dataset is valid
        try:
            self[0]
        except Exception as e:
            print(f'Error when reading the first image: {e}')
            raise e
    
    def get_light_info_str(self):
        """
        Get a string containing light information.
        Returns:
            light_info_str (str): Light information string.
        """
        light_info_str = (f'There are {self.n_far_lights} far lights and '
                          f'{self.n_near_lights} near lights '
                          f'in the {self.split} split.\n')
        # add far lights
        light_info_str += f'Far lights:\n'
        for light_idx in range(self.n_far_lights):
            light_info_str += \
                (f'  {light_idx}: '
                 f'{self.far_lights_meta["name"][light_idx]}\n')
        # add near lights
        light_info_str += f'Near lights:\n'
        for light_idx in range(self.n_near_lights):
            light_info_str += \
                (f'  {light_idx}: '
                 f'{self.near_lights_meta["pos_type"][light_idx]}\n')
        return light_info_str
    
    def define_transforms(self):
        """
        Define image transformations for the dataset.
        Returns:
            transforms (torchvision.transforms.Compose): Composed image
                transformations.
        """
        transforms = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        return transforms
    
    def __len__(self):
        return len(self.img_folders)
    
    def read_all_frames(self):
        """Read all frames in the dataset, convert them to rays, and store them
        on device."""
        tqdm_desc = f'Loading {self.split} data, ' \
                    f'data group amount: {len(self)}, ' \
                    f'far lighting amount: {self.n_far_lights}, ' \
                    f'near lighting amount: {self.n_near_lights}.'
        
        cluster_same_light = True
        
        all_cond_rays = []
        n_conds = 0
        all_conditions = {}
        
        # cluster the rays according to different lighting conditions
        # (far light index, near light state)
        for idx in tqdm(range(len(self)), desc=tqdm_desc):
            ret_dict = self[idx]
            # put on device
            for keys in ['rays', 'rgb', 'alpha', 'near_light_state']:
                ret_dict[keys] = ret_dict[keys].to(self.device)
            
            far_light_idx = ret_dict['far_light_idx']
            near_light_state = ret_dict['near_light_state']
            if cluster_same_light:
                condition = (far_light_idx,
                             tuple(near_light_state.tolist()))
                if condition not in all_conditions:
                    all_conditions[condition] = n_conds
                    all_cond_rays.append(
                        {
                            'n_rays': 0,
                            'rays': [],
                            'rgb': [],
                            'alpha': [],
                            'far_light_idx': far_light_idx,
                            'near_light_state': near_light_state,
                        }
                    )
                    n_conds += 1
                cond_id = all_conditions[condition]
            else:
                cond_id = idx
                all_cond_rays.append(
                    {
                        'n_rays': 0,
                        'rays': [],
                        'rgb': [],
                        'alpha': [],
                        'far_light_idx': far_light_idx,
                        'near_light_state': near_light_state,
                    }
                )
                n_conds += 1
            all_cond_rays[cond_id]['n_rays'] += ret_dict['n_rays']
            for keys in ['rays', 'rgb', 'alpha']:
                all_cond_rays[cond_id][keys].append(ret_dict[keys])
        
        # concatenate the rays
        for cond_id in range(n_conds):
            for keys in ['rays', 'rgb', 'alpha']:
                all_cond_rays[cond_id][keys] = torch.cat(
                    all_cond_rays[cond_id][keys], dim=0
                )
        
        torch.cuda.empty_cache()
        return all_cond_rays
    
    def __getitem__(self, idx: int):
        item_dir = self.img_folders[idx]
        item_meta_path = item_dir / 'metadata.json'
        with open(item_meta_path, 'r') as f:
            meta = json.load(f)
        
        img_hw = (int(meta['imh']), int(meta['imw']))
        
        int_mat_i2c, ext_mat_c2w = parse_pose_from_meta(
            meta, 'nearlight', self.device
        )
        
        # [H, W, 3] containing (x, y, 1)
        h, w = img_hw
        device = self.device
        # [H, W, 3] containing (x, y, 1)
        img_coord_homo = geo_utils.get_homo_image_coord(
            h, w, device
        )
        dirs_camera_opencv = img_coord_homo @ int_mat_i2c  # [H, W, 3]
        
        # get ray data
        rays_o, rays_d = geo_utils.camera_ray_to_world(
            dirs_camera_opencv, ext_mat_c2w
        )
        rays = torch.cat([rays_o, rays_d], -1)  # [H * W, 6]
        
        # read light metadata
        far_light_name = meta['far_light']
        if far_light_name == 'none':
            far_light_idx = -1
        else:
            far_light_idx = self.far_lights_meta['name'].index(far_light_name)
        
        near_light_state = meta['near_light_status']
        near_light_state = torch.tensor(  # use as int for now
            near_light_state, dtype=torch.float
        ).round().int()
        
        # Read RGB data
        img_path = item_dir / f'rgba.png'
        rgb = self.read_and_resize_image(img_path)
        
        if rgb.shape[1] == 4:
            alpha = rgb[:, -1:]
            rgb = rgb[:, :3] * alpha
        else:
            alpha = torch.ones((rgb.shape[0], 1), dtype=torch.float)
        
        # filter fg
        if self.remove_ignored:
            if (item_dir / 'ignored.png').exists():
                ignored = utils.read_image(item_dir / 'ignored.png')
                ignored = torch.from_numpy(ignored).float().to(self.device).view(-1)
                rays = rays[ignored < 0.5]
                rgb = rgb[ignored < 0.5]
                alpha = alpha[ignored < 0.5]
        
        return_dict = {
            'path': str(item_dir),
            'n_rays': rays.shape[0],
            'h': h,
            'w': w,
            'rays': rays,
            'rgb': rgb,
            'alpha': alpha,
            'far_light_idx': far_light_idx,
            'near_light_state': near_light_state,
        }
        
        if self.read_extra:
            albedo_path = item_dir / 'albedo.png'
            if albedo_path.exists():
                albedo = self.read_and_resize_image(albedo_path)
                albedo = albedo[:, :3] * alpha
            else:
                albedo = None
            
            normal_path = item_dir / 'normal.png'
            # assert normal_path.exists(), f'{normal_path} does not exist.'
            if normal_path.exists():
                normal = self.read_and_resize_image(normal_path)
                normal = normal * 2 - 1
                normal = normal[:, :3] * alpha
                normal = F.normalize(normal, p=2, dim=-1, eps=1e-6)
            else:
                normal = None
            
            specular_path = item_dir / 'rgb_spec.png'
            # assert specular_path.exists(), f'{specular_path} does not exist.'
            if specular_path.exists():
                specular = self.read_and_resize_image(specular_path)
                specular = specular[:, :3] * alpha
            else:
                specular = None
            
            diffuse_path = item_dir / 'rgb_diff.png'
            # assert diffuse_path.exists(), f'{diffuse_path} does not exist.'
            if diffuse_path.exists():
                diffuse = self.read_and_resize_image(diffuse_path)
                diffuse = diffuse[:, :3] * alpha
            else:
                diffuse = None
            
            return_dict.update(
                {
                    'albedo': albedo,
                    'normal': normal,
                    'rgb_spec': specular,
                    'rgb_diff': diffuse,
                }
            )
        
        return return_dict
    
    def read_and_resize_image(
            self,
            img_path: Union[str, Path]
    ) -> torch.Tensor:
        """Read and resize an image.

        Args:
            img_path: Path to the image.
        Returns:
            img: [H * W, C] resized and reshaped image on self.device.
        """
        img = utils.read_image(img_path)
        img = self.transform(img).to(self.device)
        if self.img_hw is not None and img.shape[1:] != self.img_hw:
            interp_mode = 'bilinear' if img.shape[1] < self.img_hw[0] \
                else 'area'
            img = F.interpolate(
                img.unsqueeze(0), self.img_hw[::-1],
                mode=interp_mode, align_corners=False
            )
            img = img.squeeze(0)
        img = img.view(img.shape[0], -1).permute(1, 0).float()
        return img
