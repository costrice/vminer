"""
This file contains utility functions.
"""
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from dateutil import tz
from PIL import Image

mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.tensor([10.]))


def current_time(format='%Y%b%d%a-%H%M%S') -> str:
    """
    Return current time as a string.
    Args:
        format (str): format of the time string.
        Refer to https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior

    Returns:
        str: the current time string.
    """
    return datetime.now().astimezone(tz.gettz('UTC+8')).strftime(format)


def tensor2ndarray(img: torch.Tensor) -> np.ndarray:
    """
    Convert an image tensor to a numpy ndarray. If the tensor contains batch
    dimension, only takes the first one in the batch.
    Args:
        img (torch.Tensor): the input tensor.

    Returns:
        np.ndarray: the converted ndarray.
    """
    if len(img.size()) == 4:
        print(
            'Warning: tensor2ndarray() received a tensor with batch '
            'dimension. Only take the first one in the batch.'
        )
        img = img[0]
    return img.cpu().detach().numpy().transpose(1, 2, 0)


def ndarray2tensor(img: np.ndarray) -> torch.Tensor:
    """
    Convert an image ndarray to a torch.Tensor with shape (1, C, H, W).
    Args:
        img (np.ndarray): the input ndarray.

    Returns:
        torch.Tensor: the converted tensor with shape (1, C, H, W).
    """
    return TF.to_tensor(img)[None]


def linrgb2srgb(
        color_linrgb: torch.Tensor
) -> torch.Tensor:
    """
    Transform an image in [0, 1] from linear sRGB to sRGB space.
    Args:
        color_linrgb (Union[np.ndarray, torch.Tensor]): the input image.

    Returns:
        Union[np.ndarray, torch.Tensor]: the converted image in sRGB space.
    """
    big = color_linrgb > 0.0031308
    # color_srgb = color_linrgb * 0
    # use this causes infinity in the gradient
    # color_linrgb = torch.clamp(color_linrgb, 1e-4, 1)
    # color_srgb = ((1.055 * (color_linrgb ** (1 / 2.4)) - 0.055) * big
    #               + 12.92 * color_linrgb * (~big))
    # color_srgb[big] = 1.055 * (color_linrgb[big] ** (1 / 2.4)) - 0.055
    # color_srgb[~big] = color_linrgb[~big] * 12.92
    # efficient and stable version?
    color_srgb = ((1.055 * (color_linrgb.clamp(min=3e-3)
                            ** (1 / 2.4)) - 0.055) * big
                  + color_linrgb * 12.92 * (~big))
    # # color_srgb = color_linrgb ** (1 / 2.2)
    return color_srgb


def linrgb2srgb_np(
        color_linrgb: np.ndarray
) -> np.ndarray:
    """
    Transform an image in [0, 1] from linear sRGB to sRGB space.
    Args:
        color_linrgb (Union[np.ndarray, torch.Tensor]): the input image.

    Returns:
        Union[np.ndarray, torch.Tensor]: the converted image in sRGB space.
    """
    big = color_linrgb > 0.0031308
    # color_srgb = color_linrgb * 0
    # use this causes infinity in the gradient
    # color_linrgb = torch.clamp(color_linrgb, 1e-4, 1)
    # color_srgb = ((1.055 * (color_linrgb ** (1 / 2.4)) - 0.055) * big
    #               + 12.92 * color_linrgb * (~big))
    # color_srgb[big] = 1.055 * (color_linrgb[big] ** (1 / 2.4)) - 0.055
    # color_srgb[~big] = color_linrgb[~big] * 12.92
    # efficient and stable version?
    color_srgb = ((1.055 * (color_linrgb.clip(min=3e-3)
                            ** (1 / 2.4)) - 0.055) * big
                  + color_linrgb * 12.92 * (~big))
    # # color_srgb = color_linrgb ** (1 / 2.2)
    return color_srgb


def srgb2linrgb(
        color_srgb: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform an image in [0, 1] from sRGB to linear sRGB space.
    Args:
        color_srgb (Union[np.ndarray, torch.Tensor]): the input image.

    Returns:
        Union[np.ndarray, torch.Tensor]: the converted image in linear sRGB
        space.
    """
    big = color_srgb > 0.0404482362771082
    color_linrgb = (((color_srgb + 0.055) / 1.055) ** 2.4 * big
                    + color_srgb / 12.92 * (~big))
    # color_linrgb = color_srgb * 0
    # color_linrgb[big] = ((color_srgb[big] + 0.055) / 1.055) ** 2.4
    # color_linrgb[~big] = color_srgb[~big] / 12.92
    # color_linrgb = color_srgb ** 2.2
    return color_linrgb


def to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert an image into np.uint8. If the input dtype is float,
    clip to [0, 1] first.
    Args:
        img (np.ndarray): the input image.

    Returns:
        np.ndarray: the converted image with dtype np.uint8.
    """
    if img.dtype == bool:
        img = img.astype(np.float32)
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img / 255).astype(np.uint8)
    if img.dtype in [np.float32, np.float64]:
        return (np.around(np.clip(img, 0, 1) * 255)
                .astype(np.uint8))
    raise ValueError(f'Unsupported dtype: {img.dtype}')


def to_uint16(img: np.ndarray) -> np.ndarray:
    """
    Convert an image into np.uint16. If the input dtype is float,
    clip to [0, 1] first.
    Args:
        img (np.ndarray): the input image.

    Returns:
        np.ndarray: the converted image with dtype np.uint16.
    """
    if img.dtype == bool:
        img = img.astype(np.float32)
    if img.dtype == np.uint8:
        return img.astype(np.uint16) * 255
    if img.dtype == np.uint16:
        return img
    if img.dtype in [np.float32, np.float64]:
        return (np.around(np.clip(img, 0, 1) * 65535)
                .astype(np.uint16))
    raise ValueError(f'Unsupported dtype: {img.dtype}')


def to_float32(img: np.ndarray) -> np.ndarray:
    """
    Convert an image into np.float32.
    Args:
        img (np.ndarray): the input image.

    Returns:
        np.ndarray: the converted image with dtype np.float32.
    """
    if img.dtype == bool:
        return img.astype(np.float32)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535
    if img.dtype == np.float32:
        return img
    raise ValueError(f'Unsupported dtype: {img.dtype}')


def read_image(read_path: Union[str, Path]) -> np.ndarray:
    """
    read an image and convert to np.float32.
    If the image is LDR, the value range should be [0, 1].
    If the image is HDR, the value range should be [0, +inf].
    Args:
        read_path (Union[str, Path]): path to the image.

    Returns:
        np.ndarray: the image with dtype np.float32.
    """
    read_path = Path(read_path)
    if read_path.exists():
        img = cv2.imread(
            str(read_path),
            flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
                  | cv2.IMREAD_UNCHANGED
        )
        if img is None:
            raise ValueError(f'File {read_path} cannot be read.')
        if len(img.shape) == 3:
            if img.shape[2] == 3:  # BGR
                img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:  # BGRA
                img = cv2.cvtColor(img, code=cv2.COLOR_BGRA2RGBA)
            else:
                raise ValueError(f'Unexpected image shape {img.shape}')
        img = to_float32(img)
        return img
    else:
        raise FileNotFoundError(f'File {read_path} not found.')


def write_image(
        write_path: Union[str, Path],
        img: np.ndarray,
        depth: int = 8
):
    """
    Write a ndarray image into output path.
    Depending on the specified extension of output path, the image saved can be
    LDR (.png, .jpg) or HDR (.hdr, .exr).

    Args:
        write_path (Union[str, Path]): the saving path.
        img (np.ndarray): the image to be saved.
        depth (int): bit-depth, can be 8 or 16.

    Returns:
        None
    """
    write_path = Path(write_path)
    if img.dtype in [np.float64, bool]:  # cv2 do not support float64?
        img = img.astype(np.float32)
    if len(img.shape) == 3:
        if img.shape[2] == 3:  # RGB
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, code=cv2.COLOR_RGBA2BGRA)
    if write_path.suffix.lower() in ['.hdr', '.exr']:
        cv2.imwrite(str(write_path), to_float32(img))
    elif write_path.suffix.lower() in ['.png', '.jpg']:
        if depth == 8:
            cv2.imwrite(str(write_path), to_uint8(img))
        elif depth == 16 and write_path.suffix.lower() == '.png':
            cv2.imwrite(str(write_path), to_uint16(img))
        else:
            raise ValueError(
                f'Unexpected depth {depth} for '
                f'{write_path.suffix}'
            )
    else:
        raise ValueError(f'Unexpected file extension in {write_path}')


def visualize_depth_numpy(
        depth: np.ndarray,
        min_max: Tuple[float, float] = None,
        cmap: int = cv2.COLORMAP_JET,
        mask: np.ndarray = None
):
    """
    Visualize depth map.
    Args:
        depth: (H, W) the depth map.
        min_max: the minimum and maximum depth value.
        cmap: the colormap used for visualization.
        mask: the mask of the depth map.

    Returns:
        A tuple containing:

        - np.ndarray: (H, W, 3) the visualization.
        - List[float]: the minimum and maximum depth value.
    """
    x = np.nan_to_num(depth)  # change nan to 0
    if min_max is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = min_max
    
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    
    if mask is not None:
        x[~mask] = 1.0
    
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi, ma]


def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log


def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()
    
    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax
    
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi, ma]


def total_num_to_resolution(
        n_voxels: int,
        aabb: torch.Tensor,
) -> List[int]:
    """Compute the voxel grid resolution.

    Compute the number of voxels in each direction given the total number of
    voxels and the bounding box. The results keep each voxel a cube with equal
    edge lengths in each direction.

    Args:
        n_voxels: the total number of voxels
        aabb: [2, 3), the axis-aligned bounding box of the scene

    Returns:
        List[int]: the number of voxels in each direction.
    """
    xyz_min, xyz_max = aabb
    # total volumes / number of all voxels
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()


def max_samples_of_voxel_grid(
        grid_reso: List[int],
        step_ratio: float,
) -> int:
    """
    Compute the maximum of samples, or the number of samples through the
    diagonal line, of the voxel grid.
    Args:
        grid_reso (List[int]): the resolution of the voxel grid.
        step_ratio (float): the ratio of step size to voxel size.

    Returns:
        int: the number of samples through the diagonal line of the voxel grid.
    """
    return int(np.linalg.norm(grid_reso) / step_ratio)


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the total variation loss of a tensorial split component.
    Args:
        x (torch.Tensor): [b, c, h, w] the split component (plane or line).

    Returns:
        torch.Tensor: the total variation loss.
    """
    b, c, h, w = x.shape
    h_tv = (x[..., 1:, :] - x[..., :-1, :]).pow(2).mean() if h > 1 else 0.0
    w_tv = (x[..., :, 1:] - x[..., :, :-1]).pow(2).mean() if w > 1 else 0.0
    return h_tv + w_tv
