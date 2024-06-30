from pathlib import Path

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from internal import utils


def combine_horizontally(
        img_1: np.ndarray,
        img_2: np.ndarray,
        offset: int,
) -> np.ndarray:
    """Combine two images horizontally, considering 0-1 alpha channel.

    Returns:

    """
    if offset < 0:
        img_left = img_1[:, :offset]
        left_img_mid = img_1[:, offset:]
        left_img_mid_rgb = left_img_mid[:, :, :3]
        left_img_mid_alpha = left_img_mid[:, :, 3:4]
        right_img_mid = img_2[:, :-offset]
        right_img_mid_rgb = right_img_mid[:, :, :3]
        right_img_mid_alpha = right_img_mid[:, :, 3:4]
        img_mid_rgb = (left_img_mid_rgb * left_img_mid_alpha +
                       right_img_mid_rgb * right_img_mid_alpha) / \
                      ((left_img_mid_alpha + right_img_mid_alpha) + 1e-4)
        img_mid_alpha = left_img_mid_alpha + right_img_mid_alpha
        img_mid = np.concatenate([img_mid_rgb, img_mid_alpha], axis=2)
        img_right = img_2[:, -offset:]
        img = np.concatenate([img_left, img_mid, img_right], axis=1)
    elif offset > 0:
        img_mid = np.zeros((img_1.shape[0], offset, 4), dtype=np.float32)
        # img_mid[..., 3] = 0
        img = np.concatenate([img_1, img_mid, img_2], axis=1)
    else:
        img = np.concatenate([img_1, img_2], axis=1)
    return img


def combine_vertically(
        img_1: np.ndarray,
        img_2: np.ndarray,
        offset: int,
) -> np.ndarray:
    """Combine two images vertically, considering 0-1 alpha channel.

    Returns:

    """
    if offset < 0:
        img_top = img_1[:offset, :]
        top_img_mid = img_1[offset:, :]
        top_img_mid_rgb = top_img_mid[:, :, :3]
        top_img_mid_alpha = top_img_mid[:, :, 3:4]
        bottom_img_mid = img_2[:-offset, :]
        bottom_img_mid_rgb = bottom_img_mid[:, :, :3]
        bottom_img_mid_alpha = bottom_img_mid[:, :, 3:4]
        img_mid_rgb = (top_img_mid_rgb * top_img_mid_alpha +
                       bottom_img_mid_rgb * bottom_img_mid_alpha) / \
                      ((top_img_mid_alpha + bottom_img_mid_alpha) + 1e-4)
        img_mid_alpha = top_img_mid_alpha + bottom_img_mid_alpha
        img_mid = np.concatenate([img_mid_rgb, img_mid_alpha], axis=2)
        img_bottom = img_2[-offset:, :]
        img = np.concatenate([img_top, img_mid, img_bottom], axis=0)
    elif offset > 0:
        img_mid = np.zeros((offset, img_1.shape[1], 4), dtype=np.float32)
        # img_mid[..., 3] = 0
        img = np.concatenate([img_1, img_mid, img_2], axis=0)
    else:
        img = np.concatenate([img_1, img_2], axis=0)
    return img


def combine_comparison_figure():
    img_hw = 800
    
    src_dir = Path(r'E:\Codes\VMINer\comparison_results')
    scene = 'debu'
    # img_id = 182
    
    tgt_dir = Path(r'E:\Codes\VMINer\comparison_results\comp_visual') / scene
    tgt_dir.mkdir(exist_ok=True, parents=True)
    
    attributes = [
        'albedo',
        'normal',
        'rgb',
        'rgb_spec'
    ]
    methods = [
        # 'reference_si',
        'NearLight_F1N1_si',
        'wildlight_F1N1_si',
        'tensoir_F1N0_si',
        'nvdiffrecmc_F1N0_si'
    ]
    
    for img_id in tqdm(range(200)):
        # lego
        horiz_offset = -int(0.4 / 2 * img_hw)
        total_width = img_hw * 5 + horiz_offset * 4
        rgb_clip_udlr = ((0.04 + (1.84 - 1.7) / 2) / 1.84 * img_hw,
                         (1.84 + 0.04 - (1.84 - 1.7) / 2) / 1.84 * img_hw,
                         (0.03 + (1.84 - 1.63) / 2) / 1.84 * img_hw,
                         (1.84 + 0.03 - (1.84 - 1.63) / 2) / 1.84 * img_hw)
        rgb_clip_udlr = [int(x) for x in rgb_clip_udlr]
        # rgb_clip_udlr = [0, 700, 120, 680]
        # other_clip_udlr = [0, 600, 0, 800]
        # vert_offsets = [
        #     0,
        #     -110,
        #     -110,
        #     0,
        #     -110,
        # ]
        vert_offsets = [
            -50,
            -220,
            -20,
            -50,
            -20,
        ]
        # vert_offset = -int(0.48 / 2 * img_hw)
        # vert_offset_small = vert_offset // 4
        
        # read alpha
        alpha = utils.read_image(
            src_dir / scene / 'NearLight_F1N1' / f'rgb_{img_id:03d}.png'
        )
        alpha = alpha[:, :, 3:4]
        
        # read images
        whole_img = None
        for a_id, attribute in enumerate(attributes):
            attribute_img = None
            for m_id, method in enumerate(methods):
                file_name = f'{attribute}_{img_id:03d}.png'
                file_path = src_dir / scene / method / file_name
                img = utils.read_image(file_path)
                if attribute != 'rgb':
                    # replace alpha
                    img[:, :, 3:4] = alpha
                    horiz_offset_used = horiz_offset
                else:
                    img = np.concatenate([img, np.ones_like(alpha)], axis=2)
                    # # clip image
                    # img = img[
                    #       rgb_clip_udlr[0]:rgb_clip_udlr[1],
                    #       rgb_clip_udlr[2]:rgb_clip_udlr[3],
                    #       :]
                    # # resize to width = 1/5 of total width
                    # resized_w = total_width // 5 + (m_id < total_width % 5)
                    # resized_h = int(resized_w * img.shape[0] / img.shape[1])
                    # # resized_h = img_hw
                    # # resized_w = int(img_hw * img.shape[1] / img.shape[0])
                    # img = cv2.resize(img, (resized_w, resized_h))
                    # # offset_used = img_hw - resized_w
                    # horiz_offset_used = 0
                # combine horizontally
                if attribute_img is None:
                    attribute_img = img
                else:
                    attribute_img = combine_horizontally(
                        attribute_img, img,
                        offset=0
                    )
                    # attribute_img = combine_horizontally(
                    #     attribute_img, img,
                    #     offset=horiz_offset_used)
            # utils.write_image(
            #     tgt_dir / f'{attribute}_{img_id:03d}.png',
            #     attribute_img,
            # )
            if whole_img is None:
                # whole_img = attribute_img[-vert_offsets[a_id]:, ...]
                whole_img = attribute_img
            else:
                # if attribute == 'rgb_spec':
                #     attribute_img = attribute_img[:vert_offsets[4], ...]
                # elif attribute == 'rgb':
                #     # if too short, add blank to both side
                #     if attribute_img.shape[1] < whole_img.shape[1]:
                #         blank = np.zeros(
                #             (attribute_img.shape[0],
                #              (whole_img.shape[1] - attribute_img.shape[1]) // 2,
                #              4),
                #             dtype=np.float32)
                #         attribute_img = np.concatenate(
                #             [blank, attribute_img, blank], axis=1)
                #     else:
                #         # cut short
                #         attribute_img = attribute_img[:,
                #                         (attribute_img.shape[1] - whole_img.shape[
                #                             1]) // 2:
                #                         (attribute_img.shape[1] + whole_img.shape[
                #                             1]) // 2,
                #                         ...]
                # whole_img = combine_vertically(
                #     whole_img, attribute_img,
                #     offset=vert_offsets[a_id])
                whole_img = combine_vertically(
                    whole_img, attribute_img,
                    offset=0
                )
        
        utils.write_image(
            tgt_dir / f'{scene}_{img_id:03d}.png',
            whole_img,
        )


def combine_synthetic_scene_examples():
    img_hw = 512
    data_root_dir = Path(r'F:\Datasets\VMINer_Synthetic')
    scenes = [
        'bag',
        'barrelsofa',
        'hotdog',
        'lego',
        'shoes',
        'trooper',
    ]
    split = 'train'
    
    n_cols = 3
    n_rows = 3
    n_imgs = n_cols * n_rows
    
    for scene in scenes:
        tgt_dir = (Path(r'E:\Codes\VMINer\comparison_results')
                   / f'inputs')
        tgt_dir.mkdir(exist_ok=True, parents=True)
        # get all img folders
        src_data_dir1 = data_root_dir / scene / f'{scene}_F2N0' / split
        img_folders = [x for x in src_data_dir1.iterdir() if x.is_dir() and
                       x.stem.startswith(split + '_')]
        src_data_dir2 = data_root_dir / scene / f'{scene}_F1N1' / split
        img_folders += [x for x in src_data_dir2.iterdir() if x.is_dir() and
                        x.stem.startswith(split + '_')]
        # uniformly sample
        img_ids = np.linspace(0, len(img_folders) - 1, n_imgs, dtype=np.int32)
        img_folders = [sorted(img_folders)[x] for x in img_ids]
        
        all_imgs = []
        for r in range(n_rows):
            row_imgs = []
            for c in range(n_cols):
                fg_path = img_folders[r * n_cols + c] / 'rgba.png'
                bg_path = img_folders[r * n_cols + c] / 'background.png'
                fg = utils.read_image(fg_path)
                bg = utils.read_image(bg_path)
                # combine
                img = fg[..., :3] * fg[..., 3:4] + bg[..., :3] * (
                        1 - fg[..., 3:4])
                # downsample to img_hw
                img = cv2.resize(img, (img_hw, img_hw), interpolation=cv2.INTER_AREA)
                row_imgs.append(img)
            row_imgs = np.concatenate(row_imgs, axis=1)
            all_imgs.append(row_imgs)
        all_imgs = np.concatenate(all_imgs, axis=0)
        utils.write_image(
            tgt_dir / f'{scene}_inputs.png',
            all_imgs,
        )


def combine_real_scene_examples():
    data_root_dir = Path(r'F:\Datasets\Unformatted_Real\srgb_images_1200')
    scenes = [
        'guanyu',
        'debu'
    ]
    # setting = 'F1N1'
    # split = 'train'
    
    img_hw = 512
    n_cols = 3
    n_rows = 3
    n_imgs = n_cols * n_rows
    
    for scene in scenes:
        # get img groups
        data_folder = data_root_dir / f'{scene}'
        # img_folders = [x for x in data_folder.iterdir() if x.is_dir() and
        #                x.stem.startswith(split + '_')]
        img_paths = [x for x in data_folder.iterdir() if x.is_file() and
                     x.suffix == '.png']
        # uniformly sample
        img_ids = np.linspace(0, len(img_paths) - 1, n_imgs, dtype=np.int32)
        # img_folders = [sorted(img_folders)[x] for x in img_ids]
        img_paths = [sorted(img_paths)[x] for x in img_ids]
        
        all_imgs = []
        for r in range(n_rows):
            row_imgs = []
            for c in range(n_cols):
                # img_path = img_folders[r * n_cols + c] / 'rgba.png'
                # img = utils.read_image(img_path)
                img = utils.read_image(img_paths[r * n_cols + c])
                img = cv2.resize(
                    img, (img_hw, img_hw), interpolation=cv2.INTER_AREA
                )
                row_imgs.append(img)
            row_imgs = np.concatenate(row_imgs, axis=1)
            all_imgs.append(row_imgs)
        
        all_imgs = np.concatenate(all_imgs, axis=0)
        res_dir = Path(
            r'E:\Codes\VMINer\comparison_results\inputs'
        )
        res_dir.mkdir(exist_ok=True, parents=True)
        utils.write_image(
            res_dir / f'{scene}_inputs.png',
            all_imgs,
        )


def combine_comparison_video():
    img_hw = 800
    n_views = 200
    
    src_dir = Path(r'E:\Codes\VMINer\comparison_results')
    scenes = [
        # 'bag',
        # 'barrelsofa',
        # 'hotdog',
        # 'lego',
        # 'shoes',
        # 'trooper',
        'guanyu',
        'debu',
    ]
    # img_id = 182
    tgt_dir_root = Path(r'E:\Codes\VMINer\comparison_results\comp_video')
    
    attributes = [
        'albedo',
        'normal',
        'rgb',
        'rgb_spec'
    ]
    methods = [
        # 'reference_si',
        'NearLight_F1N1_si',
        'wildlight_F1N1_si',
        'tensoir_F1N0_si',
        # 'tensoir_F2N0_si',
        'nvdiffrecmc_F1N0_si'
    ]
    
    for scene in scenes:
        tgt_dir = tgt_dir_root
        tgt_dir.mkdir(exist_ok=True, parents=True)
        
        # make video for each component
        # left-top: reference
        # right-top: NearLight
        # left-bottom: wildlight
        # right-bottom: tensoir
        # space: 50 pixel
        for attribute in attributes:
            video_path = tgt_dir / f'{scene}_{attribute}.mp4'
            video_writer = imageio.get_writer(video_path, fps=30)
            for img_id in tqdm(range(n_views), desc=f'{scene}_{attribute}'):
                # concatenate
                # imgs = {}
                imgs = []
                for method in methods:
                    file_name = f'{attribute}_{img_id:03d}.png'
                    file_path = src_dir / scene / method / file_name
                    img = utils.read_image(file_path)
                    imgs.append(img)
                if attribute != 'rgb':
                    alpha = imgs[0][..., 3:4]
                    for i in range(len(imgs)):
                        imgs[i][..., 0:3] *= alpha
                n_channels = imgs[0].shape[2]
                row1 = np.concatenate(
                    [imgs[0],
                     np.zeros((img_hw, 64, n_channels)),
                     imgs[1]],
                    axis=1
                )
                row2 = np.concatenate(
                    [imgs[2],
                     np.zeros((img_hw, 64, n_channels)),
                     imgs[3]],
                    axis=1
                )
                img = np.concatenate(
                    [row1, np.zeros((80, img_hw * 2 + 64, n_channels)), row2], axis=0
                )
                # # save
                # utils.write_image(
                #     tgt_dir / f'{attribute}_{img_id:03d}.png',
                #     img,
                # )
                img = (img * 255).astype(np.uint8)
                video_writer.append_data(img)
            video_writer.close()


if __name__ == '__main__':
    # combine_synthetic_scene_examples()
    # combine_comparison_figure()
    # combine_real_scene_examples()
    combine_comparison_video()
