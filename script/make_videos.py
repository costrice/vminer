from pathlib import Path
from typing import *

import imageio
import imageio.v3 as iio
import numpy as np
from PIL import Image
from tqdm import tqdm

comparison_dir = Path(r'E:\Codes\VMINer\comparison_results')
scene_name = 'GuanYu'
method_name = 'NearLight_F1N1_wbg'
# base_dir = Path(r'E:\Codes\VMINer\log')
# dataset_base_dir = Path(r'F:\Datasets\VMINer_Synthetic')
# expr_name = 'hotdog_F0N1_near_mul2-2023Oct18Wed-151408'
# view_dataset_name = '_'.join(expr_name.split('_')[:2])
# light_dataset_name = 'hotdog_relight_test'


def make_video_from_images(
        save_prefix: str = '',
        img_suffix: str = '',
        img_prefix: str = '',
):
    img_dir = comparison_dir / scene_name / method_name
    out_dir = img_dir / 'videos'
    out_dir.mkdir(exist_ok=True, parents=True)
    # collect images
    images = []
    img_paths = sorted(img_dir.glob(f'{img_prefix}*{img_suffix}.png'))
    if img_prefix == '':
        # check suffix
        img_paths = [
            img_path for img_path in img_paths
            if '_'.join(img_path.stem.split('_')[1:]) == img_suffix
        ]
    if img_suffix == '':
        # check prefix
        img_paths = [
            img_path for img_path in img_paths
            if '_'.join(img_path.stem.split('_')[:-1]) == img_prefix
        ]

    if len(img_paths) == 0:
        print(f'No images found in {img_dir} '
              f'with prefix "{img_prefix}" and suffix "{img_suffix}".')
        return
    for img_path in img_paths:
        images.append(imageio.v2.imread(img_path))
    # write to video
    video_path = out_dir / f'{save_prefix}{img_prefix}{img_suffix}.mp4'
    video_writer = imageio.get_writer(video_path, fps=30)
    for img in images:
        video_writer.append_data(img)
    video_writer.close()


def combine_images_to_video():
    scene_properties = [
        'albedo',
        'normal',
        'rgb',
        'rgb_diff',
        'rgb_spec',
    ]
    for scene_property in scene_properties:
        make_video_from_images(
            img_prefix=scene_property,
        )


# def combine_images_to_video():
#     # combine rendered images to a video
#     visual_dir = base_dir / expr_name / 'visuals'
#     for img_dirname in ['novel_view', 'novel_light']:
#         img_dir = visual_dir / img_dirname / 'iter_07000'
#         print(f'Processing {img_dir}')
#
#         output_dir = visual_dir / img_dirname / 'videos'
#         output_dir.mkdir(exist_ok=True, parents=True)
#
#         # exp_name = 'F3N3Hard_no_consistency_'
#
#         scene_properties = [
#             # 'opacity',
#             # 'depth',
#             # 'normal',
#             'shading_normal',
#             'nr_rgb',
#             'nr_rgb_far',
#             'nr_rgb_near',
#             'albedo',
#             'roughness',
#             'pbr_rgb',
#             'pbr_rgb_far',
#             # 'direct_shading_far',
#             # 'visibility_far',
#             # 'indirect_rgb_far',
#             'pbr_rgb_near',
#             # 'direct_shading_near',
#             # 'visibility_near',
#             # 'indirect_rgb_near',
#         ]
#
#         pbar = tqdm(scene_properties, desc='Making videos')
#         for scene_property in pbar:
#             pbar.set_description(f'Making video for {scene_property}')
#             make_video_from_images(
#                 img_dir=img_dir,
#                 out_dir=output_dir,
#                 # save_prefix=exp_name,
#                 save_prefix='',
#                 img_prefix=scene_property,
#                 img_suffix='',
#                 # img_prefix='',
#                 # img_suffix=scene_property,
#             )


def concat_images(
        images: List[Image.Image],
        size: Tuple[int, int],
        shape: Optional[Tuple[int, int]] = None,
):
    """Concatenates images into a grid.

    References:
        https://gist.github.com/njanakiev/1932e0a450df6d121c05069d5f7d7d6f

    Args:
        # image_paths: list of image paths
        images: list of images
        size: the size of each image
        shape: (n_row, n_col) the shape of the grid

    Returns:

    """
    # source:
    # Open images and resize them
    width, height = size
    # images = map(Image.open, image_paths)
    # images = [ImageOps.fit(image, size, Image.ANTIALIAS)
    #           for image in images]

    # Create canvas for the final image with total size
    shape = shape if shape else (1, len(images))
    image_size = (width * shape[1], height * shape[0])
    image = Image.new('RGB', image_size)

    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            image.paste(images[idx], offset)

    return image


def combine_videos_spatially():
    grid_shape = (2, 3)

    video_dir = comparison_dir / scene_name / method_name / 'videos'

    video_paths = [
        video_dir / 'albedo.mp4',
        video_dir / 'normal.mp4',
        None,
        video_dir / 'rgb.mp4',
        video_dir / 'rgb_diff.mp4',
        video_dir / 'rgb_spec.mp4',
    ]

    video_metas = []
    for video_path in video_paths:
        if video_path is None:
            video_metas.append(None)
        else:
            video_metas.append(iio.immeta(video_path, plugin='pyav'))

    n_frames = 200

    videos = []
    for video_path in video_paths:
        if video_path is None:
            # add all black frame
            videos.append(np.zeros((n_frames, 800, 800, 3), dtype=np.uint8))
        else:
            videos.append(iio.imread(video_path, plugin='pyav'))

    # write to video
    out_video_path = video_dir / 'combined.mp4'
    video_writer = imageio.get_writer(out_video_path, fps=30)

    for frame_id in tqdm(range(n_frames), desc='Concatenating frames'):
        # read one frame from all videos
        frames = []
        for video, video_meta in zip(videos, video_metas):
            # If total number of frames is less than
            # the frame_id, then use the last frame
            video_frames = len(video)
            frame_id_to_read = frame_id * video_frames // n_frames
            frame = video[frame_id_to_read]
            frames.append(frame)
        # to PIL Image
        frames = [Image.fromarray(frame) for frame in frames]
        # concatenate frames
        image = concat_images(
            images=frames,
            size=(800, 800),
            shape=grid_shape)
        # to numpy array
        image = np.array(image)
        video_writer.append_data(image)

    video_writer.close()

    print('Results written to', out_video_path)


if __name__ == '__main__':
    combine_images_to_video()
    combine_videos_spatially()
