import shutil
from pathlib import Path

from tqdm import tqdm


def get_target_filenames(img_id: int):
    return {
        'rgb': f'rgb_{img_id:03d}.png',
        'base_color': f'albedo_{img_id:03d}.png',
        'normal': f'normal_{img_id:03d}.png',
        'rgb_spec': f'rgb_spec_{img_id:03d}.png',
        'rgb_diff': f'rgb_diff_{img_id:03d}.png',
    }

def move_gt_files_to_comparison_folder(
        test_set_dir: Path,
        comparison_folder: Path,
):
    """Move ground truth files to comparison folder."""
    split = 'test'
    test_img_folders = [x for x in test_set_dir.iterdir() if
                        x.stem.startswith(split + '_')]
    test_img_folders.sort()
    print(f'Copying {len(test_img_folders)} files from {test_set_dir} to '
          f'{comparison_folder}...')
    target_dir = comparison_folder / 'reference'
    target_dir.mkdir(exist_ok=True, parents=True)
    for img_id, img_folder in tqdm(
            enumerate(test_img_folders),
            total=len(test_img_folders),
    ):
        file_names_src = {
            'rgb': 'rgba.png',
            'base_color': 'albedo.png',
            'normal': 'normal.png',
            'rgb_spec': 'rgb_spec.png',
            'rgb_diff': 'rgb_diff.png',
        }
        file_names_tgt = get_target_filenames(img_id)
        for key in file_names_src.keys():
            shutil.copy(
                img_folder / file_names_src[key],
                target_dir / file_names_tgt[key],
            )


def move_nearlight_to_comparison_folder(
        setting_name: str,
        pred_dir: Path,
        comparison_folder: Path,
):
    target_dir = comparison_folder / f'nearlight-{setting_name}'
    target_dir.mkdir(exist_ok=True, parents=True)

    n_imgs = 200
    for img_id in tqdm(range(n_imgs)):
        file_names_src = {
            'rgb': f'pbr_rgb_im{img_id:03d}.png',
            'base_color': f'albedo_im{img_id:03d}.png',
            'normal': f'normal_im{img_id:03d}.png',
            'rgb_spec': f'pbr_rgb_spec_im{img_id:03d}.png',
            'rgb_diff': f'pbr_rgb_diff_im{img_id:03d}.png',
        }
        file_names_tgt = get_target_filenames(img_id)
        for key in file_names_src.keys():
            shutil.copy(
                pred_dir / file_names_src[key],
                target_dir / file_names_tgt[key],
            )

def move_tensoir_to_comparison_folder(
        setting_name: str,
        pred_dir: Path,
        comparison_folder: Path,
):
    target_dir = comparison_folder / f'tensoir_{setting_name}'
    target_dir.mkdir(exist_ok=True, parents=True)

    # n_imgs = 200
    for img_id in tqdm(range(111, 115)):
        source_folder = pred_dir / f'test_{img_id:03d}'
        file_names_src = {
            'rgb': f'pbr_rgb.png',
            'base_color': f'albedo_gamma_corrected.png',
            'normal': f'normal.png',
            'rgb_spec': f'pbr_rgb_spec.png',
            'rgb_diff': f'pbr_rgb_diff.png',
        }
        file_names_tgt = get_target_filenames(img_id - 111)
        for key in file_names_src.keys():
            shutil.copy(
                source_folder / file_names_src[key],
                target_dir / file_names_tgt[key],
            )

if __name__ == '__main__':
    # scene_name = 'ficus'
    test_data_dir = Path(
        # r'F:\Datasets\VMINer_Synthetic\ficus_F1N1Hard\test'
        r"F:\Datasets\VMINer_Synthetic\rebuttal\hotdog_test\test"
        # r'E:\Codes\SyntheticDataGeneration\output\hotdog_F2N1Hard\test'
    )
    comparison_folder = Path(r'E:\Codes\VMINer\comparison_results\hotdog')
    move_gt_files_to_comparison_folder(
        test_set_dir=test_data_dir,
        comparison_folder=comparison_folder,
    )
    # move_nearlight_to_comparison_folder(
    #     setting_name='1Far+Flash_Long',
    #     pred_dir=Path(
    #         r'E:\Codes\VMINer\exp_log\\'
    #         r'hotdog_F1N1Hard_long-2023Oct23Mon-234837'
    #         r'\visuals\novel_light\iter_35000'
    #     ),
    #     comparison_folder=comparison_folder,
    # )
    # move_tensoir_to_comparison_folder(
    #     setting_name='F2N0',
    #     pred_dir=Path(
    #         # r'E:\Codes\FromGithub\TensoIR\scripts\relighting\hotdog'
    #         r'E:\Codes\VMINer\comparison_results\lego\lego_F2N0_vis'
    #     ),
    #     comparison_folder=comparison_folder,
    # )

