import csv
from pathlib import Path
import json

import torch
import numpy as np
from tqdm import tqdm

from internal import metrics, utils

def read_imgs(img_dir: Path, img_id: int):
    file_names = {
        'pbr_rgb': f'rgb_{img_id:03d}.png',
        'base_color': f'albedo_{img_id:03d}.png',
        'normal': f'normal_{img_id:03d}.png',
        'pbr_rgb_spec': f'rgb_spec_{img_id:03d}.png',
        'pbr_rgb_diff': f'rgb_diff_{img_id:03d}.png',
    }

    imgs = {}
    for key in file_names.keys():
        if not (img_dir / file_names[key]).exists():
            imgs[key] = None
            continue
        # read gt and pred
        img = utils.read_image(img_dir / file_names[key])
        img = torch.from_numpy(img).to(device)
        if key == 'normal':
            img = (img - 0.5) * 2
        imgs[key] = img
    return imgs


if __name__ == '__main__':
    result_base_dir = Path(r'E:\Codes\VMINer\comparison_results')
    src_data_dir = Path(r'F:\Datasets\VMINer_Synthetic')
    scene_name = 'atom'
    scene_dir = result_base_dir / scene_name
    test_data_dir = src_data_dir / 'lego' / ('lego' + '_test') / 'test'
    # test_data_dir = src_data_dir / scene_name / (scene_name + '_test') / 'test'
    # method_name = 'nearlight-1Far+Flash_Long'
    for method_name in [
        # 'NearLight_F1N1',
        'NearLight_F2N2',
        # 'wildlight_F1N1',
        # 'nvdiffrecmc_F1N0',
        # 'tensoir_F1N0',
        # 'tensoir_F2N0',
        # 'reference',
        # 'wolcon',
        # 'modcnear',
        # 'woencmat',
        # 'NearLight_F2',
        # 'NearLight_F10',
        # 'NearLight_F100',
        # 'NearLight_Flash1',
        # 'NearLight_Spot1',
        # 'NearLight_Flash1Spot1',
        # 'NearLight_noind',
        # 'NearLight_novis',
        # 'NearLight_SPP10',
        # 'NearLight_SPP80',
        # 'NearLight_SPP320',
    ]:
        # method_name = 'NearLight_F1N1'
        # method_name = 'wildlight'
        pred_dir = scene_dir / method_name
        out_dir = scene_dir / f'{method_name}_si'
        out_dir.mkdir(exist_ok=True, parents=True)
        # gt_dir = scene_dir / 'reference'

        device = torch.device('cuda:0')

        # build metrics fn
        metrics_fn = metrics.MetricsGatherer(
            measuring_list=[
                'pbr_rgb',
                'pbr_rgb_diff',
                'pbr_rgb_spec',
                'albedo',
                'normal',
            ],
            net_device=device,
            within_mask=False,
            compute_extra_metrics=True,
        )
        metrics_total = {}

        n_imgs = 200

        # all_gt_imgs = []
        # all_pred_imgs = []

        # gt_albedo_sum = torch.zeros(3, device=device, dtype=torch.float32)
        # pred_albedo_sum = torch.zeros(3, device=device, dtype=torch.float32)
        # gt_rgb_sum = torch.zeros(3, device=device, dtype=torch.float32)
        # pred_rgb_sum = torch.zeros(3, device=device, dtype=torch.float32)
        #
        # # read images and accumulate sum for RGB and albedo
        # for img_id in tqdm(range(n_imgs)[::10]):
        #     # if not (img_id % 4 == 0):
        #     #     continue
        #     # for imgs, img_dir in zip([gt_imgs, pred_imgs], [gt_dir, pred_dir]):
        #     gt_imgs = read_imgs(
        #         scene_dir / 'reference',
        #         img_id)
        #     pred_imgs = read_imgs(pred_dir, img_id)
        #
        #     gt_imgs['alpha'] = gt_imgs['pbr_rgb'][..., -1:]
        #     alpha = gt_imgs['alpha']
        #     gt_imgs['pbr_rgb'] = gt_imgs['pbr_rgb'][..., :3] * alpha
        #     pred_imgs['pbr_rgb'] = pred_imgs['pbr_rgb'][..., :3] * alpha
        #     pred_imgs['base_color'] = pred_imgs['base_color'][..., :3] * alpha
        #
        #     # for key in file_names.keys():
        #     #     for imgs in [gt_imgs, pred_imgs]:
        #     #         imgs[key] = imgs[key][..., :3] * gt_imgs['alpha']
        #
        #     # let rgb mean intensity equal to gt
        #     gt_rgb_lin: torch.Tensor = utils.srgb2linrgb(gt_imgs['pbr_rgb']) * alpha
        #     pred_rgb_lin: torch.Tensor = (utils.srgb2linrgb(pred_imgs['pbr_rgb']) *
        #                                   alpha)
        #     gt_rgb_sum += gt_rgb_lin.sum(dim=[0, 1])
        #     pred_rgb_sum += pred_rgb_lin.sum(dim=[0, 1])
        #
        #     gt_albedo_lin = utils.srgb2linrgb(gt_imgs['base_color']) * alpha
        #     pred_albedo_lin = utils.srgb2linrgb(pred_imgs['base_color']) * alpha
        #     gt_albedo_sum += gt_albedo_lin.sum(dim=[0, 1])
        #     pred_albedo_sum += pred_albedo_lin.sum(dim=[0, 1])
        #
        # print('gt_albedo_sum: ', gt_albedo_sum)
        # print('pred_albedo_sum: ', pred_albedo_sum)
        # print('gt_rgb_sum: ', gt_rgb_sum)
        # print('pred_rgb_sum: ', pred_rgb_sum)
        #
        # # compute global scale factor for rgb
        # albedo_scale = gt_albedo_sum / pred_albedo_sum
        # rgb_scale = gt_rgb_sum / pred_rgb_sum

        albedo_scale, rgb_scale = 1, 1

        csv_filename = scene_dir / f'{method_name}_metrics.csv'
        # while csv_filename.exists():
        #     csv_filename = csv_filename.with_name(csv_filename.stem + '_new.csv')

        # read scale from CSV

        # # if csv file exists, read it
        # if method_name == 'reference':
        #     rgb_scale = 1
        #     albedo_scale = 1
        # else:
        #     with open(csv_filename, 'r') as csvfile:
        #         reader = csv.reader(csvfile)
        #         row = reader.__next__()
        #         row = reader.__next__()
        #         row = reader.__next__()
        #         # rgb: N~P, albedo: Q~S
        #         rgb_scale = torch.tensor([float(row[13]), float(row[14]), float(row[15])],
        #                                     device=device)
        #         albedo_scale = torch.tensor([float(row[16]), float(row[17]), float(row[18])],
        #                                     device=device)

        print(f'rgb_scale: {rgb_scale}, albedo_scale: {albedo_scale}')

        # # read background
        # bg_img = utils.read_image(scene_dir / 'background.png')

        # read transform matrix
        meta_path = test_data_dir / 'transforms_test.json'
        # meta_path = gt_dir / 'transforms_test.json'
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # compute metrics
        for img_id in tqdm(range(n_imgs)[160:180]):
            # if not (img_id % 4 == 0):
            #     continue
            # if not (img_id >= 20):
            #     continue
            # multiplier = gt_rgb_lin.mean(dim=[0, 1], keepdim=True) / \
            #              pred_rgb_lin.mean(dim=[0, 1], keepdim=True)

            # # read bg from test data
            # bg_img = utils.read_image(
            #     test_data_dir / f'test_{img_id:03d}' / 'background.png')
            # bg_img = utils.read_image(
            #     scene_dir / 'NearLight_F1N1' / f'background_{img_id:03d}.png'
            # )

            pred_imgs = read_imgs(pred_dir, img_id)
            # gt_imgs = read_imgs(scene_dir / 'reference', img_id)
            # gt_imgs['alpha'] = gt_imgs['pbr_rgb'][..., -1:]
            # alpha = gt_imgs['alpha']
            # gt_imgs['pbr_rgb'] = gt_imgs['pbr_rgb'][..., :3] * alpha
            # # alpha = torch.ones_like(pred_imgs['pbr_rgb'][..., -1:])
            #
            # for key in pred_imgs.keys():
            #     pred_imgs[key] = pred_imgs[key][..., :3]
            #
            # pred_imgs['pbr_rgb'] = pred_imgs['pbr_rgb'][..., :3] * alpha
            # # pred_imgs['base_color'] = pred_imgs['base_color'][..., :3] * alpha
            #
            # for key in ['pbr_rgb', 'pbr_rgb_diff', 'pbr_rgb_spec']:
            #     value = pred_imgs[key]
            #     value = utils.srgb2linrgb(value) * rgb_scale
            #     if key == 'pbr_rgb_spec':
            #         value = value * 3
            #     value = utils.linrgb2srgb(value.clamp(0, 1))
            #     pred_imgs[key] = value
            # for key in ['base_color']:
            #     value = pred_imgs[key]
            #     value = utils.srgb2linrgb(value) * albedo_scale
            #     value = utils.linrgb2srgb(value.clamp(0, 1))
            #     pred_imgs[key] = value

            # # save images with alpha channel
            # alpha_img = alpha.cpu().numpy()
            # rgb_img = pred_imgs['pbr_rgb'].cpu().numpy()
            # # rgb_img = np.concatenate([rgb_img, alpha_img], axis=-1)
            # # rgb_img = rgb_img + bg_img * (1 - alpha_img)
            # rgb_diff_img = pred_imgs['pbr_rgb_diff'].cpu().numpy()
            # rgb_diff_img = np.concatenate([rgb_diff_img, alpha_img], axis=-1)
            # rgb_spec_img = pred_imgs['pbr_rgb_spec'].cpu().numpy()
            # rgb_spec_img = np.concatenate([rgb_spec_img, alpha_img], axis=-1)
            # albedo_img = pred_imgs['base_color'].cpu().numpy()
            # albedo_img = np.concatenate([albedo_img, alpha_img], axis=-1)
            normal_img = pred_imgs['normal']
            alpha_img = ((normal_img ** 2).sum(dim=2, keepdim=True)) > 0.1
            # transform normal from world space to camera space
            # read extrinsic matrix
            meta_frame = meta['frames'][img_id]
            cam2world_mat = torch.tensor(
                meta_frame["transform_matrix"],
                dtype=torch.float32, device=device)
            # read get camera extrinsic matrix from camera space (opencv coord)
            # to world space in column-major
            cam2world_R = cam2world_mat[:3, :3].T  # to row-major
            world2cam_R = torch.linalg.inv(cam2world_R)
            # transform normal from world space to camera space
            normal_img = normal_img @ world2cam_R
            # from opencv coord to blender coord
            normal_img[..., 1] *= -1
            normal_img[..., 2] *= -1
            normal_img = torch.nn.functional.normalize(normal_img, dim=-1, p=2)

            normal_img = ((normal_img + 1) / 2).cpu().numpy()
            alpha_img = alpha_img.float().cpu().numpy()
            normal_img = np.concatenate([normal_img, alpha_img], axis=-1)

            # utils.write_image(out_dir / f'rgb_{img_id:03d}.png',
            #                     rgb_img)
            # utils.write_image(out_dir / f'rgb_diff_{img_id:03d}.png',
            #                     rgb_diff_img)
            # utils.write_image(out_dir / f'rgb_spec_{img_id:03d}.png',
            #                     rgb_spec_img)
            # utils.write_image(out_dir / f'albedo_{img_id:03d}.png',
            #                     albedo_img)
            utils.write_image(out_dir / f'normal_{img_id:03d}.png',
                                normal_img)

            # pred_imgs['pbr_rgb'] *= multiplier
            # pred_imgs['pbr_rgb_diff'] *= multiplier
            # pred_imgs['pbr_rgb_spec'] *= multiplier
            # let albedo rgb mean intensity equal to gt
            # multiplier = (gt_imgs['base_color'] * alpha
            #               ).mean(dim=[0, 1], keepdim=True) / \
            #              (pred_imgs['base_color'] * alpha
            #               ).mean(dim=[0, 1], keepdim=True)
            # pred_imgs['base_color'] = (pred_imgs['base_color'] * multiplier).clamp(0, 1)

        #     # compute metrics
        #     # change name for gt
        #     gt_imgs['rgb'] = gt_imgs['pbr_rgb']
        #     del gt_imgs['pbr_rgb']
        #     gt_imgs['rgb_spec'] = gt_imgs['pbr_rgb_spec']
        #     del gt_imgs['pbr_rgb_spec']
        #     gt_imgs['rgb_diff'] = gt_imgs['pbr_rgb_diff']
        #     del gt_imgs['pbr_rgb_diff']
        #     gt_imgs['albedo'] = gt_imgs['base_color']
        #     del gt_imgs['base_color']
        #     metric = metrics_fn(
        #         gt=gt_imgs,
        #         pred=pred_imgs,
        #     )
        #     metric['count'] = torch.tensor(1.0).cpu()
        #     # print(metrics)
        #
        #     for key, value in metric.items():
        #         if key not in metrics_total:
        #             metrics_total[key] = value
        #         else:
        #             metrics_total[key] += value
        #
        # # average
        # for key, value in metrics_total.items():
        #     metrics_total[key] /= metrics_total['count']
        #     metrics_total[key] = metrics_total[key].item()
        # del metrics_total['count']
        #
        # print(metrics_total)
        #
        # # format to .csv file
        # compo_metrics_list = [
        #     'normal_MAngE',
        #     'albedo_PSNR',
        #     'albedo_SSIM',
        #     'albedo_LPIPS',
        #     'pbr_rgb_PSNR',
        #     'pbr_rgb_SSIM',
        #     'pbr_rgb_LPIPS',
        #     'pbr_rgb_diff_PSNR',
        #     'pbr_rgb_diff_SSIM',
        #     'pbr_rgb_diff_LPIPS',
        #     'pbr_rgb_spec_PSNR',
        #     'pbr_rgb_spec_SSIM',
        #     'pbr_rgb_spec_LPIPS',
        # ]
        # component_list = []
        # metric_list = []
        # # decompose metrics
        # for compo_metric in compo_metrics_list:
        #     # metric: rightmost part
        #     metric = compo_metric.split('_')[-1]
        #     # component: remaining part
        #     component = '_'.join(compo_metric.split('_')[:-1])
        #     component_list.append(component)
        #     metric_list.append(metric)
        #
        # with open(csv_filename, 'w', newline='') as csvfile:
        #     # add column names
        #     writer = csv.writer(csvfile)
        #     # write compo and metric
        #     writer.writerow(component_list)
        #     writer.writerow(metric_list)
        #     # write values
        #     row = []
        #     for compo_metric in compo_metrics_list:
        #         row.append(metrics_total[compo_metric])
        #     writer.writerow(row)

