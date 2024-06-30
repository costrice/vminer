"""
Evaluation functions for the scene.
"""
import json
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

import internal.datasets as dataset
from internal import global_vars, metrics, renderers, scenes, utils


@torch.no_grad()
def evaluate_scene(
        scene: scenes.SceneModel,
        dataset_test: dataset.VMINerDataset,
        under_novel_light: bool,
        save_dir: Path,
        metrics_fn: metrics.MetricsGatherer,
        do_pbr: bool,
        show_progress: bool,
):
    """Evaluate the reconstructed scene model on the test set.

    Args:
        scene: the reconstructed scene to evaluate
        dataset_test: the test set
        under_novel_light: whether to render under given novel lights in the
            test set
        save_dir: the directory to save the rendered images
        metrics_fn: the function to compute the metrics
        do_pbr: whether to render the PBR images
        show_progress: whether to show the progress bar

    Returns:
        None
    """
    # read global variables
    it = global_vars.cur_iter
    args = global_vars.args
    during_training = global_vars.training
    writer = global_vars.writer
    # log_to_wandb = args.log_to_wandb
    n_visual = args.n_visual_train if during_training else args.n_visual_test
    white_bg = args.white_bg
    
    device = scene.device
    
    scene.eval()
    global_vars.training = False
    
    if under_novel_light:
        # disable nr_rgb metric
        if 'nr_rgb' in metrics_fn.measuring_list:
            metric_have_nr_rgb = True
            metrics_fn.delete_from_measuring_list(['nr_rgb'])
        else:
            metric_have_nr_rgb = False
    
    # update scene lighting
    scene.update_baked_envmap()
    
    n_test = min(n_visual, len(dataset_test))
    
    metrics_dict_total = {'count': torch.tensor(0.0)}
    if save_dir is not None:
        print(f'\nRendering {n_test} images into {save_dir}...')
    
    # set up maps to be saved (check renderers.all_attr_names to get the names)
    do_nr_rgb = not under_novel_light
    if under_novel_light:
        assert do_pbr, 'Must do PBR if rendering under novel lights'
        have_far = True
        have_near = False  # for now, only render under far lights TODO
        far_lights_pbr = dataset_test.far_lights_gt
        far_lights_pbr.to_device(device)
        near_lights_pbr = None
    else:  # render under lights in the tensorf model
        have_far = scene.n_far_lights > 0
        have_near = scene.n_near_lights > 0
        far_lights_pbr = scene.far_lights
        near_lights_pbr = scene.near_lights_pbr
    
    do_nr_rgb_far = do_nr_rgb and have_far
    do_nr_rgb_near = do_nr_rgb and have_near
    do_pbr_far = do_pbr and have_far
    do_pbr_near = do_pbr and have_near
    if not (do_nr_rgb_far or do_nr_rgb_near or do_pbr_far or do_pbr_near):
        raise ValueError('No lights to render!')
    
    attr_names_to_eval = []
    attr_names_to_eval += [
        'depth',
        'opacity',
        'normal'
    ]
    if do_nr_rgb_far or do_nr_rgb_near:
        attr_names_to_eval += [
            'nr_rgb',
        ]
    # if do NR both far and near, write them separately
    if do_nr_rgb_far and do_nr_rgb_near:
        attr_names_to_eval += [
            'nr_rgb_far',
            'nr_rgb_near',
        ]
    if do_pbr_far or do_pbr_near:
        # if do pbr, have to render intrinsics
        attr_names_to_eval += [
            'pbr_rgb',
            'pbr_rgb_spec',
            'pbr_rgb_diff',
            'base_color',
            'roughness',
            'specular',
        ]
        # if do PBR both far and near, write them separately
        if do_pbr_near and do_pbr_far:
            attr_names_to_eval += [
                'pbr_rgb_far',
                'pbr_rgb_diff_far',
                'pbr_rgb_spec_far',
                'pbr_rgb_near',
                'pbr_rgb_diff_near',
                'pbr_rgb_spec_near',
            ]
    if do_pbr_far:
        attr_names_to_eval += [
            'direct_shading_far',
            'visibility_far',
            'indirect_rgb_far',
        ]
    if do_pbr_near:
        attr_names_to_eval += [
            'direct_shading_near',
            'visibility_near',
            'indirect_rgb_near',
        ]
    
    # evenly get indices from the whole test set
    indices = np.linspace(
        0, len(dataset_test) - 1, n_test,
        dtype=int,
        endpoint=True
    )
    
    if show_progress:
        pbar = tqdm(indices, desc='Rendering')
    else:
        pbar = indices
    
    save_by_iter = n_test > 10
    # else: save by img id
    # save_by_img_id = n_test <= 5
    
    if save_by_iter:
        save_folder = save_dir / f'iter_{it:05d}'
        save_folder.mkdir(parents=True, exist_ok=True)
    
    if do_pbr_far:
        # get all far lights and near lights
        envmap_dicts = []
        if under_novel_light:
            envmaps = far_lights_pbr.to_envmap()
        else:
            envmaps = scene.baked_far_lights.to_envmap()
        for envmap_id, envmap in enumerate(envmaps):
            envmap = (envmaps[envmap_id].detach().cpu().numpy())
            # normalize intensity
            envmap /= envmap.mean() / 0.3
            # envmap /= 15
            envmap_name = dataset_test.far_lights_meta['name'][envmap_id]
            envmap_dicts.append(
                {
                    'name': envmap_name,
                    'envmap': envmap,
                }
            )
        
        if save_by_iter:
            # save all far_lights to this dir
            for envmap_dict in envmap_dicts:
                utils.write_image(
                    save_folder / f'envmap_{envmap_dict["name"]}.hdr',
                    envmap_dict['envmap']
                )
    
    if have_near:
        near_light_info = []
        if not scene.pbr_enabled:
            near_lights = scene.near_lights_nr
            near_light_str = 'neural rendering'
        else:
            near_lights = scene.near_lights_pbr
            near_light_str = 'PBR'
        light_pos_types = near_lights.pos_types
        light_xyz, intensity_params = (
            near_lights.get_actual_params())
        for near_id in range(scene.n_near_lights):
            near_light = {
                'position': 'collocated'
                if light_pos_types[near_id] == 'collocated'
                else light_xyz[near_id].detach().cpu().numpy().tolist(),
                'intensity': intensity_params[near_id]
                .detach().cpu().numpy().tolist(),
            }
            near_light_info.append(near_light)
        print(f'\nNear lights ({near_light_str}) info: ')
        pprint(near_light_info)
        
        if save_by_iter:
            # save all near lights to this dir
            with open(save_folder / 'near_lights.json', 'w') as f:
                json.dump(near_light_info, f)
    
    for i in pbar:
        # get test data
        gt_ray_attrs = dataset_test[i]
        
        h, w = gt_ray_attrs['h'], gt_ray_attrs['w']
        
        # [H*W, 1 or 3 or n_near_lights]
        for key in ['rays', 'rgb', 'alpha', 'albedo', 'normal',
                    'near_light_state', 'rgb_spec', 'rgb_diff']:
            if isinstance(gt_ray_attrs[key], torch.Tensor):
                gt_ray_attrs[key] = gt_ray_attrs[key].to(device)
        n_rays = gt_ray_attrs['rays'].shape[0]
        far_light_idx = gt_ray_attrs['far_light_idx']
        near_light_state = gt_ray_attrs['near_light_state']
        
        ray_attrs_to_eval = {}
        for attr_name in attr_names_to_eval:
            ray_attrs_to_eval[attr_name] = []
        
        ray_attrs_to_eval = renderers.render(
            scene,
            inp_ray_attrs={
                'n_rays': n_rays,
                'rays': gt_ray_attrs['rays'],
                'far_light_idx': far_light_idx,
                'near_light_state': near_light_state,
            },
            is_under_novel_light=under_novel_light,
            far_lights_pbr=far_lights_pbr if under_novel_light else None,
            near_lights_pbr=near_lights_pbr if under_novel_light else None,
            do_pbr=do_pbr,
            white_bg=white_bg,
        )
        
        for attr_name in attr_names_to_eval:
            if attr_name not in ray_attrs_to_eval:
                if i == indices[0]:
                    print(
                        f'Warning: {attr_name} not in ray_attrs_to_eval, '
                        f'ignored.'
                    )
            else:
                attr_value = ray_attrs_to_eval[attr_name]
                attr_value = attr_value.view(h, w, -1)
                ray_attrs_to_eval[attr_name] = attr_value
        
        # build Ground Truth for evaluation
        for key in ['rgb', 'alpha', 'albedo', 'normal', 'rgb_spec', 'rgb_diff']:
            if isinstance(gt_ray_attrs[key], torch.Tensor):
                gt_ray_attrs[key] = gt_ray_attrs[key].view(h, w, -1)
        # add white background
        if white_bg:
            for key in ['rgb', 'rgb_spec', 'rgb_diff']:
                if isinstance(gt_ray_attrs[key], torch.Tensor):
                    gt_ray_attrs[key] += (1 - gt_ray_attrs['alpha'])
        
        if under_novel_light:
            # per-channel adjustment for PBR RGB to account for the intensity
            # ambiguity of the lighting
            value_in_mask = (utils.srgb2linrgb(ray_attrs_to_eval['pbr_rgb'])
                             * gt_ray_attrs['alpha'])
            gt_in_mask = (utils.srgb2linrgb(gt_ray_attrs['rgb'])
                          * gt_ray_attrs['alpha'])
            multiplier = (gt_in_mask.mean(dim=(0, 1), keepdim=True)
                          / value_in_mask.mean(dim=(0, 1), keepdim=True).clamp(1e-5))
            for key in [
                'pbr_rgb', 'pbr_rgb_far', 'pbr_rgb_near',
                'pbr_rgb_spec', 'pbr_rgb_spec_far', 'pbr_rgb_spec_near',
                'pbr_rgb_diff', 'pbr_rgb_diff_far', 'pbr_rgb_diff_near',
            ]:
                if key in attr_names_to_eval:
                    value = ray_attrs_to_eval[key]
                    value = utils.srgb2linrgb(value) * multiplier
                    value = (utils.linrgb2srgb(value.clamp(0, 1))
                             * gt_ray_attrs['alpha'])
                    if white_bg:
                        value += (1 - gt_ray_attrs['alpha'])
                    ray_attrs_to_eval[key] = value
        
        # compute metrics
        metrics_dict = metrics_fn(ray_attrs_to_eval, gt_ray_attrs)
        metrics_dict['count'] = torch.tensor(1.0).cpu()
        
        # accumulate metrics
        if i == 0:
            metrics_dict_total = metrics_dict
        else:
            for key, value in metrics_dict.items():
                metrics_dict_total[key] += value
        
        # save images
        if not save_by_iter:  # save by img id
            save_folder = save_dir / f'img_{i:03d}'
            save_folder.mkdir(parents=True, exist_ok=True)
            # save lights to this dir
            if do_pbr_far:
                envmap_dict = envmap_dicts[far_light_idx]
                utils.write_image(
                    save_folder /
                    f'envmap_{envmap_dict["name"]}_{it:05d}.hdr',
                    envmap_dict['envmap']
                )
            if do_pbr_near:
                with open(save_folder / f'near_lights_{it:05d}.json', 'w') as f:
                    json.dump(near_light_info, f)
        
        # post-process for visualization
        ray_attrs_to_eval['opacity'] = (
            ray_attrs_to_eval['opacity'].detach().cpu().numpy())
        for attr_name in ray_attrs_to_eval:
            if attr_name not in attr_names_to_eval:
                continue
            if (during_training and do_pbr
                    and args.fix_shape_and_radiance_in_pbr
                    and attr_name in ['depth', 'opacity', 'normal', 'nr_rgb',
                                      'nr_rgb_far', 'nr_rgb_near']):
                continue
            map_value = ray_attrs_to_eval[attr_name]
            if attr_name == 'depth':
                map_value = map_value.squeeze(-1)
                map_value, _ = utils.visualize_depth_numpy(
                    map_value.detach().cpu().numpy(),
                    min_max=args.primary_near_far,
                )
                map_value = map_value.astype(np.float32) / 255
            # prepare for saving
            if isinstance(map_value, torch.Tensor):
                map_value = map_value.detach().cpu().numpy()
                # shape is already [H, W, ?]
            if attr_name in ['normal', 'shading_normal']:
                map_value = (map_value + 1) / 2
            if attr_name in ['normal', 'shading_normal', 'depth']:
                map_value *= ray_attrs_to_eval['opacity']
            # ray_attrs_to_eval[attr_name] = map_value
            
            if save_by_iter:
                # save to iter folder if img amount > 5
                utils.write_image(
                    save_folder / f'{attr_name}_im{i:03d}.png',
                    map_value
                )
            else:
                # save to img folder
                utils.write_image(
                    save_folder / f'{attr_name}_it{it:05d}.png',
                    map_value
                )
        
        # # save GT for reference
        # for attr_name in gt_ray_attrs:
        #     map_value = gt_ray_attrs[attr_name]
        #     # prepare for saving
        #     if isinstance(map_value, torch.Tensor):
        #         map_value = map_value.detach().cpu().numpy()
        #         # shape is already [H, W, ?]
        #
        #     utils.write_image(
        #         save_folder / f'{attr_name}_gt_im{i:03d}.png',
        #         map_value)
        
        del ray_attrs_to_eval
    
    # compute average metrics
    count = metrics_dict_total['count']
    for key, value in metrics_dict_total.items():
        value = (value / count).item()
        metrics_dict_total[key] = value
    del metrics_dict_total['count']
    print('metrics_dict_total:')
    for key, value in metrics_dict_total.items():
        print(f'\t{key}: {value:.3f}')
    # log to writer
    if writer is not None:
        prefix = 'Metric_relight/' if under_novel_light else 'Metric/'
        if during_training:
            prefix += 'val/'
        for key, value in metrics_dict_total.items():
            writer.add_scalar(
                prefix + key,
                value,
                it
            )
    
    if during_training:
        scene.train()
        global_vars.training = True
    
    if under_novel_light:
        if metric_have_nr_rgb:
            metrics_fn.extend_measuring_list(['nr_rgb'])
