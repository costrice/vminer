"""
This file contains neural renderer and physically-based renderer for rendering
images using the scene model.
"""
from typing import *

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from internal import (global_vars, lighting, ray_sampler, scenes, utils)


def render(
        scene: scenes.SceneModel,
        inp_ray_attrs: Dict[str, Any],
        do_pbr: bool,
        is_under_novel_light: bool,
        far_lights_pbr: lighting.FarLight = None,
        near_lights_pbr: lighting.NearLight = None,
        white_bg: Union[torch.Tensor, bool] = True,
) -> Dict[str, torch.Tensor]:
    """Render rays and do post-processing.

    Perform tensorial rendering, physically-based rendering, and
    post-processing. If far_lights or near_lights is not None (given), can not
    perform tensorial rendering for radiance.

    Args:
        scene: the scene model
        do_pbr: whether to perform physically-based rendering additionally.
        is_under_novel_light: whether it is under novel light conditions.
            If True, will not perform neural rendering for radiance, and at
            least one of far_lights and near_lights should be not None.
            If False, will perform neural rendering for radiance, and
            far_lights and near_lights will be read from tensorf.
        far_lights_pbr: the given novel far-field light conditions.
        near_lights_pbr: the given novel near-field light conditions.
        inp_ray_attrs: a dict containing the attributes of input rays. Should
            include following keys:
            - n_rays: the number of rays
            - rays: (n_rays, 6) the rays to be rendered
            - far_light_idx: int, the index of the far-field light. -1 means
                no far-field light.
            - near_light_state: (n_near_lights, ) int, the 0-1 states of the
                near-field lights
        white_bg: whether to use white background. Can be specified for each
            ray.

    Returns:
        all_ray_attrs (Dict[str, torch.Tensor]): the rendered attributes
            for input rays
    """
    # read global variables
    it = global_vars.cur_iter
    args = global_vars.args
    is_train = global_vars.training
    device = scene.device
    
    # neural radiance do not model novel light conditions, so if novel light
    # conditions are given, we do not perform neural radiance rendering
    do_nr_rgb = not is_under_novel_light
    if is_under_novel_light:  # light is given
        assert far_lights_pbr is not None or near_lights_pbr is not None, \
            'At least one of far_lights and near_lights should be not None ' \
            'when is_under_novel_light is True'
        have_far = far_lights_pbr is not None
        have_near = near_lights_pbr is not None
    else:
        have_far = scene.n_far_lights > 0
        have_near = scene.n_near_lights > 0
        if args.use_prefiltered_envmap_for_diffuse and not is_train:
            far_lights_pbr = scene.baked_far_lights
        else:
            far_lights_pbr = scene.far_lights
        near_lights_pbr = scene.near_lights_pbr
    
    do_nr_rgb_far = do_nr_rgb and have_far
    do_nr_rgb_near = do_nr_rgb and have_near
    
    n_rays = inp_ray_attrs['n_rays']
    rays = inp_ray_attrs['rays'].to(device)
    far_light_idx = inp_ray_attrs['far_light_idx']  # int
    near_light_state = inp_ray_attrs['near_light_state']  # [n_near_lights, ]
    # near_light_indices = [id for id, state in enumerate(near_light_state)
    #                       if state == 1)
    rays_o = rays[:, :3]
    rays_d = rays[:, 3:6]
    
    if isinstance(white_bg, bool):
        if white_bg:
            white_bg = torch.ones(n_rays, 1, device=device)
        else:
            white_bg = torch.zeros(n_rays, 1, device=device)
    
    all_attrs = {}
    
    # neural render ray attributes
    # basic attributes:
    #   - depth,
    #   - opacity,
    #   - normal,
    # if do_nr_rgb_far is True, also render:
    #   - nr_rgb_far
    # if do_nr_rgb_near is True, also render:
    #   - nr_rgb_near
    # if do_pbr_near or do_pbr_far is True, intrinsic attributes are also
    # rendered, including
    #   - albedo
    #   - roughness
    #   - fresnel
    nr_attrs = {}
    
    # render in chunks to avoid OOM
    chunk_size = args.chunk_size_nr_train if is_train \
        else args.chunk_size_nr_test
    chunk_indices = torch.split(
        torch.arange(n_rays, device=device),
        chunk_size
    )
    cur_n_rays = 0
    for chunk_idx in chunk_indices:
        if do_nr_rgb_near:
            scene.near_lights_nr.current_cam_pos = rays_o[chunk_idx]
        nr_attrs_chunk = scene(
            rays_o=rays_o[chunk_idx],
            rays_d=rays_d[chunk_idx],
            return_rgb_far=do_nr_rgb_far,
            far_light_idx=far_light_idx,
            return_rgb_near=do_nr_rgb_near,
            near_light_state=near_light_state,
            return_material=do_pbr,
            return_jittered=global_vars.need_jittered,
            n_segs=scene.n_ray_segments,
            near_far=scene.near_far,
        )
        for key, value in nr_attrs_chunk.items():
            if key not in nr_attrs:
                nr_attrs[key] = [value]
            else:
                if key == 'sample_ray_indices':
                    value = value + cur_n_rays
                nr_attrs[key].append(value)
        cur_n_rays += chunk_idx.shape[0]
    
    if (scene.training and not
    (global_vars.pbr_enabled and args.fix_shape_and_radiance_in_pbr)):
        # randomly pick points in the aabb to compute eikonal loss
        sampled_pts = scene.sample_volume(n_pts=args.batch_size_train * 4)
        sdf_grad = scene.field.query_sdf(
            sampled_pts,
            with_grad=True,
        )['sdf_grad']
        nr_attrs['sample_sdf_grad'] += [sdf_grad]
    
    # concatenate the chunks
    for key, value in nr_attrs.items():
        nr_attrs[key] = torch.cat(value, dim=0)
    
    all_attrs.update(nr_attrs)
    
    if do_pbr:
        # physically-based rendering for far and near lights
        # attributes: pbr_rgb_*, direct_shading_*, indirect_rgb_*
        pbr_attrs = {}
        
        # render in chunks to avoid OOM
        chunk_size = args.chunk_size_pbr_train if is_train \
            else args.chunk_size_pbr_test
        chunk_indices = torch.split(
            torch.arange(n_rays, device=device),
            chunk_size
        )
        for chunk_idx in chunk_indices:
            # ray attributes for PBR with detached shape
            nr_attrs_chunk = {
                'opacity': nr_attrs['opacity'][chunk_idx].detach(),
                'depth': nr_attrs['depth'][chunk_idx].detach(),
                'normal': nr_attrs['normal'][chunk_idx].detach(),
            }
            # add material
            for k in scene.brdf.material_dims.keys():
                nr_attrs_chunk[k] = nr_attrs[k][chunk_idx]
            if have_far:
                pbr_attrs_far_chunk = physically_based_render(
                    scene,
                    'far',
                    rays[chunk_idx],
                    nr_attrs_chunk,
                    far_lights=far_lights_pbr,
                    far_light_idx=far_light_idx,
                )
                for key, value in pbr_attrs_far_chunk.items():
                    if key not in pbr_attrs:
                        pbr_attrs[key] = [value]
                    else:
                        pbr_attrs[key].append(value)
            if have_near:
                near_lights_pbr.current_cam_pos = rays_o[chunk_idx]
                scene.near_lights_nr.current_cam_pos = rays_o[chunk_idx]
                pbr_attrs_near_chunk = physically_based_render(
                    scene,
                    'near',
                    rays[chunk_idx],
                    nr_attrs_chunk,
                    near_lights=near_lights_pbr,
                    near_light_state=near_light_state,
                )
                for key, value in pbr_attrs_near_chunk.items():
                    if key not in pbr_attrs:
                        pbr_attrs[key] = [value]
                    else:
                        pbr_attrs[key].append(value)
        
        # concatenate the chunks
        for key, value in pbr_attrs.items():
            pbr_attrs[key] = torch.cat(value, dim=0)
        all_attrs.update(pbr_attrs)
        del pbr_attrs
    
    for key in all_attrs.keys():
        assert key in global_vars.all_attr_names, \
            f'Unknown attribute name: {key}'
    
    # combine RGB from tensorial rendering and PBR
    # attributes: nr_rgb, pbr_rgb
    if do_nr_rgb:
        all_attrs['nr_rgb'] = torch.zeros(n_rays, 3, device=device)
    if do_pbr:
        all_attrs['pbr_rgb_spec'] = torch.zeros(n_rays, 3, device=device)
        all_attrs['pbr_rgb_diff'] = torch.zeros(n_rays, 3, device=device)
    if have_far:
        if do_nr_rgb:
            all_attrs['nr_rgb'] += all_attrs['nr_rgb_far']
        if do_pbr:
            all_attrs['pbr_rgb_far'] = (
                    all_attrs['pbr_rgb_spec_far']
                    + all_attrs['pbr_rgb_diff_far'])
            all_attrs['pbr_rgb_spec'] += all_attrs['pbr_rgb_spec_far']
            all_attrs['pbr_rgb_diff'] += all_attrs['pbr_rgb_diff_far']
    if have_near:
        if do_nr_rgb:
            all_attrs['nr_rgb'] += all_attrs['nr_rgb_near']
        if do_pbr:
            all_attrs['pbr_rgb_near'] = (
                    all_attrs['pbr_rgb_spec_near']
                    + all_attrs['pbr_rgb_diff_near'])
            all_attrs['pbr_rgb_spec'] += all_attrs['pbr_rgb_spec_near']
            all_attrs['pbr_rgb_diff'] += all_attrs['pbr_rgb_diff_near']
    if do_pbr:
        all_attrs['pbr_rgb'] = (all_attrs['pbr_rgb_spec']
                                + all_attrs['pbr_rgb_diff'])
    
    # perform color space conversion, tone mapping, and background replacement
    fg_weight = all_attrs['opacity']
    bg_weight = 1 - fg_weight
    default_normal = (torch.tensor([0, 0, 0], device=device)
                      .float().view(-1, 3))
    
    for key, value in all_attrs.items():
        # I don't know why there are nan values, but it happens, so we need to
        # fix it
        value = torch.nan_to_num(value)
        
        # if 'shading' in key:
        if key in [
            'direct_shading_far',
            'direct_shading_near',
        ]:
            value = value / value.mean().clamp(1e-6) * 0.3
        
        # if 'rgb' in key or 'albedo' in key or 'shading' in key:
        if key in [
            'nr_rgb_far',
            'nr_rgb_near',
            'base_color',
            'pbr_rgb_diff_far',
            'pbr_rgb_spec_far',
            'direct_shading_far',
            'indirect_rgb_far',
            'pbr_rgb_diff_near',
            'pbr_rgb_spec_near',
            'direct_shading_near',
            'indirect_rgb_near',
            'nr_rgb',  # = ray_nr_rgb_far + ray_nr_rgb_near
            'pbr_rgb_far',  # = ray_pbr_rgb_diff_far + ray_pbr_rgb_spec_far
            'pbr_rgb_near',  # = ray_pbr_rgb_diff_near + ray_pbr_rgb_spec_near
            'pbr_rgb_spec',  # = ray_pbr_rgb_spec_far + ray_pbr_rgb_spec_near
            'pbr_rgb_diff',  # = ray_pbr_rgb_diff_far + ray_pbr_rgb_diff_near
            'pbr_rgb',  # = ray_pbr_rgb_spec + ray_pbr_rgb_diff
        ]:
            # TODO: remove clamp(0, 1)?
            value = utils.linrgb2srgb(
                (value / fg_weight.clamp(1e-6)).clamp(0, 1)
            ) * fg_weight
        
        if key in ['normal', 'shading_normal']:
            value = value + bg_weight.detach() * default_normal * white_bg
        elif key in [
            'pbr_rgb', 'pbr_rgb_far', 'pbr_rgb_near',
            'indirect_rgb_far', 'indirect_rgb_near',
        ]:
            value = value + bg_weight.detach() * white_bg
        # elif key not in [
        #     'opacity', 'normal_orient_mse', 'albedo', 'roughness',
        #     'fresnel', 'visibility_far', 'visibility_near',
        #     'direct_shading_far', 'direct_shading_near',
        # ]:
        elif key in [
            'nr_rgb',
            'nr_rgb_far',
            'nr_rgb_near',
        ]:
            value = value + bg_weight * white_bg
        
        all_attrs[key] = value
    
    del nr_attrs
    
    return all_attrs


# @dr.wrap_ad(source='torch', target='drjit')
def physically_based_render(
        scene: scenes.SceneModel,
        which_light: str,
        rays: torch.Tensor,
        nr_ray_attrs: Dict[str, torch.Tensor],
        far_lights: lighting.FarLight = None,
        far_light_idx: int = None,
        near_lights: lighting.NearLight = None,
        near_light_state: torch.Tensor = None,
):
    """Perform physically-based rendering.

    Perform physically-based rendering for the input :attr:`rays` under far
    or near lights, in a fully separated manner.
    As we have modeled the infinity-bounced radiance using tensor field, we can
    directly use it for indirect lighting, so we only need to evaluate
    rendering equation on the primary intersections of camera ray and surface.

    Args:
        scene: the scene in which the rays are rendered
        which_light: the type of light to be rendered. Can be 'far' or 'near'
        rays: (n_rays, 6), the rays to be rendered
        nr_ray_attrs: the neural rendered attributes, including following keys:
            - opacity (torch.Tensor): (n_rays, 1) the accumulated weight of the
                ray
            - depth (torch.Tensor): (n_rays, 1) the depth of the ray
            - normal (torch.Tensor): (n_rays, 3) the surface normal of the ray
            - albedo (torch.Tensor): (n_rays, 3) the albedo of the ray
            - roughness (torch.Tensor): (n_rays, 1) the roughness of the ray
            - fresnel (torch.Tensor): (n_rays, 1) the fresnel of the ray
        far_lights: the far light source under which the rays are rendered.
        far_light_idx: the index of the far light. < 0 means no far light.
            Only used when which_light is 'far'.
        near_lights: the near light sources under which the rays are rendered.
        near_light_state: [n_near_lights] the open state of the near lights.
            Should be floats in [0, 1]. Only used when which_light is 'near'.

    Returns:
        Dict[str, torch.Tensor]: the rendered attributes for input rays under
        near or far lights. May include following keys:

            - ray_pbr_rgb_diff_{which_light} (torch.Tensor): (n_rays, 3), the
                diffuse part of PBR RGB color under far or near lights
            - ray_pbr_rgb_spec_{which_light} (torch.Tensor): (n_rays, 3), the
                specular part of PBR RGB color under far or near lights
            - ray_visibility_{which_light} (torch.Tensor): (n_rays, 3), the
                visibility under far or near lights
            - ray_direct_shading_{which_light} (torch.Tensor): (n_rays, 3), the
                direct RGB color under far or near lights
            - ray_indirect_rgb_{which_light} (torch.Tensor): (n_rays, 3), the
                indirect RGB color under far or near lights
    """
    assert which_light in ['far', 'near'], \
        f'Unknown source of scene lighting: {which_light}'
    
    if which_light == 'far':
        assert far_lights is not None and far_lights.n_lights > 0, \
            f'No far light source!'
    
    # read global variables
    is_train = global_vars.training
    args = global_vars.args
    it = global_vars.cur_iter
    device = scene.device
    
    # initialize the output tensors
    n_rays_all = rays.shape[0]
    ray_pbr_rgb_diff = torch.zeros(n_rays_all, 3, device=device)
    ray_pbr_rgb_spec = torch.zeros(n_rays_all, 3, device=device)
    ray_direct_shading = torch.zeros(n_rays_all, 3, device=device)
    ray_direct_shading_shadowless = torch.zeros(n_rays_all, 3, device=device)
    ray_visibility = torch.zeros(n_rays_all, 3, device=device)
    ray_indirect_rgb = torch.zeros(n_rays_all, 3, device=device)
    
    # check if no light source
    if ((which_light == 'far' and far_light_idx < 0) or
            (which_light == 'near' and sum(near_light_state) < 1e-6)):
        return {
            f'pbr_rgb_diff_{which_light}': ray_pbr_rgb_diff,
            f'pbr_rgb_spec_{which_light}': ray_pbr_rgb_spec,
            f'visibility_{which_light}': ray_visibility,
            f'direct_shading_{which_light}': ray_direct_shading,
            f'indirect_rgb_{which_light}': ray_indirect_rgb,
        }
    
    # TODO: this is an expedient solution for indirect lighting under novel
    #  light conditions. We should find a better solution.
    consider_sec_vis = it >= args.enable_sec_vis_for_pbr_after
    # if we are under novel light conditions, we can not use indirect lighting
    # from modeled radiance field.
    # pbr_consider_indirect = (it >= args.enable_indirect_after and
    #                          not is_under_novel_light)
    use_prefilter_for_diffuse = (
            args.use_prefiltered_envmap_for_diffuse and
            which_light == 'far' and far_lights.light_type == 'pixel')
    consider_indirect = it >= args.enable_indirect_after
    
    if far_lights is not None:
        far_light_param = far_lights.get_actual_params(far_light_idx)
    else:
        far_light_param = None
    
    opacity_mask = (nr_ray_attrs['opacity'] > 0.5).view(-1)
    depth = nr_ray_attrs['depth']
    normal = nr_ray_attrs['normal']
    
    if opacity_mask.any():
        # get only valid rays
        rays = rays[opacity_mask]
        depth = depth[opacity_mask].view(-1, 1)
        normal = normal[opacity_mask]
        materials = {}
        for k in scene.brdf.material_dims.keys():
            materials[k] = (nr_ray_attrs[k].view(n_rays_all, -1)
                            )[opacity_mask]
        
        # Trick: if the scene has flashlight, only use the PBR result under the
        # flashlight to update albedo.
        if (which_light == 'far' and
                'collocated' in scene.near_light_pos_types):
            materials['base_color'] = materials['base_color'].detach()
        # # If the scene has far light, do not use PBR under near light to update
        # # roughness and specular.
        # if (which_light == 'near' and scene.n_far_lights > 0
        #         and 'fixed' not in scene.near_light_pos_types):
        #     materials['roughness'] = materials['roughness'].detach()
        #     materials['specular'] = materials['specular'].detach()
        
        roughness = materials['roughness']
        
        # get points on the surface using depth prediction
        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]
        surf_xyz = (rays_o + rays_d * depth)  # (n_rays, 3)
        surf_xyz -= rays_d * 5e-2  # avoid self-intersection
        surf_xyz = surf_xyz.detach()
        
        n_rays = rays.shape[0]
        
        # Determine the number of secondary sample directions
        if is_train:
            n_sec_sample_dirs = args.n_sec_sample_dirs_train
        else:
            n_sec_sample_dirs = args.n_sec_sample_dirs_test
        
        if args.incident_sampling_method == 'importance':
            # we need to cosine importance sample the indirect incident light
            # directions, as it can not be sampled well using light importance
            # sampling or brdf (specular) importance sampling.
            # As direct lights can be occluded, we also need to sample
            # cosine directions for direct lighting that may come from
            # unoccluded light sources.
            remaining_dirs = n_sec_sample_dirs
            if consider_sec_vis:
                n_cos_dirs = int(n_sec_sample_dirs * 0.3)
                remaining_dirs -= n_cos_dirs
            else:
                n_cos_dirs = 0
            if which_light == 'far':
                n_brdf_dirs = int(remaining_dirs * 0.5)
                n_light_dirs = int(remaining_dirs * 0.5)
            else:  # near
                n_brdf_dirs = remaining_dirs
                # do not importance sample near light directions
                n_light_dirs = 0
        elif args.incident_sampling_method == 'stratified':
            if which_light == 'far':
                n_light_dirs = n_sec_sample_dirs
                n_brdf_dirs = 0
                n_cos_dirs = 0
            else:
                n_light_dirs = 0
                n_brdf_dirs = 0
                n_cos_dirs = 0
        else:
            raise NotImplementedError(
                f'Unknown incident sampling method: '
                f'{args.incident_sampling_method}'
            )
        
        wi_sampled = {}
        sampler = ray_sampler.RaySampler(normal, device=device)
        
        # sample incident directions. do not compute gradient of sampling
        with torch.no_grad():
            # sample light source
            # (n_rays, n_light_dirs, 3 or 1)
            if n_light_dirs > 0:  # only be true when which_light == 'far'
                if far_lights.light_type == 'sg':
                    # importance sample incident light directions for light
                    wi_light, pdf_light, pdf_fn_light = \
                        sampler.importance_sample_sg_light(
                            n_light_dirs,
                            far_light_param
                        )
                elif far_lights.light_type == 'pixel':
                    wi_light, pdf_light, pdf_fn_light = \
                        sampler.importance_sample_pixel_light(
                            n_light_dirs,
                            far_light_param
                        )
                wi_sampled['light'] = {
                    'wi': wi_light,
                    'pdf': pdf_light.clamp(min=1e-8),
                    'pdf_fn': pdf_fn_light,
                    'n_dirs': n_light_dirs,
                }
                del wi_light, pdf_light, pdf_fn_light
            
            if n_brdf_dirs > 0:
                # importance sample incident light directions for brdf
                wi_brdf, pdf_brdf, pdf_fn_brdf = (
                    sampler.importance_sample_ggx_brdf(
                        n_brdf_dirs,
                        roughness,
                        -rays_d
                    ))
                wi_sampled['brdf'] = {
                    'wi': wi_brdf,
                    'pdf': pdf_brdf.clamp(min=1e-8),
                    'pdf_fn': pdf_fn_brdf,
                    'n_dirs': n_brdf_dirs,
                }
                del wi_brdf, pdf_brdf, pdf_fn_brdf
            
            if n_cos_dirs > 0:
                # cosine importance sample incident light directions for
                # indirect lighting
                wi_cos, pdf_cos, pdf_fn_cos = sampler.cosine_sample(n_cos_dirs)
                wi_sampled['cos'] = {
                    'wi': wi_cos,
                    'pdf': pdf_cos.clamp(min=1e-8),
                    'pdf_fn': pdf_fn_cos,
                    'n_dirs': n_cos_dirs,
                }
                del wi_cos, pdf_cos, pdf_fn_cos
            
            # prepare for multiple importance sampling
            for sample_type, sampled_dict in wi_sampled.items():
                # compute the pdf of wi in other sample types
                pdf_array = {}
                for other_sample_type, other_sampled_dict in wi_sampled.items():
                    if other_sample_type == sample_type:
                        pdf_array[other_sample_type] = sampled_dict['pdf']
                    else:
                        pdf_fn = other_sampled_dict['pdf_fn']
                        pdf_array[other_sample_type] = pdf_fn(
                            sampled_dict['wi'],
                            normal,
                            -rays_d,
                            roughness,
                            far_light_param
                        )
                sampled_dict['pdf_array'] = pdf_array
                wi_sampled[sample_type] = sampled_dict
        
        # pre-compute reflectance, geometry term, and visibility for each
        # sample type
        for sample_type, sampled_dict in wi_sampled.items():
            # compute BRDF and geometry term
            diff_part, spec_part, cos_term, cos_mask = (
                scene.brdf.compute_reflectance(
                    normal,
                    sampled_dict['wi'],
                    -rays_d,
                    materials
                ))
            
            sampled_dict['diff_part'] = diff_part
            sampled_dict['spec_part'] = spec_part
            sampled_dict['cos_term'] = cos_term
            sampled_dict['cos_mask'] = cos_mask
            
            # query far light source
            if which_light == 'far':
                # query direct far lighting
                sampled_dict['direct_incident'] = (
                    far_lights.query_at_directions(
                        far_light_idx, sampled_dict['wi']
                    ))
            
            # compute visibility in test mode
            if consider_sec_vis:
                sec_intersect_xyz, visibility = cast_secondary_ray(
                    scene=scene,
                    surf_xyz=surf_xyz,
                    wi=sampled_dict['wi'],
                    cos_mask=cos_mask
                )
                sampled_dict['sec_intersect_xyz'] = sec_intersect_xyz
                sampled_dict['visibility'] = visibility
                # query indirect incident
                if consider_indirect:
                    if which_light == 'near':
                        # (n_rays, 3)
                        flash_o = scene.near_lights_nr.current_cam_pos
                        scene.near_lights_nr.current_cam_pos = (
                            flash_o[opacity_mask].view(-1, 1, 3).expand_as(
                                sec_intersect_xyz
                            ))
                    indirect_incident = compute_indirect_incident_radiance(
                        scene=scene,
                        which_light=which_light,
                        wi=sampled_dict['wi'],
                        sec_intersect_xyz=sec_intersect_xyz,
                        opacity=1 - visibility,
                        cos_mask=sampled_dict['cos_mask'],
                        far_light_idx=far_light_idx,
                        near_light_state=near_light_state
                    )
                    sampled_dict['indirect_incident'] = indirect_incident
                    if which_light == 'near':
                        scene.near_lights_nr.current_cam_pos = flash_o
                        del flash_o
            
            wi_sampled[sample_type] = sampled_dict
        
        # compute direct illumination from light sources
        if which_light == 'far':
            # Compute diffuse direct.
            # If not using prefiltered environment map, we use MC integration
            # to compute the diffuse component.
            # If using prefiltered environment map, we also need MC integration
            # to compute average visibility for each point.
            used_sample_types_diff_direct = ['light']
            if consider_sec_vis:
                used_sample_types_diff_direct.append('cos')
            for sample_type in used_sample_types_diff_direct:
                sampled_dict = wi_sampled[sample_type]
                
                direct_incident = sampled_dict['direct_incident']
                cos_term = sampled_dict['cos_term']
                
                if consider_sec_vis:
                    all_incident = direct_incident * sampled_dict['visibility']
                else:
                    all_incident = direct_incident
                
                diff_reflectance = sampled_dict['diff_part']
                out_rgb_diff = all_incident * diff_reflectance * cos_term
                direct_shading = all_incident * cos_term
                direct_shading_shadowless = direct_incident * cos_term
                
                # multiple importance sampling weights
                weight = ray_sampler.power_heuristic_list(
                    used_types=used_sample_types_diff_direct,
                    this_type=sample_type,
                    pdf_dict=sampled_dict['pdf_array']
                )
                
                # compute the rendered image
                # pdf ratio to uniform sampling
                pdf = sampled_dict['pdf'] / (1 / (2 * np.pi))
                ray_direct_shading[opacity_mask] += torch.mean(
                    direct_shading / pdf * weight, dim=1
                )
                ray_direct_shading_shadowless[opacity_mask] += torch.mean(
                    direct_shading_shadowless / pdf * weight, dim=1
                )
                if not use_prefilter_for_diffuse:
                    ray_pbr_rgb_diff[opacity_mask] += torch.mean(
                        out_rgb_diff / pdf * weight, dim=1
                    )
            
            ray_visibility[opacity_mask] = \
                (ray_direct_shading[opacity_mask].clamp(1e-4) /
                 ray_direct_shading_shadowless[opacity_mask].clamp(1e-4))
            
            # if using prefiltered environment map, we use precomputed
            # irradiance and average visibility to compute the diffuse
            # component
            if use_prefilter_for_diffuse:
                far_lights: lighting.PixelFarLight
                irradiance = far_lights.query_at_directions(
                    far_light_idx,
                    normal,
                    prefiltered=True
                )
                irradiance = irradiance * ray_visibility[opacity_mask]
                # if normal faces backward view direction, set irradiance to 0
                normal_view_dot = torch.sum(normal * -rays_d, dim=1)
                irradiance = (irradiance *
                              ((F.softsign(normal_view_dot * 100) + 1) / 2
                               ).view(-1, 1))
                ray_pbr_rgb_diff[opacity_mask] = (
                        irradiance * materials['base_color'] / np.pi)
            
            used_sample_types_spec_direct = ['light', 'brdf']
            if consider_sec_vis:
                used_sample_types_spec_direct.append('cos')
            # sum up the contribution of each type of sample
            for sample_type in used_sample_types_spec_direct:
                sampled_dict = wi_sampled[sample_type]
                
                direct_incident = sampled_dict['direct_incident']
                cos_term = sampled_dict['cos_term']
                
                if consider_sec_vis:
                    all_incident = direct_incident * sampled_dict['visibility']
                else:
                    all_incident = direct_incident
                
                out_rgb_spec = (all_incident * sampled_dict['spec_part']
                                * cos_term)
                
                # multiple importance sampling weights
                weight = ray_sampler.power_heuristic_list(
                    used_types=used_sample_types_spec_direct,
                    this_type=sample_type,
                    pdf_dict=sampled_dict['pdf_array']
                )
                
                # compute the rendered image
                pdf = sampled_dict['pdf'] / (1 / (2 * np.pi))
                ray_pbr_rgb_spec[opacity_mask] += torch.mean(
                    out_rgb_spec / pdf * weight, dim=1
                )
        
        else:  # near
            # compute the contribution of near light
            # do not importance sample near light directions. instead we
            # sample all near light directions
            flash_o_all = near_lights.current_cam_pos
            near_lights.current_cam_pos = flash_o_all[opacity_mask]
            wi, dist, intensity = (
                near_lights.get_intensity_and_relative_dir_dist(surf_xyz))
            
            near_light_indices = [
                id for id, state in enumerate(near_light_state)
                if state == 1]
            for i, near_id in enumerate(near_light_indices):
                # compute BRDF and geometry term
                # (n_pts, tensorf.n_near_lights, None or 1 or 3)
                i_am_flash = (near_lights.pos_types[near_id] == 'collocated')
                have_flash = ('collocated' in scene.near_light_pos_types)
                materials_proc = materials
                if (not i_am_flash) and have_flash:
                    materials_proc['base_color'] = materials_proc['base_color'].detach()
                diff_part, spec_part, cos_term, cos_mask = (
                    scene.brdf.compute_reflectance(
                        normal,
                        wi[:, near_id].view(-1, 1, 3),
                        -rays_d,
                        materials_proc
                    ))
                diff_part = diff_part.view(-1, 3)
                spec_part = spec_part.view(-1, 3)
                cos_term = cos_term.view(-1, 1)
                
                # compute direct near lighting by consider intensity, distance,
                # and open state
                direct_incident = (
                        intensity[:, near_id] / (dist[:, near_id] ** 2) *
                        near_light_state[near_id])
                
                if consider_sec_vis:
                    # as we do not compute indirect lighting here (it is
                    # computed using importance sampling), we do not need
                    # secondary intersection
                    
                    # for collocated near light, the visibility is just one
                    if near_lights.light_type == 'collocated':
                        visibility = torch.ones(
                            direct_incident.shape[0],
                            device=device
                        )
                    else:
                        _, visibility = cast_secondary_ray(
                            scene=scene,
                            surf_xyz=surf_xyz,
                            wi=wi[:, near_id].view(-1, 1, 3),
                            cos_mask=cos_mask, )
                    visibility = visibility.view(-1, 1)
                    all_incident = direct_incident * visibility
                else:
                    all_incident = direct_incident
                
                out_rgb_spec = all_incident * spec_part * cos_term
                out_rgb_diff = all_incident * diff_part * cos_term
                direct_shading = all_incident * cos_term
                direct_shading_shadowless = direct_incident * cos_term
                
                # compute contribution from light sampling
                ray_pbr_rgb_spec[opacity_mask] += out_rgb_spec
                ray_pbr_rgb_diff[opacity_mask] += out_rgb_diff
                ray_direct_shading[opacity_mask] += direct_shading
                ray_direct_shading_shadowless[opacity_mask] += \
                    direct_shading_shadowless
            
            ray_visibility[opacity_mask] = \
                (ray_direct_shading[opacity_mask].clamp(1e-4) /
                 ray_direct_shading_shadowless[opacity_mask].clamp(1e-4))
            # restore the camera position
            near_lights.current_cam_pos = flash_o_all
        
        # compute indirect illumination
        if consider_indirect:
            # compute indirect diffuse
            used_sample_types_indirect_diff = ['cos']
            for sample_type in used_sample_types_indirect_diff:
                sampled_dict = wi_sampled[sample_type]
                
                indirect_incident = sampled_dict['indirect_incident']
                diff_reflectance = sampled_dict['diff_part']
                cos_term = sampled_dict['cos_term']
                
                indirect_diff = indirect_incident * diff_reflectance * cos_term
                
                weight = ray_sampler.power_heuristic_list(
                    used_types=used_sample_types_indirect_diff,
                    this_type=sample_type,
                    pdf_dict=sampled_dict['pdf_array']
                )
                
                # ratio to uniform (= integral)
                pdf = sampled_dict['pdf'] / (1 / (2 * np.pi))
                indirect_diff = torch.mean(
                    indirect_diff / pdf * weight, dim=1
                )
                ray_pbr_rgb_diff[opacity_mask] += indirect_diff
                ray_indirect_rgb[opacity_mask] += indirect_diff
            
            # compute indirect specular
            used_sample_types_indirect_spec = ['cos', 'brdf']
            for sample_type in used_sample_types_indirect_spec:
                sampled_dict = wi_sampled[sample_type]
                
                indirect_incident = sampled_dict['indirect_incident']
                spec_part = sampled_dict['spec_part']
                cos_term = sampled_dict['cos_term']
                
                indirect_spec = indirect_incident * spec_part * cos_term
                
                weight = ray_sampler.power_heuristic_list(
                    used_types=used_sample_types_indirect_spec,
                    this_type=sample_type,
                    pdf_dict=sampled_dict['pdf_array']
                )
                
                # ratio to uniform (= integral)
                pdf = sampled_dict['pdf'] / (1 / (2 * np.pi))
                indirect_spec = torch.mean(
                    indirect_spec / pdf * weight, dim=1
                )
                ray_pbr_rgb_spec[opacity_mask] += indirect_spec
                ray_indirect_rgb[opacity_mask] += indirect_spec
            
            del indirect_incident, indirect_diff, weight
        
        del surf_xyz, rays_o, rays_d, wi_sampled, sampler, \
            pdf_array, spec_part, cos_term, cos_mask, \
            direct_incident, all_incident, \
            out_rgb_spec, direct_shading, direct_shading_shadowless
    
    del opacity_mask, depth, normal
    
    return {
        f'pbr_rgb_diff_{which_light}': ray_pbr_rgb_diff,
        f'pbr_rgb_spec_{which_light}': ray_pbr_rgb_spec,
        f'visibility_{which_light}': ray_visibility,
        f'direct_shading_{which_light}': ray_direct_shading,
        f'indirect_rgb_{which_light}': ray_indirect_rgb,
    }


@torch.no_grad()
def cast_secondary_ray(
        scene: scenes.SceneModel,
        surf_xyz: torch.Tensor,
        wi: torch.Tensor,
        cos_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cast secondary rays and return the intersection points and visibility.

    Args:
        scene: the scene model
        surf_xyz: (n_rays, 3), the surface points of the rays
        wi: (n_rays, n_dirs, 3), the incident directions of the rays
        cos_mask: (n_rays, n_dirs), the mask of the rays

    Returns:
        A tuple of two tensors:

        - sec_intersect_xyz (torch.Tensor): (n_rays, n_dirs, 3], the
            intersection points of the secondary rays
        - visibility (torch.Tensor): (n_rays, n_dirs, 1], the secondary
            visibility
    """
    # is_train = global_vars.training
    args = global_vars.args
    assert args.detach_secondary, \
        'We only support detached secondary rays for now.'
    
    scene_training = scene.training
    scene.eval()
    
    device = scene.device
    n_rays, n_dirs = cos_mask.shape
    
    surf_xyz = surf_xyz.view(n_rays, 1, 3).expand(-1, n_dirs, -1)
    
    # initialize results
    visibility = torch.ones(n_rays, n_dirs, 1, device=device)
    sec_intersect_xyz = torch.zeros(n_rays, n_dirs, 3, device=device)
    
    # chunk_size = args.chunk_size_nr_train if is_train \
    #     else args.chunk_size_nr_test
    chunk_size = args.chunk_size_nr_test
    
    if cos_mask.any():
        chunk_idxs = torch.stack(torch.where(cos_mask), dim=1)
        chunk_idxs = torch.split(chunk_idxs, chunk_size, dim=0)
        chunk_idxs = [chunk_idx.unbind(dim=-1) for chunk_idx in chunk_idxs]
    else:
        chunk_idxs = []
    
    with torch.set_grad_enabled(not args.detach_secondary and scene.training):
        for chunk_idx in chunk_idxs:
            surf_xyz_chunk = surf_xyz[chunk_idx]
            wi_chunk = wi[chunk_idx]
            # Cast secondary rays through the tensor field
            nr_sec_ray_attrs = scene(
                rays_o=surf_xyz_chunk,
                rays_d=wi_chunk,
                return_rgb_far=False,
                return_rgb_near=False,
                return_material=False,
                n_segs=args.max_second_ray_segments,
                # TODO: change far plane to distance to light for each point
                near_far=args.second_near_far,
            )
            visibility[chunk_idx] = 1 - nr_sec_ray_attrs['opacity']
            sec_intersect_xyz[chunk_idx] = (
                    surf_xyz_chunk +
                    wi_chunk * nr_sec_ray_attrs['depth'])
    
    if args.detach_secondary:
        visibility = visibility.detach()
        sec_intersect_xyz = sec_intersect_xyz.detach()
    
    if scene_training:
        scene.train()
    
    return sec_intersect_xyz, visibility


@torch.no_grad()
def compute_indirect_incident_radiance(
        scene: scenes.SceneModel,
        which_light: str,
        wi: torch.Tensor,
        sec_intersect_xyz: torch.Tensor,
        opacity: torch.Tensor,
        cos_mask: torch.Tensor,
        far_light_idx: int = None,
        near_light_state: torch.Tensor = None,
) -> torch.Tensor:
    """Compute indirect lighting treating the secondary ray as from a hard
    surface.

    Notes:
        If which_light is 'near', you should set
        scene.near_lights_nr.current_cam_pos to the camera position (whose shape
        should be the same as :attr:`sec_intersect_xyz`) before calling this
        function.

    Args:
        scene: the scene in which the rays are rendered
        which_light: the type of light to be rendered. Can be 'far' or 'near'
        wi: (n_rays, n_dirs, 3), the incident directions of the rays
        sec_intersect_xyz: (n_rays, n_dirs, 3), the intersection points of
            the secondary rays
        opacity: (n_rays, n_dirs, 1), the accumulated weight of the rays
        cos_mask: (n_rays, n_dirs), the mask of the rays
        far_light_idx: the index of the far light. < 0 means no far light.
            Only used when which_light is 'far' and consider_indirect is True.
        near_light_state: (n_near_lights, ) the open state of the near lights.
            Should be floats in [0, 1]. Only used when which_light is 'near'
            and consider_indirect is True.

    Returns:
        torch.Tensor: (n_rays, n_dirs, 3), the indirect incident radiance
    """
    # is_train = global_vars.training
    args = global_vars.args
    secondary_as_surface = args.secondary_as_surface
    detach_secondary = args.detach_secondary
    
    if not secondary_as_surface:
        raise NotImplementedError(
            'I recommend to use secondary_as_surface=True for now.'
        )
    if not detach_secondary:
        raise NotImplementedError(
            'I recommend to use detach_secondary=True for now.'
        )
    
    device = scene.device
    n_rays, n_dirs = cos_mask.shape
    
    # initialize the output tensors
    indirect_incident = torch.zeros(n_rays, n_dirs, 3, device=device)
    
    # chunk_size = args.chunk_size_nr_train if is_train \
    #     else args.chunk_size_nr_test
    chunk_size = args.chunk_size_nr_test
    
    # get points inside the bounding box
    in_bbox_mask = ((scene.aabb[0] <= sec_intersect_xyz)
                    & (sec_intersect_xyz <= scene.aabb[1])).all(dim=-1)
    cos_mask = cos_mask & in_bbox_mask
    
    if cos_mask.any():
        chunk_idxs = torch.stack(torch.where(cos_mask), dim=1)
        chunk_idxs = torch.split(chunk_idxs, chunk_size, dim=0)
        chunk_idxs = [chunk_idx.unbind(dim=-1) for chunk_idx in chunk_idxs]
    else:
        chunk_idxs = []
    
    # if flash_o is not None:
    #     flash_o = flash_o.view(n_rays, 1, 3).expand(-1, n_dirs, -1)
    if which_light == 'near':
        flash_o_all = scene.near_lights_nr.current_cam_pos  # (n_ray, n_dirs, 3)
    
    for chunk_idx in chunk_idxs:
        wi_chunk = wi[chunk_idx]
        sec_intersect_xyz_chunk = sec_intersect_xyz[chunk_idx]
        opacity_chunk = opacity[chunk_idx]
        
        if which_light == 'far':
            nr_sec_ray_attrs = scene.field.query_field(
                sec_intersect_xyz_chunk,
                view_dirs_out=-wi_chunk,
                far_light_indices=[far_light_idx],
            )
            nr_sec_ray_rgb = nr_sec_ray_attrs[
                f'far_radiance_{far_light_idx}']
        else:  # near
            # query the near light radiance field
            scene.near_lights_nr.current_cam_pos = flash_o_all[chunk_idx]
            near_light_indices = [id for id, state in
                                  enumerate(near_light_state)
                                  if state == 1]
            nr_sec_ray_attrs = scene.field.query_field(
                sec_intersect_xyz_chunk,
                view_dirs_out=-wi_chunk,
                near_light_indices=near_light_indices,
                near_lights=scene.near_lights_nr,
            )
            nr_sec_ray_rgb = torch.zeros_like(sec_intersect_xyz_chunk)
            for i, near_id in enumerate(near_light_indices):
                nr_sec_ray_rgb += nr_sec_ray_attrs[f'near_radiance_{near_id}']
        
        indirect_incident[chunk_idx] = nr_sec_ray_rgb * opacity_chunk
    
    # restore the camera position
    if which_light == 'near':
        scene.near_lights_nr.current_cam_pos = flash_o_all
    
    indirect_incident = indirect_incident.detach()
    
    return indirect_incident
