"""
This file implements the scene model, which contains the neural field, the lighting,
and the alpha mask.
"""
from typing import *
from typing import Tuple

import configargparse
import nerfacc
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from internal import brdf, global_vars, instant_nsr, lighting, utils


class SceneModel(nn.Module):
    """The scene model that contains the common functions for rendering a ray,
    querying scene attributes at given positions, and stores the scene
    components including the lights, the neural field model, and the alpha
    mask.
    """
    
    def __init__(
            self,
            field_type: str,
            args: configargparse.Namespace,
            device: torch.device,
            n_far_lights: int,
            n_near_lights: int,
            near_light_pos_types: List[str],
            aabb: torch.Tensor = None,
            init_occ_grid: bool = True,
            **kwargs
    ):
        """Initialize the base model that contains the common functions for
        rendering a ray, querying scene attributes at given positions.

        Args:
            field_type: the used base tensorial or neural field model.
                Can be 'TensoRF' or 'Instant-NSR'.
            args: the arguments of the model
            device: the device to run the model
            n_far_lights: the number of far light conditions
            n_near_lights: the number of near lights
            near_light_pos_types: the types of the near light positions. Can be
                'fixed' or 'collocated'
            aabb: the axis-aligned bounding box of the scene
            # alpha_mask: the alpha mask of the scene

            **kwargs
        """
        super().__init__()
        
        self.args = args
        self.device = device
        if aabb is None:
            aabb = torch.tensor(args.scene_aabb).view(2, 3)
        self.aabb = aabb.to(device)
        self.pbr_enabled = False
        
        if not args.use_occ_grid:
            raise NotImplementedError(
                'Not using occ grid is not supported '
                'yet.'
            )
        
        # get the bounding box of the scene
        aabb = torch.tensor(args.scene_aabb).view(2, 3).to(device)
        # build nerfacc occupancy grid
        self.occ_grid = nerfacc.OccGridEstimator(
            roi_aabb=aabb.reshape(-1),
            resolution=utils.total_num_to_resolution(
                128 ** 3, aabb
            ),
            levels=1
        )
        self.occ_grid.to(device)
        
        self.max_samples_per_ray = args.n_samples_per_ray
        self.alpha_mask_thres = args.alpha_mask_thres
        self.ray_march_weight_thres = args.ray_march_weight_thres
        self.near_far = args.primary_near_far
        
        # initialize light
        # far light parameters
        self.n_far_lights = n_far_lights
        self.envmap_h, self.envmap_w = args.envmap_hw
        # near light parameters
        self.n_near_lights = n_near_lights
        self.near_light_pos_types = near_light_pos_types
        # initialize lights
        self.far_lights: Optional[lighting.FarLight] = None
        self.baked_far_lights: Optional[lighting.PixelFarLight] = None
        self.near_lights_nr: Optional[lighting.NearLight] = None
        self.near_lights_pbr: Optional[lighting.NearLight] = None
        self.init_light(args)
        
        # build field for geometry and material
        self.field_type = field_type
        self.brdf = brdf.get_brdf(args.brdf_type, args)
        self.material_dim = self.brdf.packed_material_dim
        if field_type == 'InstantNSR':
            self.field = instant_nsr.NeuSField(
                args,
                self.device,
                self.aabb,
                n_far_lights=self.n_far_lights,
                n_near_lights=self.n_near_lights,
                material_dim=self.material_dim,
            )
        else:
            raise ValueError(f'Unknown field type: {field_type}')
        
        # scene size
        self.aabb_size = None
        # self.inv_aabb_size = None
        self.step_length = None
        self.aabb_diag = None
        self.n_ray_segments = None
        # (self.aabb_size, self.inv_aabb_size, self.step_length, self.aabb_diag,
        #  self.n_ray_segments) = None, None, None, None, None
        self.update_size()
        
        # bake far light into environment map
        self.update_baked_envmap()
        
        if init_occ_grid:
            self.occ_grid._update(
                step=0,
                occ_eval_fn=self.field.closure_occ_eval_fn()
            )
        
        # print unused kwargs
        if len(kwargs) > 0:
            print('\nUnused kwargs for TensorBase:')
            for k, v in kwargs.items():
                print(f'\t{k}: {v}')
    
    def train(self, mode: bool = True):
        """Set the training mode for the scene model."""
        super().train(mode)
        if self.pbr_enabled and self.args.fix_shape_and_radiance_in_pbr:
            self.occ_grid.eval()
        return self
    
    def enable_pbr(self):
        """Enable PBR mode for the scene model."""
        if not self.pbr_enabled:
            # copy the nearlight and fix its intensity for neural rendering
            if self.near_lights_pbr is not None:
                self.near_lights_pbr.load_state_dict(
                    self.near_lights_nr.state_dict()
                )
                if self.args.fix_shape_and_radiance_in_pbr:
                    self.near_lights_nr.eval()
                    self.near_lights_nr.requires_grad_(False)
            # enable pbr for field
            self.field.enable_pbr()
            
            self.pbr_enabled = True
    
    def update_size(self):
        """
        Given the axis-aligned bounding box and the grid size, compute needed
        sizes for ray marching.

        Set the following attributes:
            - aabb_size (torch.Tensor): the size of the axis-aligned bounding
                box
            - inv_aabb_size (torch.Tensor): the inverse of the size of the
                axis-aligned bounding box
            - step_length (torch.Tensor): the length of each step
            - aabb_diag (torch.Tensor): the diagonal length of the
                axis-aligned bounding box
            - max_samples (int): the maximum number of samples

        Returns:
            None
        """
        self.aabb = self.aabb.to(self.device)
        self.aabb_size = self.aabb[1] - self.aabb[0]
        # self.inv_aabb_size = 1.0 / self.aabb_size
        self.aabb_diag = torch.norm(self.aabb_size, p=2)
        self.step_length = self.aabb_diag / self.max_samples_per_ray
        self.n_ray_segments = self.max_samples_per_ray
    
    def get_optim_param_groups(
            self,
    ) -> List[Dict[str, Union[torch.Tensor, str, float]]]:
        """Get the parameter groups for the optimizer.

        Returns:
            List[Dict[str, Union[torch.Tensor, float]]]: the list of parameter
                groups including parameters, name, and learning rates
        """
        param_groups = []
        args = self.args
        
        if self.pbr_enabled:
            if self.n_far_lights > 0:
                param_groups.append(
                    {'params': self.far_lights.get_optim_params(),
                     'name': 'far_lights',
                     'lr': args.lr_light
                     }
                )
            if self.n_near_lights > 0:
                param_groups.append(
                    {'params': self.near_lights_pbr.get_optim_params(),
                     'name': 'near_lights_pbr',
                     'lr': args.lr_light
                     }
                )
        
        if not (self.pbr_enabled and self.args.fix_shape_and_radiance_in_pbr):
            if self.n_near_lights > 0:
                param_groups.append(
                    {'params': self.near_lights_nr.get_optim_params(),
                     'name': 'near_lights_nr',
                     'lr': args.lr_light
                     }
                )
        
        param_groups.extend(self.field.get_optim_param_groups())
        
        return param_groups
    
    def init_light(
            self,
            args: configargparse.Namespace,
    ):
        """Initialize lighting parameters for far and near lights.

        The initialized parameters are stored in self.far_lights and
        self.near_lights, which are lists of nn.Parameter and whose length
        equals to self.n_far_lights and self.n_near_lights respectively.
        The specific content of the parameters depends on the light kind.

        Returns:
            None
        """
        # initialize far lights as spherical Gaussians
        device = self.device
        if self.n_far_lights > 0:
            self.far_lights = lighting.create_light(
                'far',
                args.far_light_type,
                self.n_far_lights,
                device,
                True,
                n_lobes=args.n_sg_lobes,
                envmap_h=args.envmap_hw[0],
                envmap_w=args.envmap_hw[1],
            )
            self.baked_far_lights = lighting.create_light(
                'far',
                'pixel',
                self.n_far_lights,
                device,
                False,
                envmap_h=args.envmap_hw[0],
                envmap_w=args.envmap_hw[1],
            )
        if self.n_near_lights > 0:
            self.near_lights_nr = lighting.create_light(
                'near',
                args.near_light_type,
                self.n_near_lights,
                device,
                True,
                pos_types=self.near_light_pos_types,
                sh_order=args.near_light_sh_order,
            )
            self.near_lights_pbr = lighting.create_light(
                'near',
                args.near_light_type,
                self.n_near_lights,
                device,
                True,
                pos_types=self.near_light_pos_types,
                sh_order=args.near_light_sh_order,
            )
    
    def update_baked_envmap(self):
        """Update the baked environment map for each far light.

        This function can be called when baked environment map is needed to be
        updated, e.g., after the far light parameters are updated and before
        testing.
        """
        if self.n_far_lights > 0:
            lighting.bake_to_envmap(
                self.far_lights,
                self.baked_far_lights,
                detach=True,
            )
    
    @torch.no_grad()
    def filter_rays(
            self,
            all_cond_rays: List[Dict[str, Any]],
            chunk_size: int = 2 ** 17,
            use_ray_opacity: bool = False,
            random_retention_ratio: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Filter the training rays in empty space.

        Filter out the rays that are outside the bounding box or have
        alpha values smaller than a threshold (after initial training).

        Args:
            all_cond_rays: the list of dictionaries containing the ray
                attributes for each image
            chunk_size: the chunk size to split the rays
            use_ray_opacity: whether to render the ray and use opacity to filter
                rays
            random_retention_ratio: the ratio of randomly retained rays, to
                suppress artifacts in the empty space

        Returns:
            Dict[str, torch.Tensor]: the filtered rays
        """
        print(f'\n======> Begin filtering rays ... ')
        print(f'Random retention ratio: {random_retention_ratio * 100:.2f}%')
        if not use_ray_opacity:
            print('Using aabb to filter rays.')
        else:
            print('Using opacity to filter rays. This may take a while.')
        
        self.eval()  # to accelerate the ray filtering
        
        # record time
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        
        # initialize
        total_n_rays = 0
        left_n_rays = 0
        all_cond_rays_filtered = []
        for ray_attrs in all_cond_rays:
            rays = ray_attrs['rays']
            n_rays = ray_attrs['n_rays']
            
            idx_chunks = torch.split(
                torch.arange(n_rays),
                chunk_size
            )
            
            filter_mask = []
            for idx_chunk in idx_chunks:
                rays_chunk = rays[idx_chunk].to(self.device)
                rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3: 6]
                
                if use_ray_opacity:
                    # render the ray and see opacity
                    nr_attrs_chunk = self._forward_test(
                        rays_o=rays_o,
                        rays_d=rays_d,
                        return_rgb_far=False,
                        return_rgb_near=False,
                        return_material=False,
                        n_segs=64,
                        near_far=self.near_far,
                    )
                    mask_opaque = nr_attrs_chunk['opacity'] > 0.1
                    filter_mask.append(mask_opaque)
                else:
                    # use aabb only
                    vec = torch.where(
                        rays_d == 0,
                        torch.full_like(rays_d, 1e-6),
                        rays_d
                    )
                    t_1 = (self.aabb[1] - rays_o) / vec
                    t_2 = (self.aabb[0] - rays_o) / vec
                    t_min = torch.minimum(t_1, t_2).amax(-1)
                    t_max = torch.maximum(t_1, t_2).amin(-1)
                    mask_in_bbox = torch.greater(t_max, t_min)
                    filter_mask.append(mask_in_bbox)
            
            filter_mask = torch.cat(filter_mask).view(-1)
            
            # randomly retain some rays
            filter_mask = torch.logical_or(
                filter_mask,
                torch.less(
                    torch.rand(filter_mask.shape, device=self.device),
                    random_retention_ratio
                )
            )
            
            # filter the rays
            ray_attrs_filtered = {}
            for key, value in ray_attrs.items():
                if isinstance(value, torch.Tensor):
                    if value.shape[0] == ray_attrs['n_rays']:
                        value = value[filter_mask]
                ray_attrs_filtered[key] = value
            ray_attrs_filtered['n_rays'] = torch.sum(filter_mask).item()
            total_n_rays += ray_attrs['n_rays']
            left_n_rays += ray_attrs_filtered['n_rays']
            all_cond_rays_filtered.append(ray_attrs_filtered)
        
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        print(
            f'Ray filtering done! Takes '
            f'{start.elapsed_time(end) / 1000:.2f} seconds.'
            f'\nRay retention ratio: '
            f'{left_n_rays / total_n_rays * 100:.2f}%'
        )
        print('======> End filtering rays.\n')
        
        return all_cond_rays_filtered
    
    def unpack_brdf(
            self,
            packed_material: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Unpack the brdf tensor to separate parameters.

        Args:
            packed_material: (n_pts, material_dim) the material vector output by
                the field model

        Returns:
            A dict of tensors, whose keys may include:

            - BRDF parameters specified in self.brdf.material_dims
        """
        ret_dict = {}
        brdf_dict = self.brdf.unpack_brdf(packed_material)
        ret_dict.update(brdf_dict)
        return ret_dict
    
    def forward(
            self,
            *args,
            **kwargs,
    ):
        """Tensorial volume rendering the given rays.

        If :attr:`occ_grid` is not None, use nerfacc to render the rays.
        Otherwise, use volume rendering without nerfacc.

        Args:
            *args: the arguments to pass to :meth:`_forward` or
                :meth:`_forward_nerfacc`
            **kwargs: the keyword arguments to pass to :meth:`_forward` or
                :meth:`_forward_nerfacc`

        Returns:
            See :meth:`_forward` or :meth:`_forward_nerfacc`
        """
        assert self.field_type == 'InstantNSR', \
            'Only InstantNSR is supported.'
        if self.occ_grid is None:
            raise NotImplementedError
        if self.training:
            return self._forward(*args, **kwargs)
        return self._forward_test(*args, **kwargs)
    
    # @profile
    def _forward(
            self,
            rays_o: torch.Tensor,
            rays_d: torch.Tensor,
            return_rgb_far: bool,
            return_rgb_near: bool,
            return_material: bool,
            return_jittered: bool,
            n_segs: int,
            near_far: Tuple[float, float],
            far_light_idx: int = None,
            near_light_state: torch.Tensor = None,
            **kwargs,
    ):
        """Volume rendering the given rays.

        Args:
            rays_o: (n_rays, 3) the origin of the rays
            rays_d: (n_rays, 3) the unit direction of the rays
            return_rgb_far: whether to return RGB radiance under the
                far light
            return_rgb_near: whether to return RGB radiance under the
                near light
            return_material: whether to return material attributes, including
                shading normal, albedo, roughness, and fresnel
            return_jittered: whether to return jittered material attributes
                and normal
            n_segs: the number of segments to sample for each ray
            near_far (Tuple[float, float]): the near bound and far bound for
                the ray marching.
            far_light_idx: the index of the far light. < 0 means no far light.
            near_light_state: [n_near_lights] the open state of the near
                lights. Should be floats in [0, 1].

        Returns:
            Dict[str, torch.Tensor] containing the rendered attributes.
            The returned values are in the linear color space (if applicable).
            Possible keys:
                - opacity: (n_rays, 1) the accumulated weight of the ray
                - depth: (n_rays, 1) the mean depth of the ray
                - normal: (n_rays, 3) the accumulated derived surface
                    normal
                - sample_sdf: (n_samples, 1) the sampled sdf values
                - sample_sdf_grad: (n_samples, 3) the sampled sdf gradients
                - sample_weight: (n_samples, ) the weight for each sample
                - sample_ray_indices: (n_samples, ) the ray indices for each
                    sample
                - sample_intervals: (n_samples, 1) the interval length for each
                    sample
                - nr_rgb_far: (n_rays, 3) the RGB radiance under the far
                    light with index :attr:`far_light_idx`
                - nr_rgb_near: (n_rays, 3) the RGB radiance under the near
                    light with state :attr:`near_light_state`
                - {brdf_key}: (n_rays, 3 or 1) the accumulated brdf
                    attributes. Keys are specified in self.brdf.material_dims
                - sample_{brdf_key}: (n_samples, 3 or 1) the sampled brdf
                    attributes. Keys are specified in self.brdf.material_dims
                - sample_jittered_{brdf_key}: (n_samples, 3 or 1) the sampled
                    jittered brdf attributes. Keys are specified in
                    self.brdf.material_dims
        """
        device = self.device
        n_rays = rays_o.shape[0]
        is_train = global_vars.training
        
        def alpha_fn(
                t_starts: torch.Tensor,
                t_ends: torch.Tensor,
                ray_indices: torch.Tensor
        ):
            """Query the alpha values for the given ray segments.

            Args:
                t_starts: (n_samples, ) the starting t values of the ray
                    segments
                t_ends: (n_samples, ) the ending t values of the ray segment
                ray_indices: (n_samples, ) to which ray each segment
                    belongs

            Returns:
                torch.Tensor: (n_samples, ) the alpha values for the given
                    ray segments
            """
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            t_mid = (t_starts + t_ends) / 2
            pts_xyz = t_origins + t_dirs * t_mid.view(-1, 1)
            seg_lengths = t_ends - t_starts
            if t_origins.shape[0] == 0:
                return torch.zeros((0,), device=self.device)
            retd = self.field.query_sdf(pts_xyz, with_grad=True)
            sdf = retd['sdf']
            sdf_grad = retd['sdf_grad']
            sdf_grad = F.normalize(sdf_grad, p=2, dim=-1, eps=1e-6)
            return self.field.get_alpha(
                sdf,
                normal=sdf_grad,
                view_dirs_in=t_dirs,
                seg_length=seg_lengths.view(-1, 1),
            ).view(-1)
        
        # step_length = self.step_length
        # if n_segs < 128:  # mainly for secondary rays
        #     step_length = self.step_length * 128 / (n_segs + 1)
        step_length = self.field.aabb_diag / n_segs
        
        with torch.no_grad():
            ray_indices, t_starts, t_ends = self.occ_grid.sampling(
                rays_o=rays_o,
                rays_d=rays_d,
                alpha_fn=alpha_fn,
                near_plane=near_far[0],
                far_plane=near_far[1],
                render_step_size=step_length,
                stratified=is_train,
                alpha_thre=1e-3,
            )
        
        # (n_samples, 1 or 3)
        sampled_o = rays_o[ray_indices]
        sampled_d = rays_d[ray_indices]
        sampled_t_mid = (t_starts + t_ends).view(-1, 1) / 2
        sampled_xyz = sampled_o + sampled_d * sampled_t_mid
        sampled_seg_lengths = (t_ends - t_starts).view(-1, 1)
        
        far_light_indices = [far_light_idx] if (return_rgb_far and
                                                far_light_idx >= 0) else []
        near_light_indices = [
            id for id, state in enumerate(near_light_state)
            if state == 1] if return_rgb_near else []
        
        # TODO: smoothness loss by jittering the sampled points
        # sampled_xyz_jitter = sampled_xyz + torch.rand_like(
        #         sampled_xyz) * self.jitter_std
        if return_rgb_near:
            flash_o = self.near_lights_nr.current_cam_pos
            self.near_lights_nr.current_cam_pos = flash_o[ray_indices]
        pts_attrs = self.field.query_field(
            sampled_xyz,
            return_sdf=True,
            return_sdf_grad=True,
            return_geo_feature=False,
            return_material=return_material,
            return_jittered=return_jittered,
            view_dirs_out=-sampled_d,
            far_light_indices=far_light_indices,
            near_light_indices=near_light_indices,
            near_lights=self.near_lights_nr,
        )
        if return_rgb_near:
            self.near_lights_nr.current_cam_pos = flash_o
        sdf = pts_attrs['sdf']
        sdf_grad = pts_attrs['sdf_grad']
        normal = F.normalize(sdf_grad, p=2, dim=-1, eps=1e-6)
        alpha = self.field.get_alpha(
            sdf,
            normal=normal,
            view_dirs_in=sampled_d,
            seg_length=sampled_seg_lengths,
        )
        weights, _ = nerfacc.render_weight_from_alpha(
            alpha.view(-1),
            ray_indices=ray_indices, n_rays=n_rays
        )
        
        # sampled_density = self.field.query_density(sampled_xyz).view(-1)
        # weights, _, alphas = nerfacc.render_weight_from_density(
        #     t_starts, t_ends, sampled_density,
        #     ray_indices=ray_indices, n_rays=n_rays)
        ray_opacity = nerfacc.accumulate_along_rays(
            weights, None,
            ray_indices=ray_indices, n_rays=n_rays
        )
        ray_depth = nerfacc.accumulate_along_rays(
            weights, sampled_t_mid,
            ray_indices=ray_indices, n_rays=n_rays
        )
        ray_depth /= ray_opacity.clamp(min=1e-6)
        ray_normal = nerfacc.accumulate_along_rays(
            weights, normal,
            ray_indices=ray_indices, n_rays=n_rays
        )
        ray_normal = F.normalize(ray_normal, p=2, dim=-1, eps=1e-6)
        attrs = {
            'opacity': ray_opacity,
            'depth': ray_depth,
            'normal': ray_normal
        }
        
        # jittered material
        if return_jittered:
            attrs.update(
                {
                    'sample_normal': normal,
                    'sample_jittered_normal': pts_attrs['normal_jitter'],
                }
            )
        
        if self.training:
            attrs.update(
                {
                    'sample_sdf': sdf,
                    'sample_sdf_grad': sdf_grad,
                    'sample_weights': weights,
                    'sample_ray_indices': ray_indices,
                    'sample_intervals': sampled_seg_lengths
                }
            )
        
        if return_rgb_far:
            if far_light_idx >= 0:
                rgb_far = pts_attrs[f'far_radiance_{far_light_idx}']
                ray_rgb_far = nerfacc.accumulate_along_rays(
                    weights, rgb_far,
                    ray_indices=ray_indices, n_rays=n_rays
                )
            else:
                ray_rgb_far = torch.zeros(n_rays, 3, device=device)
            attrs.update({'nr_rgb_far': ray_rgb_far})
        
        if return_rgb_near:
            ray_rgb_near = torch.zeros(n_rays, 3, device=device)
            for near_light_index in near_light_indices:
                rgb_near = pts_attrs[f'near_radiance_{near_light_index}']
                ray_rgb_near += nerfacc.accumulate_along_rays(
                    weights, rgb_near,
                    ray_indices=ray_indices, n_rays=n_rays
                )
            attrs.update({'nr_rgb_near': ray_rgb_near})
        
        if return_material:
            unpacked_material = self.unpack_brdf(pts_attrs['material'])
            for key in self.brdf.material_dims.keys():
                ray_attr = nerfacc.accumulate_along_rays(
                    weights, unpacked_material[key],
                    ray_indices=ray_indices, n_rays=n_rays
                )
                attrs.update({key: ray_attr})
            # jittered material
            if return_jittered:
                # return original sample
                attrs.update(
                    {f'sample_{key}': unpacked_material[key]
                     for key in self.brdf.material_dims.keys()}
                )
                # return jittered sample
                unpacked_material_jitter = self.unpack_brdf(
                    pts_attrs['material_jitter']
                )
                attrs.update(
                    {f'sample_jittered_{key}': unpacked_material_jitter[key]
                     for key in self.brdf.material_dims.keys()}
                )
        
        return attrs
    
    @torch.no_grad()
    # @profile
    def _forward_test(
            self,
            rays_o: torch.Tensor,
            rays_d: torch.Tensor,
            return_rgb_far: bool,
            return_rgb_near: bool,
            return_material: bool,
            n_segs: int,
            near_far: Tuple[float, float],
            far_light_idx: int = None,
            near_light_state: torch.Tensor = None,
            **kwargs,
    ):
        """Volume rendering the given rays in test mode."""
        device = self.device
        n_rays = rays_o.shape[0]
        
        min_samples = 1
        max_samples = n_segs
        iter_samples = total_samples = 0
        
        t_mins, t_maxs, hits = nerfacc.ray_aabb_intersect(
            rays_o, rays_d,
            aabbs=self.occ_grid.aabbs
        )
        
        n_grids = self.occ_grid.binaries.shape[0]  # should be 1
        t_sorted = torch.cat([t_mins, t_maxs], dim=-1)  # already sorted
        t_indices = torch.arange(
            0, 2 * n_grids, device=device, dtype=torch.int64
        ).expand(n_rays, n_grids * 2)
        
        near_planes = torch.full_like(
            rays_o[..., 0],
            fill_value=near_far[0]
        )
        far_planes = torch.full_like(
            rays_o[..., 0],
            fill_value=near_far[1]
        )
        
        early_stop_eps = 1e-3
        opacity_thres = 1 - early_stop_eps
        alpha_thres = 1e-3
        
        # initialize ray attributes
        ray_opacity = torch.zeros(n_rays, 1, device=device)
        ray_depth = torch.zeros(n_rays, 1, device=device)
        ray_mask = torch.ones(n_rays, device=device, dtype=torch.bool)
        ray_normal = torch.zeros(n_rays, 3, device=device)
        if return_rgb_far:
            ray_rgb_far = torch.zeros(n_rays, 3, device=device)
        if return_rgb_near:
            ray_rgb_near = torch.zeros(n_rays, 3, device=device)
        material_rays = {}
        if return_material:
            for key, dim in self.brdf.material_dims.items():
                material_rays[key] = torch.zeros(n_rays, dim, device=device)
        
        # convert near light state to indices with state > 0
        far_light_indices = [far_light_idx] if (return_rgb_far and
                                                far_light_idx >= 0) else []
        near_light_indices = [
            id for id, state in enumerate(near_light_state)
            if state == 1] if return_rgb_near else []
        
        while iter_samples < max_samples:
            n_alive = ray_mask.sum().item()
            if n_alive == 0:
                break
            
            # the number of samples to add on each ray
            n_samples = max(min(n_rays // n_alive, 64), min_samples)
            iter_samples += n_samples
            
            # ray marching
            intervals, samples, termination_planes = nerfacc.traverse_grids(
                # rays
                rays_o,  # (n_rays, 3)
                rays_d,  # (n_rays, 3)
                # grids
                self.occ_grid.binaries,  # [m, resx, resy, resz]
                self.occ_grid.aabbs,  # [m, 6]
                # options
                near_planes,  # [n_rays]
                far_planes,  # [n_rays]
                self.step_length,
                0.0,
                n_samples,
                True,
                ray_mask,
                # pre-compute intersections
                t_sorted,  # [n_rays, m*2]
                t_indices,  # [n_rays, m*2]
                hits,  # [n_rays, m]
            )
            t_starts = intervals.vals[intervals.is_left]
            t_ends = intervals.vals[intervals.is_right]
            ray_indices = samples.ray_indices[samples.is_valid]
            packed_info = samples.packed_info
            
            sampled_o = rays_o[ray_indices]
            sampled_d = rays_d[ray_indices]
            sampled_t_mid = (t_starts + t_ends).view(-1, 1) / 2
            sampled_xyz = sampled_o + sampled_d * sampled_t_mid
            sampled_seg_lengths = (t_ends - t_starts).view(-1, 1)
            
            if return_rgb_near:
                flash_o = self.near_lights_nr.current_cam_pos
                self.near_lights_nr.current_cam_pos = flash_o[ray_indices]
            pts_attrs = self.field.query_field(
                sampled_xyz,
                return_sdf=True,
                return_sdf_grad=True,
                return_geo_feature=False,
                return_material=return_material,
                view_dirs_out=-sampled_d,
                far_light_indices=far_light_indices,
                near_light_indices=near_light_indices,
                near_lights=self.near_lights_nr,
            )
            if return_rgb_near:
                self.near_lights_nr.current_cam_pos = flash_o
            sdf = pts_attrs['sdf']
            sdf_grad = pts_attrs['sdf_grad']
            normal = F.normalize(sdf_grad, p=2, dim=-1, eps=1e-6)
            alpha = self.field.get_alpha(
                sdf,
                normal=normal,
                view_dirs_in=sampled_d,
                seg_length=sampled_seg_lengths,
            )
            # get weights with prefix_trans
            weights, _ = nerfacc.render_weight_from_alpha(
                alpha.view(-1),
                ray_indices=ray_indices, n_rays=n_rays,
                prefix_trans=1 - ray_opacity[ray_indices].view(-1)
            )
            
            nerfacc.volrend.accumulate_along_rays_(
                weights, None,
                ray_indices=ray_indices, outputs=ray_opacity
            )
            nerfacc.volrend.accumulate_along_rays_(
                weights, sampled_t_mid,
                ray_indices=ray_indices, outputs=ray_depth
            )
            nerfacc.volrend.accumulate_along_rays_(
                weights, normal,
                ray_indices=ray_indices, outputs=ray_normal
            )
            
            if return_rgb_far:
                if far_light_idx >= 0:
                    rgb_far = pts_attrs[f'far_radiance_{far_light_idx}']
                    nerfacc.volrend.accumulate_along_rays_(
                        weights, rgb_far,
                        ray_indices=ray_indices, outputs=ray_rgb_far
                    )
            
            if return_rgb_near:
                for near_light_index in near_light_indices:
                    rgb_near = pts_attrs[f'near_radiance_{near_light_index}']
                    nerfacc.volrend.accumulate_along_rays_(
                        weights, rgb_near,
                        ray_indices=ray_indices, outputs=ray_rgb_near
                    )
            
            if return_material:
                unpacked_material = self.unpack_brdf(pts_attrs['material'])
                for key in material_rays.keys():
                    nerfacc.volrend.accumulate_along_rays_(
                        weights, unpacked_material[key],
                        ray_indices=ray_indices, outputs=material_rays[key]
                    )
            
            # update near_planes using termination planes
            near_planes = termination_planes
            # update ray status
            ray_mask = torch.logical_and(
                # early stopping
                ray_opacity.view(-1) <= opacity_thres,
                # remove rays that have reached the far plane
                packed_info[:, 1] == n_samples,
            )
            total_samples += ray_indices.shape[0]
        
        # collect and post-processing
        ray_depth = ray_depth / ray_opacity.clamp(min=1e-6)
        ray_normal = F.normalize(ray_normal, p=2, dim=-1, eps=1e-6)
        ray_attrs = {
            'depth': ray_depth,
            'opacity': ray_opacity,
            'normal': ray_normal,
        }
        
        if return_rgb_far:
            ray_attrs.update(
                {
                    'nr_rgb_far': ray_rgb_far,
                }
            )
        
        if return_rgb_near:
            ray_rgb_near /= self.n_near_lights
            ray_attrs.update(
                {
                    'nr_rgb_near': ray_rgb_near,
                }
            )
        
        if return_material:
            ray_attrs.update(material_rays)
        
        return ray_attrs
    
    def get_save_dict(self):
        """
        Get the dictionary of self attributes to be saved.
        Returns:
            Dict: the dictionary to be saved
        """
        # get model arguments
        model_args = vars(self).copy()
        model_args = {
            k: v for k, v in model_args.items() if not k.startswith('_')
        }
        # pprint(model_args)
        # get state dict
        ckpt = {
            'model_args': model_args,
            'state_dict': self.state_dict(),
        }
        # if self.alpha_mask is not None:
        #     alpha_volume = self.alpha_mask.alpha_volume.bool().cpu().numpy()
        #     ckpt['alpha_mask.shape'] = alpha_volume.shape
        #     ckpt['alpha_mask.mask'] = np.packbits(alpha_volume.reshape(-1))
        #     ckpt['alpha_mask.aabb'] = self.alpha_mask.aabb.cpu()
        return ckpt
    
    @torch.no_grad()
    def get_dense_sdf(
            self,
            dense_sample_size: Union[Tuple[int], int, torch.Tensor] = None,
    ):
        """Densely sample the grid and compute the sdf values.

        Args:
            dense_sample_size: the number of samples in each dimension

        Returns:
            A tuple of two torch.Tensor:

            - dense_xyz (torch.Tensor): [*dense_sample_size, 3] the densely
                positions
            - sdf (torch.Tensor): [*dense_sample_size] the sdf values
        """
        if dense_sample_size is None:
            dense_sample_size = tuple(self.grid_size)
        elif isinstance(dense_sample_size, torch.Tensor):
            dense_sample_size = tuple(dense_sample_size)
        elif isinstance(dense_sample_size, int):
            dense_sample_size = (dense_sample_size,) * 3
        
        # densely sample the grid
        dense_xyz = torch.meshgrid(
            torch.linspace(
                0, 1, dense_sample_size[0],
                device=self.device
            ),
            torch.linspace(
                0, 1, dense_sample_size[1],
                device=self.device
            ),
            torch.linspace(
                0, 1, dense_sample_size[2],
                device=self.device
            ),
            indexing='ij'
        )
        dense_xyz = torch.stack(dense_xyz, dim=-1).view(-1, 3)
        dense_xyz = self.aabb[0] * (1 - dense_xyz) + self.aabb[1] * dense_xyz
        
        # compute sdf by chunks
        dense_xyz = dense_xyz.view(-1, 3)
        chunk_size = 1048576
        xyz_chunks = torch.split(dense_xyz, chunk_size)
        sdf_chunks = []
        for xyz_chunk in tqdm(xyz_chunks, desc='Querying dense SDF'):
            sdf_chunk = self.field.query_sdf(xyz_chunk)['sdf']
            sdf_chunks.append(sdf_chunk)
        sdf = torch.cat(sdf_chunks, dim=0)
        
        dense_xyz = dense_xyz.view(*dense_sample_size, 3)
        sdf = sdf.view(*dense_sample_size)
        
        return dense_xyz, sdf
    
    def get_regularization_loss(
            self,
            reg_type: str,
            component: str,
    ):
        """
        Compute the regularization loss :attr:`reg_type` for the given
        :attr:`component`.

        Args:
            reg_type: the type of the regularization (e.g. 'L1')
            component: the component to be regularized (e.g. 'light intensity')

        Returns:
            torch.Tensor: scalar, the regularization loss
        """
        components = []
        effective_on = {}
        if self.pbr_enabled:
            # near light intensity is soft-bounded by near light decoder
            effective_on['light_intensity'] = [self.far_lights]
            # far light color should not be assumed to be neutral unless it
            # is the only light
            effective_on['light_color_balance'] = [self.near_lights_pbr] if \
                self.near_lights_pbr is not None else [self.far_lights]
        else:
            effective_on['light_intensity'] = []
            effective_on['light_color_balance'] = [self.near_lights_nr]
        if component in ['light_intensity', 'light_color_balance']:
            for light in effective_on[component]:
                if light is None:
                    continue
                # [n_lights, 3]
                per_cond_mean_intensity = light.get_mean_intensity()
                for intensity in per_cond_mean_intensity:
                    if component == 'light_intensity':
                        components.append(intensity)
                    else:  # light_color_balance
                        mean_overall_intensity = intensity.mean()
                        intensity = intensity / mean_overall_intensity.clamp(
                            min=1e-6
                        )
                        color_balance_diff = intensity - 1
                        components.append(color_balance_diff)
        else:
            raise ValueError(f'Unknown component: {component}')
        
        supported_reg_types = ['L1', 'L2']
        loss = torch.zeros(1, device=self.device)
        if reg_type == 'L1':
            for comp in components:
                loss += torch.mean(torch.abs(comp))
        elif reg_type == 'L2':
            for comp in components:
                loss += torch.mean(comp ** 2)
        else:
            raise ValueError(
                f'Unknown regularization type: {reg_type} for component '
                f'{component}. Supported types: {supported_reg_types}'
            )
        return loss
    
    def sample_volume(self, n_pts: int):
        """
        Uniformly sample 3D points within the AABB.
        Args:
            n_pts: the number of points to sample

        Returns:
            torch.Tensor: (n_pts, 3) the sampled points
        """
        return torch.rand(n_pts, 3, device=self.device) * self.aabb_size + \
            self.aabb[0]


def compute_smoothness_mse(
        value: torch.Tensor,
        value_jitter: torch.Tensor,
        value_type: str,
        error_type: str
) -> torch.Tensor:
    """Compute relative smoothness loss for the given value.

    Args:
        value: [..., 3] the value to be smoothed
        value_jitter: [..., 3] the jittered value
        value_type: the type of the value (e.g. 'normal', 'rgb', 'brdf')
        error_type: the type of the error (e.g. 'mse', 'l1')

    Returns:
        torch.Tensor: [..., 1] the smoothness loss
    """
    # compute the relative smoothness loss
    if value_type != 'normal':
        base = torch.maximum(value, value_jitter).clip(min=1e-6)
        difference = (value - value_jitter) / base
    else:
        # base = torch.maximum(value.abs(), value_jitter.abs()
        #                      ).clip(min=1e-6)
        difference = value - value_jitter
        # difference = (value - value_jitter) / base
        # dot = torch.sum(value * value_jitter, dim=-1, keepdim=True)
        # value_norm = torch.norm(value, p=2, dim=-1, keepdim=True)
        # value_jitter_norm = torch.norm(value_jitter, p=2, dim=-1, keepdim=True)
        # difference = value_norm * value_jitter_norm - dot
    
    if error_type == 'mse':
        difference = torch.sum(difference ** 2, dim=-1, keepdim=True)
    elif error_type == 'l1':
        difference = torch.sum(torch.abs(difference), dim=-1, keepdim=True)
    
    return difference
