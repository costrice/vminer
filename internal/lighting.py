"""
This file implements different types of near- and far-field lighting.
"""
import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from internal import geo_utils, sph_harm, utils


def normalize_envmap(
        x: torch.Tensor
):
    """Normalize the intensity of the environment map."""
    return utils.linrgb2srgb((x / x.mean() * 0.3).clamp(0, 1))


def regular_sample_envmap(
        h: int,
        w: int,
        device: torch.device
):
    """Regularly sample the environment map to get the directions of each pixel.

    Args:
        h: the height of the environment map
        w: the width of the environment map
        device: the device to put the output on

    Returns:
        torch.Tensor: (h, w, 3), the view directions of each pixel
    """
    # sample image coordinates
    step_size_h = 2.0 / h
    step_size_w = 2.0 / w
    image_coord_h = torch.linspace(
        -1.0 + step_size_h / 2.0,
        1.0 - step_size_h / 2.0,
        h, device=device
    )
    image_coord_w = torch.linspace(
        -1.0 + step_size_w / 2.0,
        1.0 - step_size_w / 2.0,
        w, device=device
    )
    image_coord_h, image_coord_w = torch.meshgrid(
        image_coord_h, image_coord_w,
        indexing='ij'
    )
    # convert to view direction
    view = geo_utils.image_coord2view_vec(
        torch.stack([image_coord_h, image_coord_w], dim=-1)
    )
    return view


class Lighting(nn.Module):
    """
    Base class for lighting.
    """
    
    def __init__(
            self,
            n_lights: int,
            device: torch.device,
            requires_grad: bool,
    ):
        """Initialize the lighting.

        Args:
            n_lights: the number of light conditions
            device: the device to put the lighting on
            requires_grad: whether the lighting parameters require gradients
        """
        super().__init__()
        self.n_lights = n_lights
        self.device = device
        self.requires_grad = requires_grad
        self.params = None
        self.which = None
        self.light_type = None
        self.actual_param_shape = None
    
    def init_param(self):
        """Initialize the parameters of the lighting."""
        
        # convert to nn.Parameter
        self.params = nn.Parameter(
            self.params, requires_grad=self.requires_grad
        )
    
    def to_device(self, device: torch.device):
        """Move the lighting to the given device.

        Args:
            device: the device to put the lighting on
        """
        self.device = device
        self.params = nn.Parameter(
            self.params.to(device), requires_grad=self.requires_grad
        )
    
    def get_optim_params(self):
        """Get the parameters that need to be optimized.

        Returns:
            List[torch.Tensor]: the parameters that need to be optimized
        """
        if self.requires_grad:
            return [self.params]
        else:
            return []
    
    def get_mean_intensity(
            self
    ) -> torch.Tensor:
        """Get per-channel mean intensity for regularization.

        Returns:
            torch.Tensor: (n_lights, 3), the mean intensity for each light
        """
        raise NotImplementedError
    
    def get_actual_params(
            self
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Convert the lighting parameters to the actual lighting parameters.

        Returns:
            A Tuple of 1 or 2 torch.Tensor, containing:

            - light_xyz (Optional[torch.Tensor]): (n_lights, 3) the near
                light positions
            - intensity_param (torch.Tensor): (n_lights, ?) the actual
                intensity parameters for RGB
        """
        raise NotImplementedError


class FarLight(Lighting):
    """
    Implements different types of far lighting.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.which = 'far'
        # self.light_type = None
        self.dark_actual_param = None
        self.actual_param_overriden = None
    
    def to_device(self, device: torch.device):
        super().to_device(device)
        self.dark_actual_param = self.dark_actual_param.to(device)
        if self.actual_param_overriden is not None:
            self.actual_param_overriden = self.actual_param_overriden.to(device)
    
    def init_param(self):
        super().init_param()
    
    def get_actual_params(self, index: int = None) -> torch.Tensor:
        """Get actual lighting parameters with index.

        Args:
            index: the index of the light source to get. If None, return
                all light sources. If -1, return the dark actual param.

        Returns:
            torch.Tensor: [(n_lights,) *param_shape], the actual parameters
        """
        if index == -1:
            return self.dark_actual_param
        
        if self.actual_param_overriden is not None:
            actual_params = self.actual_param_overriden
        else:
            actual_params = self.get_non_overridden_actual_params()
        
        if index is not None:
            actual_params = actual_params[index]
        return actual_params
    
    def get_non_overridden_actual_params(self) -> torch.Tensor:
        """"Get all actual lighting parameters.

        Returns:
            torch.Tensor: [n_lights, *param_shape], the actual parameters
        """
        raise NotImplementedError
    
    def query_at_directions(
            self,
            far_light_idx: int,
            view_dirs: torch.Tensor,
    ) -> torch.Tensor:
        """Query the lighting at the given directions.

        Args:
            far_light_idx: the index of the far light source to query. -1 means
                no far light source is used.
            view_dirs: (..., 3) the view directions

        Returns:
            torch.Tensor: (..., 3), the RGB radiance from each direction
        """
        raise NotImplementedError
    
    def to_envmap(
            self,
            envmap_dirs: torch.Tensor
    ):
        """Bake the far light to an environment map.

        Args:
            envmap_dirs: (..., 3), the directions of each pixel in the
                environment map

        Returns:
            torch.Tensor: [n_lights, ..., 3), the RGB radiance of each pixel
                for each light source
        """
        envmaps = []
        for i in range(self.n_lights):
            envmaps.append(
                self.query_at_directions(
                    i, envmap_dirs
                )
            )
        return torch.stack(envmaps, dim=0)
    
    def override_actual_params(
            self,
            actual_params: torch.Tensor,
            mutable: bool = False,
    ):
        """Set the actual parameters to the given values.

        Args:
            actual_params: [n_lights, *param_shape], the actual parameters to
                use
            mutable: whether to allow the actual parameters to be further
                modified
        """
        raise NotImplementedError


class SGFarLight(FarLight):
    """
    Implements spherical Gaussian far lighting.
    """
    
    def __init__(
            self,
            n_lobes: int,
            reparam_lobes: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.light_type = 'sg'
        self.n_lobes = n_lobes
        self.reparam_lobes = reparam_lobes
        if self.reparam_lobes:
            self.lobe_center = None
            self.lobe_s = None
            self.lobe_t = None
            self.lobe_move_range = None
        self.actual_param_shape = (n_lobes, 7)  # 3 + 1 + 3
        # else:
        #     self.actual_param_shape = (n_lobes, 6)
        self.init_param()
    
    def to_device(self, device: torch.device):
        super().to_device(device)
        if self.reparam_lobes:
            self.lobe_center = self.lobe_center.to(device)
            self.lobe_s = self.lobe_s.to(device)
            self.lobe_t = self.lobe_t.to(device)
            self.lobe_move_range = self.lobe_move_range.to(device)
    
    def init_param(self):
        n_lobes = self.n_lobes
        n_lights = self.n_lights
        device = self.device
        
        # deterministically initialize the lobe positions
        lobes = geo_utils.fibonacci_sphere(n_lobes, device)
        if self.reparam_lobes:
            # compute mean distance between neighboring lobes
            dots = torch.matmul(
                lobes, lobes.transpose(0, 1)
            )
            dots = dots - torch.eye(dots.shape[0], device=device)
            max_dots = dots.max(dim=1)[0]
            # compute the tangential plane of each lobe
            lobe_s, lobe_t = geo_utils.coordinate_system(lobes)
            
            self.lobe_center = lobes.view(1, n_lobes, 3)
            self.lobe_s = lobe_s.view(1, n_lobes, 3)
            self.lobe_t = lobe_t.view(1, n_lobes, 3)
            # move in neighborhood of the lobe
            self.lobe_move_range = (1 - max_dots).view(1, n_lobes, 1) * 2
            lobe_params = torch.zeros(n_lights, n_lobes, 2, device=device)
        else:
            lobe_params = lobes.view(1, n_lobes, 3).expand(
                n_lights, n_lobes, 3
            ).contiguous()
        sharpness_params = 3 * torch.ones(n_lights, n_lobes, 1, device=device)
        intensity_params = torch.zeros(n_lights, n_lobes, 3, device=device)
        
        self.params = torch.cat(
            [lobe_params, sharpness_params, intensity_params],
            dim=-1
        )
        
        # used when no far light is used
        self.dark_actual_param = torch.tensor(
            [0., 0., 1., 20, 0., 0., 0.], device=device
        ).view(1, 7).expand(n_lobes, 7).contiguous()
        
        super().init_param()
    
    def get_mean_intensity(
            self
    ) -> torch.Tensor:
        intensity = self.get_actual_params()[..., 4:7]
        # (n_lights, n_lobes, 3) -> (n_lights, 3)
        return intensity.mean(dim=1)
    
    def get_non_overridden_actual_params(self, index: int = None):
        """Convert self.params to true SG parameters.

        Returns:
            torch.Tensor: [n_lights, n_lobes, 7], contains SG params
                including [position, sharpness, RGB intensity] for each lobe.
        """
        param = self.params
        if self.reparam_lobes:
            s_step = torch.tanh(param[..., 0:1]) * self.lobe_move_range
            t_step = torch.tanh(param[..., 1:2]) * self.lobe_move_range
            lobe = (self.lobe_center +
                    s_step * self.lobe_s + t_step * self.lobe_t)
            param = param[..., 2:]
        else:
            lobe = param[..., :3]
            param = param[..., 3:]
        lobe = F.normalize(lobe, dim=-1, p=2, eps=1e-6)
        sharpness = torch.exp(param[..., 0:1].clamp(max=math.log(2000)))
        intensity = torch.exp(param[..., 1:4].clamp(max=13))  # e^13 = 4.4e5
        actual_param = torch.cat([lobe, sharpness, intensity], dim=-1)
        
        return actual_param
    
    def query_at_directions(
            self,
            far_light_idx: int,
            view_dirs: torch.Tensor,
    ) -> torch.Tensor:
        if far_light_idx == -1:
            return torch.zeros_like(view_dirs)
        
        input_shape = view_dirs.shape[:-1]
        actual_param = self.get_actual_params(index=far_light_idx)
        actual_param = actual_param.view(
            *([1] * len(input_shape)), *self.actual_param_shape
        )
        view_dirs = view_dirs.view(*input_shape, 1, 3)
        view_dirs = F.normalize(view_dirs, dim=-1, p=2, eps=1e-6)
        
        # unpack sg light sources
        # (n_lobes, 3 or 1)
        lobes = actual_param[..., :3]
        sharpness = actual_param[..., 3:4]
        intensity = actual_param[..., 4:7]
        
        # query SG intensity
        # (..., n_lobes, 3)
        rgb = (intensity *
               torch.exp(
                   sharpness * (
                           torch.sum(
                               lobes * view_dirs,
                               dim=-1, keepdim=True
                               ) - 1)
                   ))
        # normalize w.r.t. sharpness
        rgb *= sharpness / (2 * torch.pi * (1 - torch.exp(-2 * sharpness)))
        rgb = torch.sum(rgb, dim=-2)
        
        # idk why there are nan values, but this should fix it
        rgb = torch.nan_to_num(rgb)
        
        return rgb
    
    def override_actual_params(
            self,
            actual_params: torch.Tensor,
            mutable: bool = False,
    ):
        """Set the actual parameters to the given values.

        Args:
            actual_params: [n_lights, n_lobes, 7], the actual parameters to use
            mutable: whether to allow the actual parameters to be further
                modified
        """
        # pre-process: sort lobes
        n_lights, n_lobes, _ = actual_params.shape
        self.n_lights = n_lights
        if n_lobes != self.n_lobes:
            raise ValueError(
                f'Number of lobes mismatch: {n_lobes} vs {self.n_lobes}'
            )
        
        if not mutable:
            self.requires_grad = False
            self.actual_param_overriden = actual_params
        else:
            # don't know how to inverse the process
            raise NotImplementedError


class PixelFarLight(FarLight):
    """
    Implements environment map far lighting stored in pixels.
    """
    
    def __init__(
            self,
            envmap_h: int,
            envmap_w: int,
            use_prefilter: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.light_type = 'pixel'
        self.h = envmap_h
        self.w = envmap_w
        self.max_log_intensity = 13  # e^13 = 4.4e5
        self.light_dirs = regular_sample_envmap(  # [h, w, 3]
            self.h, self.w, self.device
        )
        # sin_theta = torch.sqrt(1 - self.light_dirs[..., 2:3] ** 2)
        # self.area_size = 1 / self.w * sin_theta / self.h * torch.pi ** 2
        self.actual_param_shape = (self.h, self.w, 6)
        self.init_param()
        self.use_prefilter = use_prefilter
        self.prefiltered_envmap = None
        self.prefilter()
    
    def to_device(self, device: torch.device):
        super().to_device(device)
        self.light_dirs = self.light_dirs.to(device)
        if self.prefiltered_envmap is not None:
            self.prefiltered_envmap = self.prefiltered_envmap.to(device)
    
    def init_param(self):
        self.params = torch.zeros(
            self.n_lights, self.h * self.w, 3,
            dtype=torch.float, device=self.device
        ).uniform_(0, 2)
        # used when no far light is used
        self.dark_actual_param = torch.zeros(
            self.h, self.w, 6, device=self.device
        )
        
        super().init_param()
    
    def prefilter(self):
        """Prefilter the environment map for diffuse reflection."""
        if not self.use_prefilter:
            return
        # [n_lights, h, w, 6]
        actual_params = self.get_actual_params()
        intensity = actual_params[..., 3:6]
        cos_theta = torch.sqrt(1 - self.light_dirs[..., 2:3] ** 2)
        # print(cos_theta.sum() / self.h / self.w)  # = 2 / pi
        # get self dot product of light directions
        light_dirs = self.light_dirs.view(self.h * self.w, 3)
        light_dot = torch.matmul(
            light_dirs, light_dirs.transpose(0, 1)
        )
        light_dot = light_dot.clamp(min=0, max=1)
        intensity = intensity.view(self.n_lights, self.h * self.w, 3)
        intensity = intensity * cos_theta.view(1, self.h * self.w, 1)
        intensity = intensity.permute(0, 2, 1) @ light_dot
        intensity = intensity.permute(0, 2, 1).view(
            self.n_lights, self.h, self.w, 3
        )
        intensity = intensity / self.h / self.w * torch.pi
        self.prefiltered_envmap = intensity
    
    def get_mean_intensity(
            self
    ) -> torch.Tensor:
        intensity = self.get_actual_params()[..., 3:6]
        # [n_lights, h, w, 3] -> [n_lights, 3]
        return intensity.mean(dim=(1, 2))
    
    def get_non_overridden_actual_params(self):
        """Convert self.params to true envmaps.

        Returns:
            torch.Tensor: [n_lights, h, w, 6], contains directions (for
                convenience) and RGB intensity for each pixel.
        """
        intensity = torch.exp(
            self.params.clamp(max=self.max_log_intensity)
        ).view(self.n_lights, self.h, self.w, 3)
        actual_params = torch.cat(
            [self.light_dirs.view(
                1, self.h, self.w, 3
            ).expand(
                self.n_lights, self.h, self.w, 3
            ),
                intensity], dim=-1
        )
        return actual_params
    
    def query_at_directions(
            self,
            far_light_idx: int,
            view_dirs: torch.Tensor,
            prefiltered: bool = False,
    ) -> torch.Tensor:
        if far_light_idx == -1:
            return torch.zeros_like(view_dirs)
        
        input_shape = view_dirs.shape[:-1]
        
        if not prefiltered:
            actual_params = self.get_actual_params(index=far_light_idx)
            # actual_params = actual_params.view(
            #     *([1] * len(input_shape)), self.h, self.w, 6)
            # view_dirs = view_dirs.view(*input_shape, 1, 1, 3)
            
            # unpack envmap light sources
            # [h, w, 3]
            intensity = actual_params[..., 3:6]
        else:
            intensity = self.prefiltered_envmap[far_light_idx]
        
        # convert view dirs onto image coordinates
        coord_hw = geo_utils.view_vec2image_coord(view_dirs)  # in [-1, 1],
        
        # use bilinear interpolation to query envmap
        coord_wh = torch.stack([coord_hw[..., 1], coord_hw[..., 0]], dim=-1)
        rgb = F.grid_sample(
            intensity.permute(2, 0, 1).unsqueeze(0),
            coord_wh.view(1, 1, -1, 2),
            align_corners=True, mode='bilinear'
        )
        rgb = rgb.squeeze(0).permute(1, 2, 0).view(*input_shape, 3)
        
        return rgb
    
    def to_envmap(
            self,
            envmap_dirs: torch.Tensor = None,
    ):
        # special case: if the envmap is the same as the light directions,
        # directly return the intensity without querying
        if envmap_dirs is None or self.light_dirs.shape == envmap_dirs.shape:
            return self.get_actual_params()[..., 3:6]
        else:
            return super().to_envmap(envmap_dirs)
    
    def override_actual_params(
            self,
            actual_params: torch.Tensor,
            mutable: bool = False,
    ):
        """Set the actual parameters to the given values.

        Args:
            actual_params: [n_lights, h, w, 3], the actual parameters to use
            mutable: whether to allow the actual parameters to be further
                modified
        """
        # pre-process: down- or up-sample to self.h, self.w
        assert actual_params.shape[3] == 3, \
            f'Do not specify directions in actual_params: {actual_params.shape}'
        n_lights, h, w, _ = actual_params.shape
        if h != self.h or w != self.w:
            mode = 'area' if h > self.h else 'bilinear'
            actual_params = F.interpolate(
                actual_params.permute(0, 3, 1, 2),
                (self.h, self.w),
                mode=mode
            ).permute(0, 2, 3, 1)
        self.n_lights = n_lights
        assert actual_params.min() >= 0, \
            f'Negative value detected in envmap: {actual_params.min()}'
        
        # set params
        if not mutable:
            self.requires_grad = False
            actual_params = torch.cat(
                [
                    self.light_dirs.unsqueeze(0)
                    .expand(self.n_lights, -1, -1, -1),
                    actual_params,
                ], dim=-1
            )
            self.actual_param_overriden = actual_params
        else:
            # inverse function of self.get_actual_params()
            new_max_log_intensity = torch.log(actual_params.max()).item()
            if new_max_log_intensity > self.max_log_intensity:
                self.max_log_intensity = new_max_log_intensity
            self.params = nn.Parameter(
                torch.log(actual_params.clamp(min=1e-9)).view(
                    n_lights, self.h * self.w, 3
                ),
                requires_grad=self.requires_grad
            )
        
        self.prefilter()


class NearLight(Lighting):
    """
    Implements different types of near lighting.
    """
    
    def __init__(
            self,
            pos_types: List[str],
            **kwargs
    ):
        super().__init__(**kwargs)
        self.which = 'near'
        
        assert len(pos_types) == self.n_lights, \
            f'Length of near light pos_types ({pos_types}) ' \
            f'does not match the number of near lights ({self.n_lights})'
        for pos_type in pos_types:
            assert pos_type in ['fixed', 'collocated'], \
                f'Unrecognized near light pos_type: {pos_type}'
        self.pos_types = pos_types
        # self.need_flash_o = 'collocated' in self.pos_types
        self.fixed_pos_mask = None
        self.current_cam_pos: torch.Tensor = None
    
    def to_device(self, device: torch.device):
        super().to_device(device)
        self.fixed_pos_mask = self.fixed_pos_mask.to(device)
    
    def init_param(self):
        super().init_param()
        fixed_pos_mask = [pos_type != 'collocated'
                          for pos_type in self.pos_types]
        # [n_lights, ]
        self.fixed_pos_mask = torch.tensor(
            fixed_pos_mask, dtype=torch.bool, device=self.device
        ).view(-1)
    
    def get_actual_params(self, index: int = None):
        """Get near light positions. Intensity is leaved to subclasses.

        Args:
            index: the index of the light source to get. If None, return
                all light sources.

        Returns:
            A tuple containing:

            - light_xyz (torch.Tensor): [(n_lights,) 3], the xyz position of
                each light source. If a light source is collocated with the
                camera, its xyz position is set to 0.
            - intensity_param: None, leave to subclasses
        """
        light_xyz = self.params[..., :3]
        light_xyz = light_xyz * self.fixed_pos_mask.view(-1, 1)
        if index is None:
            return light_xyz, None
        else:
            return light_xyz[index], None
    
    def get_intensity_and_relative_dir_dist(
            self,
            pts_xyz: torch.Tensor,
            # flash_o: torch.Tensor,
    ):
        """Compute the intensity, the relative direction and distance from each
        point to each near light. Should set self.current_cam_pos to camera
        origins for each ray before calling this function.

        Args:
            pts_xyz: (n_pts, 3) the positions of the points

        Returns:
            A tuple containing 3 tensors:

                - wi (torch.Tensor): (n_pts, n_lights, 3]
                    the relative direction of each near light from each point
                - dist (torch.Tensor): (n_pts, n_lights, 1)
                    the distance of each near light for each point
                - intensities (torch.Tensor): (n_pts, n_lights, 3]
                    the RGB intensity of each near light seen from each point
        """
        n_pts = pts_xyz.shape[0]
        if not isinstance(self.current_cam_pos, torch.Tensor):
            raise ValueError(
                'current_cam_pos is not set. Please set it to the camera '
                'origin for each ray before calling this function.'
            )
        elif self.current_cam_pos.shape != (n_pts, 3):
            raise ValueError(
                f'current_cam_pos has shape {self.current_cam_pos.shape}, '
                f'but expected {(n_pts, 3)}.'
            )
        flash_o = self.current_cam_pos.view(-1, 1, 3).expand(n_pts, -1, -1)
        # get light positions
        light_xyz, _ = self.get_actual_params()
        light_xyz = light_xyz.view(1, self.n_lights, 3).repeat(n_pts, 1, 1)
        # replace collocated light positions with camera positions
        light_xyz = (light_xyz * self.fixed_pos_mask.view(1, -1, 1) +
                     flash_o * (~self.fixed_pos_mask).view(1, -1, 1))
        wi = light_xyz - pts_xyz.view(n_pts, 1, 3)
        dist = torch.norm(wi, dim=-1, keepdim=True)
        dist = torch.clamp(dist, min=1e-2)  # avoid inf irradiance
        wi = F.normalize(wi, dim=-1, p=2, eps=1e-8)
        # leave intensity to subclasses
        return wi, dist, None


class SHNearLight(NearLight):
    """
    Implements point near lighting with anisotropic spherical harmonics exitant
    radiance.
    """
    
    def __init__(
            self,
            sh_order: int,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.light_type = 'sh_point'
        
        self.sh_order = sh_order
        
        self.init_param()
    
    def to_device(self, device: torch.device):
        super().to_device(device)
    
    def init_param(self):
        device = self.device
        intensity_params = torch.ones(
            self.n_lights, 3, (self.sh_order + 1) ** 2, device=device
        )
        # # randomly initialize the intensity
        # intensity_params *= torch.rand(
        #     self.n_lights, 1, (self.sh_order + 1) ** 2, device=device
        # ).float() * 2
        # brighter for collocated lights as they are further away
        for i, pos_type in enumerate(self.pos_types):
            if pos_type == 'collocated':
                intensity_params[i] *= 3  # e^3 = 20
                # intensity_params[i] *= 1  # e^1 = 2.7
            else:
                intensity_params[i] *= 1  # e^1 = 2.7
        intensity_params = intensity_params.view(
            self.n_lights, 3 * (self.sh_order + 1) ** 2
        )
        
        position_params = torch.randn(
            self.n_lights, 3, device=self.device
        )
        # init position to be 2 meters away from the object? TODO
        position_params = F.normalize(
            position_params, dim=-1, p=2, eps=1e-6
        ) * 2
        # only at upper hemisphere
        position_params[..., 2] = torch.abs(position_params[..., 2])
        # self.params = torch.cat(
        #     [position_params, intensity_params], dim=-1)
        self.params = nn.Parameter(
            torch.cat([position_params, intensity_params], dim=-1),
            requires_grad=self.requires_grad
        )
        
        super().init_param()
    
    def get_mean_intensity(
            self
    ) -> torch.Tensor:
        _, intensity = self.get_actual_params()
        # [n_lights, 3, (sh_order + 1) ** 2] -> [n_lights, 3]
        intensity = intensity.mean(dim=-1)
        return intensity
    
    def get_actual_params(self, index: int = None):
        """Convert self.params to true SH parameters.

        Args:
            index: the index of the light source to get. If None, return
                all light sources.

        Returns:
            A tuple containing:

            - light_xyz (torch.Tensor): [(n_lights,) 3], the xyz position of
                each light source
            - sh_coeff (torch.Tensor): [(n_lights,) 3, (sh_order + 1) ** 2],
                the SH coefficients for each light source
        """
        light_xyz, _ = super().get_actual_params(index=index)
        sh_coeff = self.params[..., 3:].view(self.n_lights, 3, -1)
        sh_coeff = torch.exp(sh_coeff.clamp(max=13))  # e^13 = 4.4e5
        if index is not None:
            sh_coeff = sh_coeff[index]
        return light_xyz, sh_coeff
    
    def get_intensity_and_relative_dir_dist(
            self,
            pts_xyz: torch.Tensor,
    ):
        # (n_pts, n_lights, 3 or 1]
        wi, dist, _ = super().get_intensity_and_relative_dir_dist(pts_xyz)
        # [n_lights, 3, (sh_order + 1) ** 2]
        _, sh_coeff = self.get_actual_params()
        # compute SH intensity
        intensity = sph_harm.eval_sh(
            deg=self.sh_order,
            dirs=-wi,
            sh_weights=sh_coeff.view(1, self.n_lights, 3, -1).expand(
                pts_xyz.shape[0], -1, -1, -1
            )
        )
        intensity = intensity.clamp(min=0)  # avoid negative radiance
        return wi, dist, intensity


def create_light(
        which: str,
        light_type: str,
        n_lights: int,
        device: torch.device,
        requires_grad: bool,
        **kwargs,
) -> Union[SGFarLight, SHNearLight, PixelFarLight]:
    """Create a far light.

    Args:
        which: can be 'far' or 'near'
        light_type: the type of the light
        n_lights: the number of light conditions
        device: the device to put the light on
        requires_grad: whether the light parameters require gradients
        **kwargs: additional arguments for the light

    Returns:
        Lighting: the lighting instance
    """
    assert which in ['far', 'near'], \
        f'Unrecognized light type: {which}'
    if which == 'far':
        if light_type == 'sg':
            return SGFarLight(
                n_lights=n_lights,
                device=device,
                requires_grad=requires_grad,
                n_lobes=kwargs['n_lobes'],
                reparam_lobes=True,
            )
        elif light_type == 'pixel':
            return PixelFarLight(
                n_lights=n_lights,
                device=device,
                requires_grad=requires_grad,
                envmap_h=kwargs['envmap_h'],
                envmap_w=kwargs['envmap_w'],
            )
        else:
            raise ValueError(f'Unrecognized far light kind: {light_type}')
    else:  # near light
        if light_type == 'sh_point':
            return SHNearLight(
                n_lights=n_lights,
                device=device,
                requires_grad=requires_grad,
                pos_types=kwargs['pos_types'],
                sh_order=kwargs['sh_order'],
            )
        else:
            raise ValueError(f'Unrecognized near light kind: {light_type}')


def bake_to_envmap(
        far_light: FarLight,
        pixel_far_light: PixelFarLight,
        detach: bool = True,
):
    """Convert :attr:`far_light` to an environment map and store it in
    :attr:`pixel_far_light`.

    Args:
        far_light: the far light to be baked
        pixel_far_light: the pixel far light to store the baked environment map
        detach: whether to detach the environment map from the graph
    """
    with torch.set_grad_enabled(not detach):
        envmap = far_light.to_envmap(pixel_far_light.light_dirs)
        if detach:
            envmap = envmap.detach()
        pixel_far_light.override_actual_params(
            envmap, mutable=False
        )
