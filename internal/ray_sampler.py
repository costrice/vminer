"""
Sample incident ray directions from a surface point for MC Integration.
References:
    Mitsuba 3: https://github.com/mitsuba-renderer/mitsuba3/blob/stable/include/mitsuba/core/frame.h
    MILO-Renderer: https://codeberg.org/ybh1998/LibreDR/src/branch/main/worker/src/render/render_cl/jit_kernel_reflection.rs
    NeFII: https://github.com/FuxiComputerVision/Nefii/blob/main/code/model/path_tracing_render.py
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from internal import geo_utils

ENVMAP_IMPORTANCE_SAMPLE_SHAPE_LIMIT = (64, 128)


def sample_interval(
        alpha: torch.Tensor,
        n_samples: int,
) -> torch.Tensor:
    """Given weight for each interval, sample the index for each sample.

    Args:
        alpha: (n_pts, 1, n_intervals) weight for each interval. Should sum to 1
            unless all weights are zero.
        n_samples: the number of samples to be drawn for each point

    Returns:
        torch.Tensor: (n_pts, n_samples, 1), the interval index for each sample.
    """
    n_pts, _, n_intervals = alpha.shape
    device = alpha.device
    alpha_cumsum_right = torch.cumsum(alpha, dim=-1)
    alpha_cumsum_right = torch.cat(
        [torch.zeros_like(alpha_cumsum_right[..., :1]),
         alpha_cumsum_right], dim=-1
    )
    alpha_cumsum_right[..., -1] = 1.0  # numerical stable
    r0 = torch.rand(n_pts, n_samples, 1, device=device)
    # binary search
    k = torch.searchsorted(
        alpha_cumsum_right,
        r0.permute(0, 2, 1),
        right=True
    ).permute(0, 2, 1) - 1
    
    return k


class PdfCalc(object):
    """
    Base class for pdf calculation of incident light directions under different
    sampling methods.
    """
    
    def __init__(self):
        pass
    
    @torch.no_grad()
    def __call__(
            self,
            wi: torch.Tensor,
            normal: Optional[torch.Tensor],
            view_dirs: Optional[torch.Tensor],
            roughness: Optional[torch.Tensor],
            light_param: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the pdf of the incident direction under a sampling method.

        Args:
            wi: (n_pts, n_dirs, 3), the incident direction
            normal: (n_pts, 3), the surface normal
            view_dirs: (n_pts, 3), the view direction
            roughness: (n_pts, 1), the roughness of GGX BRDF
            light_param: (*param_shape), the light source parameters

        Returns:
            torch.Tensor: (n_pts, n_dirs, 1), the pdf of the incident direction
        """
        raise NotImplementedError


class PdfMixedSG(PdfCalc):
    """
    Compute the pdf of the incident direction under mixed spherical Gaussian
    light source sampling.
    """
    
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def __call__(
            self,
            wi: torch.Tensor,
            normal: torch.Tensor,
            view_dirs: Optional[torch.Tensor],
            roughness: Optional[torch.Tensor],
            light_param: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the pdf of the incident direction under spherical
        Gaussian light source mixed importance sampling.

        Args:
            wi: (n_pts, n_dirs, 3), the incident direction
            normal: (n_pts, 3), the surface normal
            view_dirs: None
            roughness: None
            light_param: [n_lobes, 7], the SG lights used. Contains [xyz, lambda,
                RGB] for each lobe.
        """
        n_pts = wi.shape[0]
        n_dirs = wi.shape[1]
        n_lobes = light_param.shape[-2]
        normal = normal.view(n_pts, 1, 1, 3)
        wi = wi.view(n_pts, n_dirs, 1, 3)
        light_param = light_param.view(1, 1, n_lobes, 7)
        
        lobe = F.normalize(light_param[..., :3], p=2, dim=-1, eps=1e-6)
        sharpness = light_param[..., 3:4]
        intensity = light_param[..., 4:]
        
        energy = torch.sum(intensity, dim=-1, keepdim=True)
        normal_lobe_dots = torch.sum(
            lobe * normal, dim=-1,
            keepdim=True
            )  # (n_pts, 1, n_lobes, 1)
        # for offset, see comments in importance_sample_sg_light()
        sharpness_thres = 3
        sharp_mask = (sharpness > sharpness_thres).float()
        offset = torch.sqrt(
            torch.clamp(
                sharpness_thres * 2 / sharpness
                - sharpness_thres ** 2 / sharpness ** 2,
                min=1e-6
            )
        ) * sharp_mask + 1 * (1 - sharp_mask)
        # offset = 0  # TODO: remove this
        weight = (energy *
                  (normal_lobe_dots + offset).clamp(min=1e-6, max=1))
        alpha = weight / torch.sum(weight, dim=-2, keepdim=True)
        
        # compute normalization factor
        c = sharpness / (2 * torch.pi * (1 - torch.exp(-2 * sharpness)))
        
        # compute pdf
        dots = torch.sum(wi * lobe, dim=-1, keepdim=True)
        pdf_wi = alpha * c * torch.exp(sharpness * (dots - 1))
        pdf_wi = torch.sum(pdf_wi, dim=-2)  # (n_pts, n_dirs, 1)
        
        del n_pts, n_dirs, n_lobes, lobe, sharpness, intensity, energy, \
            normal_lobe_dots, sharp_mask, offset, weight, alpha, c, dots
        
        return pdf_wi


class PdfPixelLight(PdfCalc):
    """
    Compute the pdf of the incident direction under pixel light source sampling.
    """
    
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def __call__(
            self,
            wi: torch.Tensor,
            normal: torch.Tensor,
            view_dirs: Optional[torch.Tensor],
            roughness: Optional[torch.Tensor],
            light_param: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the pdf of the incident direction under pixel light source
        importance sampling.

        Args:
            wi: (n_pts, n_dirs, 3), the incident direction
            normal: (n_pts, 3), the surface normal
            view_dirs: None
            roughness: None
            light_param: [h, w, 6], the pixel lights used. Each pixel contains
                [xyz, RGB].
        """
        # expand shape to (n_pts, n_dirs, n_pixels, ...]
        device = wi.device
        pixel_lights = light_param
        n_pts = wi.shape[0]
        n_dirs = wi.shape[1]
        h, w, _ = light_param.shape
        # if too large, downsample for efficiency
        shape_limit = ENVMAP_IMPORTANCE_SAMPLE_SHAPE_LIMIT
        if h > shape_limit[0] or w > shape_limit[1]:
            pixel_lights = pixel_lights.view(1, h, w, 6)
            pixel_lights = F.interpolate(
                pixel_lights.permute(0, 3, 1, 2),
                size=shape_limit, mode='area'
            )
            pixel_lights = pixel_lights.permute(0, 2, 3, 1)
            h, w = shape_limit
        
        pixel_lights = pixel_lights.view(1, 1, h * w, 6)
        normal = normal.view(n_pts, 1, 1, 3)
        wi = wi.view(n_pts, n_dirs, 1, 3)
        
        # unpack pixel lights
        # (n_pts, n_lights, 3)
        light_dir = F.normalize(pixel_lights[..., :3], p=2, dim=-1, eps=1e-6)
        intensity = pixel_lights[..., 3:6]
        light_coord = geo_utils.view_vec2image_coord(light_dir)
        
        energy = torch.sum(intensity, dim=-1, keepdim=True)
        normal_dir_dot = torch.sum(
            light_dir * normal,
            dim=-1, keepdim=True
        ).clamp(min=1e-9)
        # account for environment map warping
        theta = light_coord[..., 0:1] * np.pi / 2  # in [-pi/2, pi/2]
        cos_theta = torch.cos(theta)  # area of envmap pixels
        weight = energy * normal_dir_dot * cos_theta
        alpha = weight / torch.sum(
            weight, dim=-2, keepdim=True
        ).clamp(min=1e-6)
        alpha = alpha.view(n_pts, 1, h * w)
        
        # convert direction to pixel coordinates
        size_tensor = torch.tensor([h, w], device=device)
        # (n_pts, n_dirs, 1, 2], in [-1, 1], step size 2 / h
        wi_coord = geo_utils.view_vec2image_coord(wi)
        wi_theta = wi_coord[..., 0] * np.pi / 2  # in [-pi/2, pi/2]
        # in [0, h), step size 1
        wi_coord = ((wi_coord + 1) / 2).clamp(min=0, max=1 - 1e-6) * size_tensor
        wi_coord = wi_coord.long()
        # (n_pts, n_dirs, 1, 1], flatten in [0, h * w)
        wi_coord_flat = wi_coord[..., 0] * w + wi_coord[..., 1]
        # read pdf from alpha
        wi_alpha = torch.gather(
            alpha.expand(-1, n_dirs, -1),
            dim=-1, index=wi_coord_flat
        )
        # divide area size so that pdf integrates to 1 on the hemisphere
        wi_cos_theta = torch.cos(wi_theta)
        # don't know why this constant
        wi_area_size = 1 / w * wi_cos_theta / h * 2 * torch.pi * torch.pi
        pdf_wi = wi_alpha / wi_area_size
        pdf_wi = pdf_wi.view(n_pts, n_dirs, 1)
        
        return pdf_wi


class PdfGGXBRDF(PdfCalc):
    """
    Compute the pdf of the incident direction under GGX BRDF importance
    sampling.
    """
    
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def __call__(
            self,
            wi: torch.Tensor,
            normal: torch.Tensor,
            view_dirs: torch.Tensor,
            roughness: torch.Tensor,
            light_param: Optional[torch.Tensor],
    ) -> torch.Tensor:
        view_dirs = view_dirs.view(-1, 1, 3)
        normal = normal.view(-1, 1, 3)
        roughness = roughness.view(-1, 1, 1)
        # compute half vector
        h = F.normalize(wi + view_dirs, p=2, dim=-1, eps=1e-6)
        # if wi = - viewdir, then their half vector should be normal or beyond
        # semi-sphere, which would be rendered as zero afterward
        mask = torch.isnan(h)
        h[mask] = normal.expand_as(h)[mask]
        cos_theta_h = torch.sum(normal * h, dim=-1, keepdim=True)
        cos_theta_h = cos_theta_h.clamp(min=1e-6)
        root = cos_theta_h ** 2 + (1 - cos_theta_h ** 2) / (roughness ** 4)
        pdf_h = cos_theta_h / (torch.pi * (roughness ** 4) * root * root)
        h_dot_v = torch.sum(h * view_dirs, dim=-1, keepdim=True)
        h_dot_v = h_dot_v.clamp(min=1e-6)
        pdf_wi = pdf_h / (4 * h_dot_v)
        
        del h, mask, cos_theta_h, root, pdf_h, h_dot_v
        
        return pdf_wi


class PdfCos(PdfCalc):
    """
    Compute the pdf of the incident direction under cosine importance sampling.
    """
    
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def __call__(
            self,
            wi: torch.Tensor,
            normal: Optional[torch.Tensor],
            view_dirs: Optional[torch.Tensor],
            roughness: Optional[torch.Tensor],
            light_param: Optional[torch.Tensor],
    ) -> torch.Tensor:
        normal = normal.view(-1, 1, 3)
        cos_theta = torch.sum(normal * wi, dim=-1, keepdim=True)
        cos_theta = cos_theta.clamp(min=1e-6)
        pdf = cos_theta / torch.pi
        
        del cos_theta
        
        return pdf


class PdfUniform(PdfCalc):
    """
    Compute the pdf of the incident direction under uniform sampling.
    """
    
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def __call__(
            self,
            wi: torch.Tensor,
            normal: Optional[torch.Tensor],
            view_dirs: Optional[torch.Tensor],
            roughness: Optional[torch.Tensor],
            light_param: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # integrate to 1 over the hemisphere
        return torch.ones_like(wi[..., :1]) / (2 * np.pi)


class RaySampler(object):
    """
    Sample rays from a surface.
    """
    
    def __init__(
            self,
            normals: torch.Tensor,
            device: torch.device,
    ):
        """Sample incident light directions for points on the surface.

        Args:
            normals: (n_pts, 3), the surface normals
            device: the device to operate on
        """
        self.frame = geo_utils.Frame(normals, device)
        self.n_pts = normals.shape[0]
        self.device = device
    
    @torch.no_grad()
    def uniform_sample(
            self,
            n_dirs: int,
            upper: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, PdfCalc]:
        """Sample light directions uniformly sample on a sphere surface.

        References:
            https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere

        Args:
            n_dirs: the number of light directions to be sampled
            upper: whether to sample on the upper hemisphere in local frame

        Returns:
            A tuple containing

            - vec (torch.Tensor): (n_pts, n_dirs, 3), the sampled directions
                pointing outward the surface
            - pdfs (torch.Tensor): (n_pts, n_dirs, 1), the probability density
                functions of the sampled light directions. The pdfs integrate
                to 1 over the hemisphere.
            - pdf_fn (PdfCalc): a function that takes the incident direction
                (and other parameters) and returns the pdf of the incident
                direction
        """
        n_dim = 3
        vec_local = torch.randn(
            self.n_pts, n_dirs, n_dim,
            device=self.device, dtype=torch.float
        )
        vec_local = F.normalize(vec_local, p=2, dim=-1, eps=1e-6)
        if upper:
            vec_local[..., 2] = torch.abs(vec_local[..., 2])
        
        pdf_fn = PdfUniform()
        pdf = pdf_fn(
            vec_local, self.frame.n,
            None, None, None
        )
        
        vec = self.frame.to_world(vec_local)
        
        del vec_local
        
        return vec, pdf, pdf_fn
    
    @torch.no_grad()
    def cosine_sample(
            self,
            n_dirs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, PdfCalc]:
        """Cosine importance sample light directions.

        Args:
            n_dirs: the number of light directions to be sampled

        Returns:
            A tuple containing

            - vec (torch.Tensor): (n_pts, n_dirs, 3), the sampled directions
                pointing outward the surface
            - pdfs (torch.Tensor): (n_pts, n_dirs, 1), the probability density
                functions of the sampled light directions. The pdfs integrate
                to 1 over the hemisphere.
            - pdf_fn (PdfCalc): a function that takes the incident direction
                (and other parameters) and returns the pdf of the incident
                direction
        """
        # sampling h in local frame
        r1 = torch.rand(self.n_pts, n_dirs, 1, device=self.device)
        r2 = torch.rand(self.n_pts, n_dirs, 1, device=self.device)
        theta = torch.arccos(torch.sqrt(1 - r1))
        phi = 2 * torch.pi * r2
        z = theta.cos()
        y = theta.sin() * phi.sin()
        x = theta.sin() * phi.cos()
        wi = torch.cat([x, y, z], dim=-1)
        # rotate to normal
        wi = self.frame.to_world(wi)
        # calculate pdf
        pdf = z / torch.pi
        
        del r1, r2, theta, phi, z, y, x
        
        return wi, pdf, PdfCos()
    
    @torch.no_grad()
    def importance_sample_sg_light(
            self,
            n_dirs: int,
            sg_lights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, PdfCalc]:
        """Importance sample light directions for SG lights.

        Mix gaussian sampling:
        first utilize the probability ak to decide which Gaussian component to
        draw from, then draw wi from the k-th Gaussian distribution.
        pdf(w_i) = sum_{k=1}^M alpha_k c_k exp{sharpness_k(w_i cdot xi_k - 1)}
        alpha_k = frac{mu_k}{sum_{j=1}^M mu_j}
        1. sample based on alpha to decide use which single gaussian component
        2. sample w_i using the single gaussian component

        References:
            https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
            https://github.com/FuxiComputerVision/Nefii/blob/main/code/model/path_tracing_render.py

        Args:
            n_dirs: the number of light directions to be sampled
            sg_lights: [n_lobes, 7], the SG light sources used for
                each point for importance sampling. Each SG light source
                contains [xyz, lambda, RGB] for each lobe.

        Returns:
            A tuple containing

            - vec (torch.Tensor): (n_pts, n_dirs, 3), the sampled directions
                pointing outward the surface
            - pdfs (torch.Tensor): (n_pts, n_dirs, 1), the probability density
                functions of the sampled light directions. The pdfs integrate
                to 1 over the hemisphere.
            - pdf_fn (PdfCalc): a function that takes the incident direction
                (and other parameters) and returns the pdf of the incident
                direction
        """
        n_lobes = sg_lights.shape[-2]
        sg_lights = sg_lights.view(1, n_lobes, 7)
        
        # unpack sg light sources
        # [1, n_lobes, 3 or 1)
        lobe = sg_lights[..., :3]
        sharpness = sg_lights[..., 3:4]
        intensity = sg_lights[..., 4:]
        
        # compute \alpha_k = \frac{\mu_k}{\sum_{j=1}^M \mu_j}
        energy = torch.sum(intensity, dim=-1, keepdim=True)
        normal_lobe_dots = torch.sum(
            lobe * self.frame.n, dim=-1,
            keepdim=True
            )
        # because sg lobe is not a single point but a distribution, even
        # when normal_lobe_dots < 0, there can still be light coming from
        # the lobe. Therefore, we add a positive value according to the
        # sharpness of the lobe (sharper lobe, smaller value) to make sure
        # rough lobes are not completely ignored.
        # the value is decided by:
        #     e^k(mu*w - 1) < 0.05
        #        => k(mu*w - 1) < about -3
        sharpness_thres = 3
        sharp_mask = (sharpness > sharpness_thres).float()
        offset = torch.sqrt(
            torch.clamp(
                sharpness_thres * 2 / sharpness
                - sharpness_thres ** 2 / sharpness ** 2,
                min=1e-6
            )
        ) * sharp_mask + 1 * (1 - sharp_mask)
        # offset = 0  # TODO: remove this
        weight = (energy *
                  (normal_lobe_dots + offset).clamp(min=1e-6, max=1))
        alpha = weight / torch.sum(
            weight, dim=-2, keepdim=True
        ).clamp(min=1e-6)
        alpha = alpha.view(self.n_pts, 1, n_lobes)
        
        # sample a lobe index for each point
        k = sample_interval(alpha, n_dirs)
        
        # (n_pts, n_dirs, 3 or 1)
        lobe_k = torch.gather(
            lobe.expand(self.n_pts, -1, -1),
            dim=-2, index=k.expand(-1, -1, 3)
        )
        sharpness_k = torch.gather(
            sharpness.expand(self.n_pts, -1, -1),
            dim=-2, index=k
        )
        intensity_k = torch.gather(
            intensity.expand(self.n_pts, -1, -1),
            dim=-2, index=k.expand(-1, -1, 3)
        )
        
        # normalization factor
        c_k = sharpness_k / (2 * torch.pi *
                             (1 - torch.exp(-2 * sharpness_k)))
        
        # sample direction w.r.t. the k-th lobe frame
        r1 = torch.rand(self.n_pts, n_dirs, 1, device=self.device)
        r2 = torch.rand(self.n_pts, n_dirs, 1, device=self.device)
        theta = torch.arccos(
            1.0 / sharpness_k * torch.log(
                torch.clamp(  # for numeric stable
                    1 - sharpness_k * r1 / (2 * torch.pi * c_k),
                    1e-6
                )
            ) + 1
        )
        phi = 2 * torch.pi * r2
        z = theta.cos()
        y = theta.sin() * phi.sin()
        x = theta.sin() * phi.cos()
        wi = torch.stack([x, y, z], dim=-1)
        # rotate to world
        wi = (geo_utils.Frame(lobe_k.view(-1, 3))
              .to_world(wi.view(-1, 1, 3)))
        wi = wi.view(self.n_pts, n_dirs, 3)
        
        pdf_fn = PdfMixedSG()
        
        pdf_wi = pdf_fn(
            wi, self.frame.n,
            None, None,
            sg_lights
        )
        
        del lobe_k, sharpness_k, intensity_k, c_k, r1, r2, theta, phi, z, y, \
            x, alpha, weight, normal_lobe_dots, offset, \
            sharp_mask, energy, intensity, k, lobe, sharpness
        
        return wi, pdf_wi, pdf_fn
    
    @torch.no_grad()
    def importance_sample_pixel_light(
            self,
            n_dirs: int,
            pixel_lights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, PdfCalc]:
        """Importance sample light directions for pixel (or envmap) lights.

        Args:
            n_dirs (int): the number of light directions to be sampled
            pixel_lights (torch.Tensor): [h, w, 6], the pixel lights used.
                Each pixel contains [xyz, RGB].

        Returns:
            A tuple containing

            - vec (torch.Tensor): (n_pts, n_dirs, 3], the sampled directions
                pointing outward the surface
            - pdfs (torch.Tensor): (n_pts, n_dirs, 1), the probability density
                functions of the sampled light directions. The pdfs integrate
                to 1 over the hemisphere.
            - pdf_fn (PdfCalc): a function that takes the incident direction
                (and other parameters) and returns the pdf of the incident
                direction
        """
        h, w, _ = pixel_lights.shape
        # if too large, downsample for efficiency
        shape_limit = ENVMAP_IMPORTANCE_SAMPLE_SHAPE_LIMIT
        if h > shape_limit[0] or w > shape_limit[1]:
            pixel_lights = pixel_lights.view(1, h, w, 6)
            pixel_lights = F.interpolate(
                pixel_lights.permute(0, 3, 1, 2),
                size=shape_limit, mode='area'
            )
            pixel_lights = pixel_lights.permute(0, 2, 3, 1)
            h, w = shape_limit
        pixel_lights = pixel_lights.view(1, h * w, 6)
        
        # unpack pixel lights
        # [1, h * w, 3]
        light_dir = F.normalize(pixel_lights[..., :3], p=2, dim=-1, eps=1e-6)
        intensity = pixel_lights[..., 3:6]
        light_coord = geo_utils.view_vec2image_coord(light_dir)
        
        # dot product between light direction and normal
        # [1, h * w, 1)
        energy = torch.sum(intensity, dim=-1, keepdim=True)
        normal_dir_dot = torch.sum(
            light_dir * self.frame.n.view(self.n_pts, 1, 3),
            dim=-1, keepdim=True
        ).clamp(min=1e-9)
        # account for environment map warping
        theta = light_coord[..., 0:1] * np.pi / 2  # in [-pi/2, pi/2]
        cos_theta = torch.cos(theta)  # area
        weight = energy * normal_dir_dot * cos_theta
        alpha = weight / torch.sum(
            weight, dim=-2, keepdim=True
        ).clamp(min=1e-6)
        alpha = alpha.view(self.n_pts, 1, h * w)
        
        # sample a pixel index for each point
        k = sample_interval(alpha, n_dirs)
        
        # sample a direction within each pixel
        size_tensor = torch.tensor([h, w], device=self.device)
        # in [-1 + 1/h, 1 - 1/h], step size 2/h
        wi_coord_center = torch.gather(
            light_coord.expand(self.n_pts, -1, -1),
            dim=-2,
            index=k.expand(-1, -1, 2)
        )
        # in [1/2, h - 1/2], step size 1
        wi_coord_center = (wi_coord_center + 1) / 2 * size_tensor
        r1 = torch.rand(self.n_pts, n_dirs, 1, device=self.device)
        r2 = torch.rand(self.n_pts, n_dirs, 1, device=self.device)
        # add -1/2~1/2 to the center coordinate -> in [0, h], step size 1
        wi_coord = (wi_coord_center +
                    torch.cat([r1 * 0.99, r2 * 0.99], dim=-1) / 2)
        # in [-1, 1], step size 2/h
        wi_coord = wi_coord / size_tensor * 2 - 1
        wi_coord = wi_coord.clamp(min=-1, max=1)
        wi = geo_utils.image_coord2view_vec(wi_coord)
        pdf_fn = PdfPixelLight()
        pdf_wi = pdf_fn(
            wi, self.frame.n,
            None, None,
            pixel_lights.view(h, w, 6)
        )
        
        del energy, normal_dir_dot, cos_theta, weight, alpha, \
            k, light_dir, intensity
        
        return wi, pdf_wi, pdf_fn
    
    @torch.no_grad()
    def importance_sample_ggx_brdf(
            self,
            n_dirs: int,
            roughness: torch.Tensor,
            view_dirs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, PdfCalc]:
        """Importance sample light directions for GGX specular BRDF.

        Args:
            n_dirs: the number of light directions to be sampled
            roughness: (n_pts, 1), the roughness of the surface
            view_dirs: (n_pts, 3], the view directions

        Returns:
            A tuple containing

            - vec (torch.Tensor): (n_pts, n_dirs, 3], the sampled directions
                pointing outward the surface
            - pdfs (torch.Tensor): (n_pts, n_dirs, 1], the probability density
                functions of the sampled light directions. The pdfs integrate
                to 1 over the hemisphere.
            - pdf_fn (PdfCalc): a function that takes the incident direction
                (and other parameters) and returns the pdf of the incident
                direction
        """
        # sampling h in local frame
        r1 = torch.rand(self.n_pts, n_dirs, 1, device=self.device)
        r2 = torch.rand(self.n_pts, n_dirs, 1, device=self.device)
        roughness = roughness.view(self.n_pts, 1, 1)
        theta = torch.arctan(
            roughness ** 2 * torch.sqrt(r1 / (1 - r1))
        )
        phi = 2 * torch.pi * r2
        z = theta.cos()
        y = theta.sin() * phi.sin()
        x = theta.sin() * phi.cos()
        h = torch.cat([x, y, z], dim=-1)
        # to world frame
        h = self.frame.to_world(h)  # (n_pts, n_dirs, 3]
        
        # reflect view direction
        v = view_dirs.view(self.n_pts, 1, 3)
        wi = 2 * torch.sum(h * v, dim=-1, keepdim=True) * h - v
        wi = F.normalize(wi, p=2, dim=-1, eps=1e-6)
        
        # calculate pdf
        pdf_fn = PdfGGXBRDF()
        pdf_wi = pdf_fn(
            wi, self.frame.n,
            view_dirs, roughness, None
        )
        
        del r1, r2, roughness, theta, phi, z, y, x, h, v
        
        return wi, pdf_wi, pdf_fn


@torch.no_grad()
def power_heuristic_list(
        used_types: List[str],
        this_type: str,
        pdf_dict: Dict[str, torch.Tensor],
):
    """Compute MIS weight using power heuristic with exponent 2.

    Args:
        used_types: the types of sampling methods used
        this_type: the type of sampling method to compute the MIS weight for
        pdf_dict: the pdfs of the incident directions sampled by this type of
            sampling method, under all different types of sampling methods

    Returns:
        torch.Tensor: (n_pts, n_dirs, 1), the MIS weight
    """
    n_all_dirs = 0
    for sample_type in used_types:
        n_all_dirs += pdf_dict[sample_type].shape[1]
    
    cur = ((pdf_dict[this_type].shape[1] / n_all_dirs * pdf_dict[this_type])
           ** 2)
    all_sum = torch.zeros_like(cur)
    for sample_type in used_types:
        all_sum += ((pdf_dict[sample_type].shape[1] / n_all_dirs
                     * pdf_dict[sample_type])
                    ** 2)
    all_sum = torch.clamp(all_sum, min=1e-6)
    result = cur / all_sum
    
    del cur, all_sum
    
    return result
