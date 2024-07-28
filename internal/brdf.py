"""
This file implements the bidirectional reflectance distribution function (BRDF) models.
"""
from typing import Dict, Tuple

import configargparse
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

EPS = 1e-6
EPS_2 = 1e-12


class BaseBRDF(nn.Module):
    """Base class for BRDF models."""
    
    def __init__(self):
        super().__init__()
    
    @property
    def material_dims(self) -> Dict[str, int]:
        """Return the keys and dimensionality of the material parameters."""
        raise NotImplementedError
    
    @property
    def packed_material_dim(self):
        """The needed dimensionality of the output of material network to
        unpack."""
        raise NotImplementedError
    
    def unpack_brdf(
            self,
            packed_brdf: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Unpack the network output code to the material parameters.

        Args:
            packed_brdf: (n_pts, self.material_code_dim) the output of the
                material network

        Returns:
            A dictionary containing the material parameters, with keys and dims
                specified by self.material_dims
        """
        raise NotImplementedError
    
    def forward(
            self,
            normal: torch.Tensor,
            view_dir_out: torch.Tensor,
            light_dir_out: torch.Tensor,
            materials: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the BRDF given the surface normal, the view direction, the
        light direction, and the material parameters.

        Args:
            normal: (n_pts, 3), the surface normal
            view_dir_out: (n_pts, 3), the view direction
            light_dir_out: (n_pts, 3), the light direction
            materials: a dictionary containing the material parameters. The
                keys and dims (the shape is (n_pts, ?)) are specified by
                self.material_dims.

        Returns:
            A tuple of torch.Tensor containing:

            - diffuse (torch.Tensor): (n_pts, 3), the diffuse part of BRDF
            - specular (torch.Tensor): (n_pts, 3), the specular part of BRDF
        """
        raise NotImplementedError
    
    def compute_reflectance(
            self,
            normal: torch.Tensor,
            light_dirs_out: torch.Tensor,
            view_dirs_out: torch.Tensor,
            materials: Dict[str, torch.Tensor],
    ):
        """Compute specular reflectance, cos term, and cos mask.

        Helper function to compute the reflectance of the surface points given the
        surface normal, the view directions, the light directions, and BRDF
        parameters. The reflectance is computed as the sum of the specular term and
        the diffuse term.

        Args:
            normal: (n_pts, 3), the unit surface normal pointing outward the
                surface
            light_dirs_out: (n_pts, n_dirs, 3), the unit direction from the points
                to the light source
            view_dirs_out: (n_pts, 3), the unit direction from the points to the
                viewer
            materials: a dictionary containing the material parameters, whose keys
                are decided by the BRDF model used.

        Returns:
            A tuple containing the following torch.Tensor:

            - diffuse_reflectance (torch.Tensor): (n_pts, n_dirs, 3), the diffuse
                reflectance of the surface points
            - spec_reflectance (torch.Tensor): (n_pts, n_dirs, 3), the specular
                reflectance of the surface points
            - cos_term (torch.Tensor): (n_pts, n_dirs, 1), the cosine of the angle
                between the surface normal and the light direction
            - mask (torch.Tensor): (n_pts, n_dirs), the mask indicating whether the
                cosine term is > 0
        """
        n_rays, n_dirs = light_dirs_out.shape[:2]
        
        # compute cosine term between the surface normal and the light direction
        cos_term = torch.sum(
            light_dirs_out * normal.view(n_rays, 1, 3),
            dim=-1, keepdim=True
        )
        cos_term.clamp_(0, 1)
        # # if normal is back facing the viewer, the cosine term is set to 0
        # normal_view_dot = torch.sum(
        #     normal * view_dirs,
        #     dim=-1, keepdim=True)
        # cos_term = (cos_term *
        #             ((F.softsign(normal_view_dot * 100) + 1) / 2
        #              ).view(n_rays, 1, 1))
        # (n_rays, n_dirs)
        cos_mask = (cos_term > 1e-6).view(n_rays, n_dirs)
        
        # compute BRDF using the surface normal, the view direction, the light
        # direction, and BRDF parameters
        brdf_term_diffuse, brdf_term_specular = self(
            normal.repeat_interleave(n_dirs, dim=0),
            view_dirs_out.repeat_interleave(n_dirs, dim=0),
            light_dirs_out.view(-1, 3),
            materials={k: v.repeat_interleave(n_dirs, dim=0) for k, v in
                       materials.items()}
        )
        brdf_term_diffuse = brdf_term_diffuse.view(n_rays, n_dirs, 3)
        brdf_term_specular = brdf_term_specular.view(n_rays, n_dirs, 3)
        
        return brdf_term_diffuse, brdf_term_specular, cos_term, cos_mask


class GGXBRDF(BaseBRDF):
    """
    Simplified GGX BRDF, having 3 parameters:
    - base_color (or diffuse albedo)
    - roughness
    - fresnel (or specular intensity)
    
    The implementation is borrowed from TensoIR:
    https://github.com/Haian-Jin/TensoIR/blob/main/models/relight_utils.py
    """
    
    def __init__(self):
        super().__init__()
    
    @property
    def material_dims(self) -> Dict[str, int]:
        return {
            'base_color': 3,
            'roughness': 1,
            'fresnel': 1,
        }
    
    @property
    def packed_material_dim(self):
        return 4  # albedo 3 + roughness 1
    
    def unpack_brdf(
            self,
            packed_brdf: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # albedo interval: [0., 1.]
        # roughness interval: [0.09, 0.99]
        return {
            'base_color': torch.sigmoid(packed_brdf[..., :3] - 2.),
            'roughness': (0.9 * torch.sigmoid(packed_brdf[..., 3:4]) +
                          0.09),
            'fresnel': torch.ones_like(packed_brdf[..., 3:4]) * 0.04,
        }
    
    def forward(
            self,
            normal: torch.Tensor,
            view_dir_out: torch.Tensor,
            light_dir_out: torch.Tensor,
            materials: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        albedo = materials['base_color']
        roughness = materials['roughness']
        fresnel = materials['fresnel']
        
        # compute specular term
        L = F.normalize(light_dir_out, dim=-1, eps=EPS)  # (n_pts, 3)
        
        V = F.normalize(view_dir_out, dim=-1, eps=EPS)  # (n_pts, 3)
        H = F.normalize((L + V) / 2.0, dim=-1)  # (n_pts, 3)
        N = F.normalize(normal, dim=-1, eps=EPS)  # (n_pts, 3)
        
        NoV = torch.sum(V * N, dim=-1, keepdim=True)  # (n_pts, 1)
        N = N * NoV.sign()  # (n_pts, 3)
        
        # (n_pts, 1)
        NoL = torch.sum(N * L, dim=-1, keepdim=True).clamp_(EPS, 1)
        NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(EPS, 1)
        NoH = torch.sum(N * H, dim=-1, keepdim=True).clamp_(EPS, 1)
        VoH = torch.sum(V * H, dim=-1, keepdim=True).clamp_(EPS, 1)
        
        # (n_pts, 1)
        alpha = roughness * roughness
        alpha2 = alpha * alpha
        k = (alpha + 2 * roughness + 1.0) / 8.0
        FMi = ((-5.55473) * VoH - 6.98316) * VoH
        frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)
        frac = frac0 * alpha2
        nom0 = NoH * NoH * (alpha2 - 1) + 1
        nom1 = NoV * (1 - k) + k
        nom2 = NoL * (1 - k) + k
        nom = (4 * np.pi * nom0 * nom0 * nom1 * nom2).clamp_(EPS, 4 * np.pi)
        spec = frac / nom
        
        spec = spec.repeat(1, 3)  # to RGB
        
        del L, V, H, N, NoV, NoL, NoH, VoH, alpha, alpha2, k, FMi, frac0, \
            frac, nom0, nom1, nom2, nom
        
        # compute diffuse term
        diff = albedo / np.pi
        
        return diff, spec


class DisneyPrincipledBRDF(BaseBRDF):
    """
    Simplified Disney principled BRDF, having 4 parameters:
    - base_color (diffuse albedo)
    # - metallic
    - roughness
    - specular (specular intensity)
    """
    
    def __init__(self):
        super().__init__()
    
    @property
    def material_dims(self) -> Dict[str, int]:
        return {
            'base_color': 3,
            # 'metallic': 1,
            'roughness': 1,
            'specular': 1,
        }
    
    @property
    def packed_material_dim(self):
        return 5  # albedo 3 + roughness 1 + specular 1
    
    def unpack_brdf(
            self,
            packed_brdf: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # base_color interval: [0., 1.]
        # roughness interval: [0.01, 1]
        # # metallic interval: [0., 1.] (0: dielectric, 1: metal)
        # specular interval: [0., 1.]
        ret_dict = {
            'base_color': torch.sigmoid(packed_brdf[..., :3] - 2.),
            'roughness': (0.99 * torch.sigmoid(packed_brdf[..., 3:4]) + 0.01),
            'specular': torch.sigmoid(packed_brdf[..., 4:5])
            # 'metallic': torch.sigmoid(brdf_code[..., 5:6] - 2.),
        }
        return ret_dict
    
    def forward(
            self,
            normal: torch.Tensor,
            view_dir_out: torch.Tensor,
            light_dir_out: torch.Tensor,
            materials: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_color = materials['base_color']
        # metallic = materials['metallic']
        roughness = materials['roughness']
        specular = materials['specular']
        # specular = torch.ones_like(roughness) * 0.5
        
        L = F.normalize(light_dir_out, dim=-1, eps=EPS)
        V = F.normalize(view_dir_out, dim=-1, eps=EPS)
        H = F.normalize(L + V, dim=-1, eps=EPS)  # half vector
        N = F.normalize(normal, dim=-1, eps=EPS)
        # h dot n = cos(theta_h)
        n_dot_h = torch.sum(H * N, dim=-1, keepdim=True).clamp_(EPS)
        n_dot_v = torch.sum(N * V, dim=-1, keepdim=True).clamp_(EPS)
        n_dot_l = torch.sum(N * L, dim=-1, keepdim=True).clamp_(EPS)
        # h dot l = h dot v = cos(theta_d)
        h_dot_l = torch.sum(H * L, dim=-1, keepdim=True).clamp_(EPS)
        
        # fresnel for diffuse
        f_d90 = 0.5 + 2 * roughness * (h_dot_l ** 2)
        f_d = ((1 + (f_d90 - 1) * schlick_fresnel(n_dot_l)) *
               (1 + (f_d90 - 1) * schlick_fresnel(n_dot_v)))
        # no subsurface scattering; metallic = 0
        diff = f_d * base_color / torch.pi
        # diff = diff * (1 - metallic)
        
        # the primary specular lobe in Disney BRDF
        alpha = 0.0001 + (roughness ** 2) * (1 - 0.0002)
        # distribution term
        d_s = gtr2(n_dot_h, alpha)
        # fresnel term
        f_0 = specular * 0.08
        # f_0 = (1 - metallic) * f_0 + metallic * base_color
        f_s = f_0 + (1 - f_0) * schlick_fresnel(h_dot_l)
        # g_metal = 2 / (torch.sqrt(1 + alpha_2 * (1 / cos_theta_h_2 - 1)) + 1)
        # geometry term
        alpha_g = (0.5 + roughness / 2) ** 2
        g_s = (smith_g_ggx(n_dot_l, alpha_g) *
               smith_g_ggx(n_dot_v, alpha_g))
        spec = d_s * f_s * g_s / (4 * n_dot_l * n_dot_v + EPS)
        spec = spec.expand(-1, 3)  # to RGB
        
        return diff, spec


def gtr2(
        n_dot_h: torch.Tensor,
        alpha: torch.Tensor
) -> torch.Tensor:
    """Generalized-Trowbridge-Reitz (GTR) distribution."""
    alpha_2 = alpha ** 2
    return alpha_2 / torch.pi / ((n_dot_h ** 2) * (alpha_2 - 1) + 1 + EPS) ** 2


def schlick_fresnel(dot: torch.Tensor) -> torch.Tensor:
    """Schlick Fresnel term interpolation factor.

    Args:
        dot: n_dot_l or n_dot_v

    Returns:
        torch.Tensor: a weight use for interpolation between F0 and 1
    """
    return torch.clamp(1.0 - dot, 0, 1) ** 5


def smith_g_ggx(dot: torch.Tensor, alpha_g: torch.Tensor) -> torch.Tensor:
    """Smith shadowing factor used in GGX geometry term.

    Args:
        dot: n_dot_l or n_dot_v
        alpha_g: alpha used for geometry term

    Returns:
        torch.Tensor: the geometry term
    """
    alpha_g_2 = alpha_g ** 2
    dot_2 = dot ** 2
    # safe sqrt
    root = torch.sqrt(
        torch.clamp(alpha_g_2 + dot_2 - alpha_g_2 * dot_2, EPS_2)
    )
    return 1.0 / (dot + root)


def get_brdf(
        brdf_type: str,
        args: configargparse.Namespace,
) -> BaseBRDF:
    """Get the BRDF model given the brdf_type.

    Args:
        brdf_type: the type of BRDF model, can be 'GGX' or 'Disney'
        args: the command line arguments

    Returns:
        BaseBRDF: the BRDF model
    """
    if brdf_type == 'GGX':
        return GGXBRDF()
    elif brdf_type == 'Disney':
        return DisneyPrincipledBRDF()
    else:
        raise ValueError(f'Unknown BRDF type: {brdf_type}')
