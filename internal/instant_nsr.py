"""
This file contains neural fields.
References: Instant-NSR (https://github.com/zhaofuq/Instant-NSR)
"""
import math
from typing import Dict, List, Tuple, Union

import configargparse
import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F

from internal import global_vars, lighting


def scale_anything(
        dat: torch.Tensor,
        inp_scale: Union[torch.Tensor, Tuple[float, float]],
        tgt_scale: Union[torch.Tensor, Tuple[float, float]],
):
    """Scale the input tensor to the target scale.

    Args:
        dat: (..., ?) the input tensor
        inp_scale: (2, ?) the source scale
        tgt_scale: (2, ?) the target scale

    Returns:
        torch.Tensor: the scaled tensor with the same shape as the input
    """
    # if inp_scale is None:
    #     inp_scale = [dat.min(), dat.max()]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def contract_to_unicube(
        xyz: torch.Tensor,
        aabb: torch.Tensor,
):
    """Contract the points to the unit cube (0, 1) along each dimension.

    Args:
        xyz: (..., 3) the input points
        aabb: (2, 3) the axis-aligned bounding box of the points

    Returns:
        torch.Tensor: the contracted points in (0, 1)^3
    """
    xyz = scale_anything(xyz, aabb, (0, 1))
    return xyz


class _TruncExp(torch.autograd.Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def get_activation(name: str):
    """Get the activation function by name.

    Args:
        name: the name of the activation function. Can be one of the following:
            - 'none': identity function
            - 'scaleX': scale the input by X
            - 'clampX': clamp the input to [0, X]
            - 'mulX': multiply the input by X
            - 'lin2srgb': convert the input from linear RGB to sRGB
            - 'trunc_exp': exponential function with truncation at 15 when back
                propagating
            - '+X': add X to the input
            - '-X': subtract X from the input
            - 'sigmoid': sigmoid function
            - 'tanh': tanh function
            - other activation functions in torch.nn.functional

    Returns:
        Callable: the activation function that takes a tensor and returns a
            tensor of the same shape.
    """
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == 'none':
        return lambda x: x
    if name.startswith('scale'):
        scale_factor = float(name[5:])
        return lambda x: x.clamp(0., scale_factor) / scale_factor
    if name.startswith('clamp'):
        clamp_max = float(name[5:])
        return lambda x: x.clamp(0., clamp_max)
    if name.startswith('mul'):
        mul_factor = float(name[3:])
        return lambda x: x * mul_factor
    if name == 'lin2srgb':
        return lambda x: torch.where(
            x > 0.0031308, torch.pow(
                torch.clamp(x, min=0.0031308), 1.0 / 2.4
            ) * 1.055 - 0.055,
            12.92 * x
        ).clamp(0., 1.)
    if name == 'trunc_exp':
        return trunc_exp
    if name.startswith('+') or name.startswith('-'):
        return lambda x: x + float(name)
    if name == 'sigmoid':
        return torch.sigmoid
    if name == 'tanh':
        return torch.tanh
    return getattr(F, name)


class VanillaMLP(nn.Module):
    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            n_hidden_layers: int,
            n_neurons: int,
            sphere_init: bool = False,
            sphere_init_radius: float = 1.0,
            weight_norm: bool = False,
            output_activation: str = 'none',
    ):
        """Initialize a vanilla MLP.

        Args:
            dim_in: the dimension of the input tensor
            dim_out: the dimension of the output tensor
            n_hidden_layers: the number of hidden layers
            n_neurons: the number of neurons in each hidden layer
            sphere_init: whether to initialize the output of the network to be
                the SDF of a sphere
            sphere_init_radius: the radius of the initialized sphere
            weight_norm: whether to use weight normalization
            output_activation: the activation function to use for the output
                layer
        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.sphere_init = sphere_init
        self.sphere_init_radius = sphere_init_radius
        self.weight_norm = weight_norm
        self.output_activation = output_activation
        # make first layer
        self.layers = [
            self.make_linear_layer(
                dim_in, n_neurons,
                is_first_layer=True,
                is_last_layer=False
            ),
            self.make_activation()
        ]
        # make hidden layers
        for _ in range(n_hidden_layers - 1):
            self.layers += [
                self.make_linear_layer(
                    n_neurons, n_neurons,
                    is_first_layer=False,
                    is_last_layer=False
                ),
                self.make_activation()
            ]
        # make last layer
        self.layers += [
            self.make_linear_layer(
                n_neurons, dim_out,
                is_first_layer=False,
                is_last_layer=True
            ),
        ]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = get_activation(output_activation)
    
    @torch.cuda.amp.autocast(False)
    def forward(self, x):
        """Forward pass of the network."""
        x = self.layers(x.float())
        x = self.output_activation(x)
        return x
    
    def make_linear_layer(
            self,
            dim_in: int,
            dim_out: int,
            is_first_layer: bool = False,
            is_last_layer: bool = False,
    ):
        """Create a linear layer with proper initialization."""
        # network without bias will degrade quality
        layer = nn.Linear(dim_in, dim_out, bias=True)
        if self.sphere_init:
            if is_last_layer:
                torch.nn.init.constant_(
                    layer.bias, -self.sphere_init_radius
                )
                torch.nn.init.normal_(
                    layer.weight,
                    mean=math.sqrt(math.pi) / math.sqrt(dim_in),
                    std=0.0001
                )
            elif is_first_layer:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(
                    layer.weight[:, :3], 0.0,
                    std=math.sqrt(2) / math.sqrt(dim_out)
                )
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(
                    layer.weight, 0.0,
                    std=math.sqrt(2) / math.sqrt(dim_out)
                )
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(
                layer.weight, nonlinearity='relu'
            )
        
        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer
    
    def make_activation(self):
        """Create an activation layer."""
        if self.sphere_init:
            return nn.Softplus(beta=100)
        # else:
        return nn.ReLU(inplace=True)


class VarianceNetwork(nn.Module):
    """Stores current variance for converting sdf to density."""
    
    def __init__(
            self,
            device: torch.device,
            init_val: float = 0.3,
    ):
        super().__init__()
        self.device = device
        self.init_val = init_val
        self.variance = nn.Parameter(
            torch.tensor(self.init_val, device=device)
        )
    
    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0).clip(1e-6, 1e6)
        return val
    
    def forward(self, x):
        return (torch.ones([len(x), 1], device=self.device)
                * self.inv_s)


class NeuSField(nn.Module):
    """The neural signed distance field that returns the signed distance,
    derived normal, radiance, and material at the given positions.
    """
    
    def __init__(
            self,
            args: configargparse.Namespace,
            device: torch.device,
            aabb: torch.Tensor,
            n_far_lights: int,
            n_near_lights: int,
            material_dim: bool,
    ):
        """Initialize the NSR.

        Create the geometric and appearance encoding, and decoder MLPs
        including SDF, radiance, and material.

        Args:
            args: the command line arguments
            device: the device to run the model
            aabb: (2, 3) the axis-aligned bounding box of the scene
            n_far_lights: the number of far light sources
            n_near_lights: the number of near light sources
            material_dim: the output dimension of the material decoder
        """
        super().__init__()
        
        self.args = args
        self.device = device
        self.aabb = aabb
        self.n_far_lights = n_far_lights
        self.n_near_lights = n_near_lights
        self.material_dim = material_dim
        self.aabb_size = aabb[1] - aabb[0]
        self.aabb_diag = torch.tensor(self.aabb_size).norm(p=2).item()
        self.step_length = self.aabb_diag / args.n_samples_per_ray
        self.anneal_end_iter = args.anneal_end_iter
        
        self.initialize_mat_encoding_as_clone = True
        self.fix_shape_and_radiance_in_pbr = args.fix_shape_and_radiance_in_pbr
        
        # create two resolution hash grid encodings
        finest_reso = args.hash_grid_finest_reso
        coarsest_reso = args.hash_grid_coarsest_reso
        n_levels = args.hash_grid_n_levels
        per_level_scale = math.exp(
            math.log(finest_reso / coarsest_reso)
            / (n_levels - 1)
        )
        self.hash_config = {
            'otype': 'HashGrid',
            'n_levels': n_levels,
            'n_features_per_level': 2,
            'log2_hashmap_size': 19,
            'base_resolution': coarsest_reso,
            'per_level_scale': per_level_scale,
            # 'interpolation': args.geo_encoding_interpolation,
        }
        with torch.cuda.device(device):
            # create hash grid encoding for geometry and material
            self.geo_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=self.hash_config,
            )
            self.mat_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=self.hash_config,
            )
            # directional encoding for view and light directions
            self.dir_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    'otype': 'SphericalHarmonics',
                    'degree': 4,
                }
            )
        
        # note that geometry feature and geometry encoding are different.
        # geometry feature is the output of the SDF network, while geometry
        # encoding is the input to the SDF network, or the output of the
        # hash grid encoding.
        self.n_geo_feature_dims = args.n_geo_feature_dims
        assert self.n_geo_feature_dims >= 1, \
            'n_feature_dims must be at least 1, ' \
            'because the first dimension is the signed distance.'
        
        # create SDF decoder MLP, whose
        #   input is (xyz, geo_encoding),
        #   output is (geo_feature (sdf as the first dim))
        # do not use fully fused network, as it does not support
        # second-order derivatives?
        self.sdf_decoder = VanillaMLP(
            3 + self.geo_encoding.n_output_dims,
            self.n_geo_feature_dims,
            n_hidden_layers=1,
            n_neurons=64,
            sphere_init=True,
            sphere_init_radius=args.sphere_init_radius,
            weight_norm=True,
        )
        self.sdf_decoder = self.sdf_decoder.to(device)
        self.grad_type = 'analytic'
        
        # create light code embedding vector for appearance under each light
        # source.
        self.n_light_code_dims = args.n_light_code_dims
        self.light_codes = nn.Parameter(
            torch.zeros(
                n_far_lights + n_near_lights,
                self.n_light_code_dims,
                device=device,
            ).normal_(0, 0.1),
            requires_grad=True
        )
        
        # create appearance decoder as FullyFusedMLPs, including:
        # - far-field radiance network, whose
        # input is (xyz, light_modulated_app_encoding, encoded_view_dirs,
        #           geo_feature),
        # output is (far_radiance)
        self.far_rgb_decoder = tcnn.Network(
            n_input_dims=3 +  # position
                         self.n_geo_feature_dims +
                         3 +  # normal
                         # 3 +  # reflected view direction
                         self.n_light_code_dims +
                         self.dir_encoding.n_output_dims,
            n_output_dims=3,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'sigmoid',
                'n_neurons': 64,
                'n_hidden_layers': 2,
            }
        )
        
        # - near-field radiance network, whose
        # input is (xyz, light_modulated_app_encoding, encoded_view_dirs,
        #           encoded_light_dirs, geo_feature),
        # output is (near_radiance)
        self.near_rgb_decoder = tcnn.Network(
            n_input_dims=3 +  # position
                         self.n_geo_feature_dims +
                         3 +  # normal
                         self.n_light_code_dims +
                         # self.dir_encoding.n_output_dims +  # light_dir
                         3 +  # light direction
                         self.dir_encoding.n_output_dims,
            n_output_dims=3,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'sigmoid',
                'n_neurons': 64,
                'n_hidden_layers': 2,
            }
        )
        
        # - material network, whose
        # input is (average_app_encoding, geo_feature),
        # output is (Optional[normal], BRDF parameters)
        self.material_decoder = tcnn.Network(
            n_input_dims=3 +  # position
                         self.n_geo_feature_dims +
                         3 +  # derived normal
                         self.mat_encoding.n_output_dims,
            n_output_dims=self.material_dim,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'none',
                'n_neurons': 64,
                'n_hidden_layers': 2,
            }
        )
        
        # variance network
        self.variance = VarianceNetwork(device=device)
        
        self.pbr_enabled = False
        
        # print('NeuSField initialized. Parameters:')
        # for name, param in self.named_parameters():
        #     print(f'  {name}: {param.shape}')
    
    def enable_pbr(self):
        """Add materials for PBR.

        Returns:
            List[Dict]: a list of material-related parameter groups for
                optimizer
        """
        if not self.pbr_enabled:
            # create material encoding
            if self.initialize_mat_encoding_as_clone:
                self.mat_encoding.params.data = (
                    self.geo_encoding.params.data.clone())
            
            self.pbr_enabled = True
            print('PBR enabled for NeuSField.')
            
            if self.fix_shape_and_radiance_in_pbr:
                # fix shape and radiance in PBR
                self.geo_encoding.requires_grad_(False)
                self.sdf_decoder.requires_grad_(False)
                self.light_codes.requires_grad_(False)
                self.far_rgb_decoder.requires_grad_(False)
                self.near_rgb_decoder.requires_grad_(False)
                self.variance.requires_grad_(False)
                print('Shape and radiance fixed in PBR.')
        
        # return material-related param groups
        lr_material = self.args.lr_material_init
        param_groups = [
            {'params': list(self.mat_encoding.parameters()),
             'name': 'mat_encoding', 'lr': lr_material
             },
            {'params': list(self.material_decoder.parameters()),
             'name': 'material_decoder', 'lr': lr_material
             },
        ]
        return param_groups
    
    def get_alpha(
            self,
            sdf: torch.Tensor,
            normal: torch.Tensor,
            view_dirs_in: torch.Tensor,
            seg_length: torch.Tensor,
    ) -> torch.Tensor:
        """Get the alpha value for the given sdf.

        Args:
            sdf: (..., 1) the signed distance
            normal: (..., 3) the normal
            view_dirs_in: (..., 3) the normalized view directions pointing
                TOWARDS the surface
            seg_length: (..., 1) the length of the ray segment

        Returns:
            torch.Tensor: (..., 1) the alpha value
        """
        cur_iter = global_vars.cur_iter
        n_pts = sdf.shape[0]
        assert seg_length.shape == (n_pts, 1)
        # sdf = sdf.view(n_pts, 1)
        # seg_length = seg_length.view(n_pts, 1)
        # get variance
        inv_s = self.variance.inv_s
        inv_s = inv_s.expand(n_pts, 1)
        
        # compute SDF for prev and next step
        true_cos = (view_dirs_in * normal).sum(dim=-1, keepdim=True)
        anneal_ratio = 1.0 if self.anneal_end_iter <= 0 \
            else min(1.0, cur_iter / self.anneal_end_iter)
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training
        # iterations. The anneal strategy below makes the cos value "not dead"
        # at the beginning training iterations, for better convergence.
        # always non-positive
        used_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - anneal_ratio)
                     + F.relu(-true_cos) * anneal_ratio)
        estimated_next_sdf = sdf + used_cos * seg_length * 0.5  # <= sdf
        estimated_prev_sdf = sdf - used_cos * seg_length * 0.5  # >= sdf
        
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        
        p = prev_cdf - next_cdf
        c = prev_cdf
        
        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
        return alpha.view(-1, 1)
    
    def query_field(
            self,
            xyz: torch.Tensor,
            return_sdf: bool = False,
            return_sdf_grad: bool = False,
            return_geo_feature: bool = False,
            return_material: bool = False,
            return_jittered: bool = False,
            view_dirs_out: torch.Tensor = None,
            far_light_indices: List[int] = None,
            near_light_indices: List[int] = None,
            near_lights: lighting.NearLight = None,
    ) -> Dict[str, torch.Tensor]:
        """Query the NSR at the given positions for needed outputs.

        Args:
            xyz: (..., 3) the positions to query in the world space
            return_sdf: whether to return the signed distance
            return_sdf_grad: whether to return the gradient of the signed
                distance w.r.t. :attr:`xyz`
            return_geo_feature: whether to return the geometry feature
            return_material: whether to return the material. May include
                shading normal and BRDF parameters.
            return_jittered: whether to also return material at
                jittered positions. Needed when computing smoothness loss.
            view_dirs_out: (..., 3) the normalized view directions pointing
                OUTWARDS the surface. Needed whenever :attr:`far_light_indices`
                or :attr:`near_light_indices` is not [].
            far_light_indices: the indices of the far light sources to query
                for radiance. If [], no far radiance will be computed.
            near_light_indices: the indices of the near light sources to query
                for radiance. If [], no near radiance will be computed.
                Otherwise, :attr:`near_lights` must be provided.
            near_lights: the near light sources. Needed when
                :attr:`near_light_indices` is not [].

        Returns:
            A dict of tensors, whose keys may include:

            - sdf: (..., 1) the signed distance
            - sdf_grad: (..., 3) the gradient of the signed distance if
                :attr:`return_sdf_grad`
            - geo_feature: (..., n_geo_feature_dims) the geometry feature if
                :attr:`return_geo_feature`
            - material: (..., material_dim) the material code to unpack, if
                :attr:`return_material`
            - far_radiance_i: (..., 3) the radiance under the i-th far light
                source if :attr:`return_far_radiance` includes i
            - near_radiance_i: (..., 3) the radiance under the i-th near light
                source if :attr:`return_near_radiance` includes i
        """
        cur_iter = global_vars.cur_iter
        # assert len(xyz.shape) == 2
        # n_pts = xyz.shape[0]
        # check inputs
        if return_material and not self.pbr_enabled:
            raise ValueError(
                'Can not return material when PBR is not enabled.'
            )
        
        if near_light_indices is None:
            near_light_indices = []
        return_near_radiance = len(near_light_indices) > 0
        if return_near_radiance and near_lights is None:
            raise ValueError(
                'near_lights must be provided when '
                'return_near_radiance is True.'
            )
        for near_light_index in near_light_indices:
            assert 0 <= near_light_index < self.n_near_lights, \
                'near_light_index must be in [0, n_near_lights)'
        
        if far_light_indices is None:
            far_light_indices = []
        for far_light_index in far_light_indices:
            assert 0 <= far_light_index < self.n_far_lights, \
                'far_light_index must be in [0, n_far_lights)'
        return_far_radiance = len(far_light_indices) > 0
        
        if return_jittered:
            assert self.training, \
                'Should only need jittered material and normal in training.'
        
        assert return_sdf or return_sdf_grad or return_geo_feature or \
               return_material or return_far_radiance or return_near_radiance, \
            'Nothing is queried.'
        
        # assert xyz inside the bounding box
        assert (xyz >= self.aabb[0] - 1e-5).all() and (
                xyz <= self.aabb[1] + 1e-5).all(), \
            'xyz must be inside the bounding box.'
        
        need_sdf_grad = return_sdf_grad or return_far_radiance or \
                        return_near_radiance or return_material
        
        jitter_on_surface = False
        jitter_range = self.args.jitter_range
        
        input_shape = xyz.shape[:-1]
        xyz = xyz.view(-1, 3)
        ret_dict = {}
        with torch.set_grad_enabled(self.training):
            if need_sdf_grad and self.grad_type == 'analytic':
                # points may be in inference mode, get a copy to enable grad
                if not self.training:
                    xyz = xyz.clone()
                xyz.requires_grad_(True)
            
            if return_jittered:
                with torch.no_grad():
                    # jitter in 3D
                    r0 = torch.rand(
                        xyz.shape[0], 3,
                        device=self.device
                    ) * 2 - 1
                    r0 *= jitter_range
                    xyz_jitter = xyz + r0
                xyz_jitter.requires_grad_(True)
                xyz_jitter_normed = scale_anything(
                    xyz_jitter, self.aabb, (0, 1)
                )
                xyz_jitter_normed = xyz_jitter_normed.clamp(0, 1)
            
            # compute the sdf and geometry feature
            with torch.set_grad_enabled(self.training or need_sdf_grad):
                # points normalized to (0, 1)
                xyz_normed = scale_anything(
                    xyz, self.aabb, (0, 1)
                )
                
                # encode the points
                xyz_geo_encoding = self.geo_encoding(xyz_normed)
                decoder_input = torch.cat(
                    [xyz_normed * 2 - 1,
                     xyz_geo_encoding],
                    dim=-1
                )
                geo_feature = self.sdf_decoder(decoder_input)
                geo_feature = geo_feature.float()
                sdf = geo_feature[:, 0:1]
                
                # compute sdf gradient
                if need_sdf_grad:
                    sdf_grad = torch.autograd.grad(
                        sdf, xyz,
                        torch.ones_like(sdf),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    normal = F.normalize(sdf_grad, dim=-1, p=2, eps=1e-8)
                
                if return_jittered:
                    xyz_geo_encoding_jitter = self.geo_encoding(
                        xyz_jitter_normed
                    )
                    decoder_input = torch.cat(
                        [xyz_jitter_normed * 2 - 1,
                         xyz_geo_encoding_jitter],
                        dim=-1
                    )
                    geo_feature_jitter = self.sdf_decoder(decoder_input)
                    geo_feature_jitter = geo_feature_jitter.float()
                    sdf_jitter = geo_feature_jitter[:, 0:1]
                    sdf_grad_jitter = torch.autograd.grad(
                        sdf_jitter, xyz_jitter,
                        torch.ones_like(sdf_jitter),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    normal_jitter = F.normalize(
                        sdf_grad_jitter,
                        dim=-1, p=2, eps=1e-8
                    )
            
            if return_sdf:
                ret_dict['sdf'] = sdf
            if return_geo_feature:
                ret_dict['geo_feature'] = geo_feature
            if return_sdf_grad:
                ret_dict['sdf_grad'] = sdf_grad
            if return_jittered:
                ret_dict['normal_jitter'] = normal_jitter
            
            # compute radiance under far light sources
            if return_far_radiance:
                for far_light_index in far_light_indices:
                    light_code = self.light_codes[far_light_index]
                    light_code = light_code.view(1, -1).expand(xyz.shape[0], -1)
                    
                    # # reflect view_dirs by normal
                    # normal_de = normal.detach()
                    # refl_view_dirs_out = (
                    #         2 * normal_de *
                    #         (view_dirs_out * normal_de).sum(-1, keepdims=True)
                    #         - view_dirs_out)
                    # refl_view_dirs_out = F.normalize(
                    #     refl_view_dirs_out, dim=-1, p=2, eps=1e-8)
                    
                    # the input of dir_encoding must in [0, 1]
                    encoded_view_dirs = self.dir_encoding(
                        (view_dirs_out + 1) / 2
                    )
                    
                    with torch.set_grad_enabled(self.training):
                        decoder_input = torch.cat(
                            [xyz_normed * 2 - 1,
                             geo_feature,
                             normal,
                             # refl_view_dirs_out,
                             # xyz_app_encoding * light_modulation,
                             light_code,
                             # geo_feature_modulated,
                             encoded_view_dirs,
                             ],
                            dim=-1
                        )
                        radiance = self.far_rgb_decoder(decoder_input)
                        radiance = radiance.float()
                    ret_dict[f'far_radiance_{far_light_index}'] = radiance
            
            # compute radiance under near light sources
            if return_near_radiance:
                # query the near light sources for relative light directions
                wi, dist, intensities = (
                    near_lights.get_intensity_and_relative_dir_dist(xyz))
                for near_light_index in near_light_indices:
                    light_code = self.light_codes[
                        self.n_far_lights + near_light_index]
                    light_code = light_code.view(1, -1).expand(xyz.shape[0], -1)
                    
                    # # reflect view_dirs by normal
                    # view_dirs_out = (
                    #         2 * normal *
                    #         (view_dirs_out * normal).sum(-1, keepdims=True)
                    #         - view_dirs_out)
                    # view_dirs_out = F.normalize(
                    #     view_dirs_out, dim=-1, p=2, eps=1e-8)
                    
                    # the input of dir_encoding must in [0, 1]
                    encoded_view_dirs = self.dir_encoding(
                        (view_dirs_out + 1) / 2
                    )
                    
                    light_dirs = wi[:, near_light_index]
                    # encoded_light_dirs = self.dir_encoding(
                    #     (light_dirs + 1) / 2)
                    
                    with torch.set_grad_enabled(self.training):
                        decoder_input = torch.cat(
                            [xyz_normed * 2 - 1,
                             geo_feature,
                             normal,
                             light_code,
                             # xyz_app_encoding * light_modulation,
                             light_dirs,
                             encoded_view_dirs,
                             ],
                            dim=-1
                        )
                        radiance = self.near_rgb_decoder(decoder_input)
                        radiance = radiance.float()
                        # post-process the radiance
                        # multiply by cosine term
                        anneal_ratio = 1.0 \
                            if self.anneal_end_iter <= 0 \
                            else min(1.0, cur_iter / self.anneal_end_iter)
                        true_cos = (light_dirs * normal).sum(
                            dim=-1, keepdim=True
                        )
                        used_cos = (F.relu(true_cos)
                                    * anneal_ratio
                                    + F.relu(true_cos * 0.5 + 0.5) *
                                    (1.0 - anneal_ratio))
                        true_irradiance = (
                                intensities[:, near_light_index]
                                / (dist[:, near_light_index] ** 2))
                        used_irradiance = (
                                true_irradiance * anneal_ratio +
                                1.0 * (1.0 - anneal_ratio))
                        radiance = radiance * used_cos * used_irradiance
                    
                    ret_dict[f'near_radiance_{near_light_index}'] = radiance
            
            # compute material
            if return_material:
                # encode the points
                xyz_mat_encoding = self.mat_encoding(xyz_normed)
                decoder_input = torch.cat(
                    [xyz_normed * 2 - 1,
                     geo_feature,
                     normal,
                     xyz_mat_encoding
                     ],
                    dim=-1
                )
                material = self.material_decoder(decoder_input)
                material = material.float()
                # unpacked_material = self.unpack_brdf(material)
                ret_dict['material'] = material
                
                if return_jittered:
                    xyz_mat_encoding_jitter = self.mat_encoding(xyz_jitter_normed)
                    decoder_input = torch.cat(
                        [xyz_jitter_normed * 2 - 1,
                         geo_feature_jitter,
                         normal_jitter,
                         xyz_mat_encoding_jitter
                         ],
                        dim=-1
                    )
                    material_jitter = self.material_decoder(decoder_input)
                    material_jitter = material_jitter.float()
                    ret_dict['material_jitter'] = material_jitter
        
        for k, v in ret_dict.items():
            v = v.view(*input_shape, v.shape[-1])
            v = v if self.training else v.detach()
            ret_dict[k] = v
        
        return ret_dict
    
    def query_sdf(
            self,
            xyz: torch.Tensor,
            with_grad: bool = False,
            with_geo_feature: bool = False,
            return_jittered: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Query the signed distance field at the given positions.

        Args:
            xyz: (..., 3) the positions in the world space
            with_grad: whether to compute the gradient of the SDF w.r.t. the
                input positions
            with_geo_feature: whether to return the geometry feature predicted
                by the SDF network
            return_jittered: whether to also return the normal at the jittered

        Returns:
            A list of tensors, may include the following in order:

            - sdf: (..., 1) the signed distance
            - sdf_grad: (..., 3) the gradient of the signed distance if
                :attr:`with_grad` is ``True``
            - feature: (..., n_feature_dims) the geometry feature if
                :attr:`with_feature` is ``True``
        """
        return self.query_field(
            xyz,
            return_sdf=True,
            return_sdf_grad=with_grad,
            return_geo_feature=with_geo_feature,
            return_jittered=return_jittered,
        )
    
    def query_material(
            self,
            xyz: torch.Tensor,
            return_jittered: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Query the material at the given positions.

        Args:
            xyz: (..., 3) the positions in the world space
            return_jittered: whether to also return the material at the
                jittered position (for smoothness loss)

        Returns:
            A dict of tensors. See :meth:`unpack_brdf` for the keys and values.
        """
        return self.query_field(
            xyz,
            return_material=True,
            return_jittered=return_jittered,
        )
    
    def query_far_radiance(
            self,
            xyz: torch.Tensor,
            view_dirs: torch.Tensor,
            far_light_indices: List[int],
    ):
        """Query the radiance under far light sources at the given positions.

        Args:
            xyz: (..., 3) the positions to query in the world space
            view_dirs: (..., 3) the normalized view directions pointing OUTWARDS
                from the surface. Needed whenever :attr:`far_light_indices` or
                :attr:`near_light_indices` is not [].
            far_light_indices: the indices of the far light sources to query
                for radiance. If [], no far radiance will be computed.

        Returns:
            A dict of tensors, whose keys may include:

            - far_radiance_i: (..., 3) the radiance under the i-th far light
                source if :attr:`return_far_radiance` includes i
        """
        return self.query_field(
            xyz,
            view_dirs_out=view_dirs,
            far_light_indices=far_light_indices,
        )
    
    def query_near_radiance(
            self,
            xyz: torch.Tensor,
            view_dirs: torch.Tensor,
            near_light_indices: List[int],
            near_lights: lighting.NearLight,
    ):
        """Query the radiance under near light sources at the given positions.

        Args:
            xyz: (..., 3) the positions to query in the world space
            view_dirs: (..., 3) the normalized view directions pointing OUTWARDS
                from the surface. Needed whenever :attr:`far_light_indices` or
                :attr:`near_light_indices` is not [].
            near_light_indices: the indices of the near light sources to query
                for radiance. If [], no near radiance will be computed.
                Otherwise, :attr:`near_lights` must be provided.
            near_lights: the near light sources. Needed when
                :attr:`near_light_indices` is not [].

        Returns:
            A dict of tensors, whose keys may include:

            - near_radiance_i: (..., 3) the radiance under the i-th near light
                source if :attr:`return_near_radiance` includes i
        """
        return self.query_field(
            xyz,
            view_dirs_out=view_dirs,
            near_light_indices=near_light_indices,
            near_lights=near_lights,
        )
    
    def get_optim_param_groups(
            self,
    ) -> List[Dict[str, Union[torch.Tensor, str, float]]]:
        """Get the parameter groups for the optimizer."""
        param_groups = []
        lr_geometry = self.args.lr_geometry_init
        lr_variance = self.args.lr_variance_init
        
        if not self.pbr_enabled or not self.fix_shape_and_radiance_in_pbr:
            param_groups += [
                {'params': list(self.geo_encoding.parameters()),
                 'name': 'geo_encoding', 'lr': lr_geometry
                 },
                {'params': list(self.sdf_decoder.parameters()),
                 'name': 'sdf_decoder', 'lr': lr_geometry
                 },
                {'params': self.light_codes,
                 'name': 'light_codes', 'lr': lr_geometry
                 },
                {'params': list(self.variance.parameters()),
                 'name': 'variance', 'lr': lr_variance
                 },
            ]
            if self.n_far_lights > 0:
                param_groups += [
                    {'params': list(self.far_rgb_decoder.parameters()),
                     'name': 'far_rgb_decoder', 'lr': lr_geometry
                     },
                ]
            if self.n_near_lights > 0:
                param_groups += [
                    {'params': list(self.near_rgb_decoder.parameters()),
                     'name': 'near_rgb_decoder', 'lr': lr_geometry
                     },
                ]
        
        if self.pbr_enabled:
            param_groups += self.enable_pbr()
        
        return param_groups
    
    def closure_occ_eval_fn(self):
        def occ_eval_fn(pts_xyz: torch.Tensor) -> torch.Tensor:
            """Compute density at the given points."""
            chunk_size = 2 ** 20
            n_pts = pts_xyz.shape[0]
            chunk_indices = torch.split(
                torch.arange(n_pts, device=self.device),
                chunk_size
            )
            result = []
            for chunk_idx in chunk_indices:
                xyz_chunk = pts_xyz[chunk_idx]
                sdf = self.query_sdf(xyz_chunk)['sdf']
                inv_s = self.variance.inv_s
                inv_s = inv_s.expand(sdf.shape[0], 1)
                estimated_prev_sdf = sdf + self.step_length * 0.5
                estimated_next_sdf = sdf - self.step_length * 0.5
                prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
                next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
                p = prev_cdf - next_cdf
                c = prev_cdf
                alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
                result.append(alpha)
                # result.append(
                #     scene.query_density(pts_xyz[chunk_idx]))
            return torch.cat(result, dim=0)
        
        return occ_eval_fn
