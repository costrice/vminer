"""
This module defines the encoding layers used in the project.
"""
import tinycudann as tcnn
import torch
import torch.nn as nn


class FrequencyEncoding(nn.Module):
    def __init__(self, n_freq: int, in_dim: int, device: torch.device):
        """

        Args:
            n_freq: the number of frequencies to use
            in_dim: the dimension of the input appearance feature vector
            device: the device to run the model
        """
        super().__init__()
        
        self.n_freq = n_freq
        self.in_dim = in_dim
        self.out_dim = in_dim * n_freq * 2
        self.device = device
        
        self.freq_bands = (2 ** torch.arange(
            n_freq, device=device, dtype=torch.float32
        )).requires_grad_(False)
    
    def forward(self, x: torch.Tensor):
        """
        Perform frequency encoding on the input tensor (on the last dim).
        Args:
            x: (n_pts, code_dim) the input tensor to be encoded.

        Returns:
            torch.Tensor: (n_pts, code_dim * n_freq * 2) the encoded tensor.
        """
        encoded = (x[..., None] * self.freq_bands).view(
            -1, self.in_dim * self.n_freq
        )
        encoded = torch.cat(
            [torch.sin(encoded), torch.cos(encoded)],
            dim=-1
        )
        return encoded


class HashGridEncoding(nn.Module):
    """The multi-resolution hash grid encoding used in Instant-NGP."""
    
    def __init__(
            self,
            n_input_dims: int,
            device: torch.device,
            include_xyz: bool,
            xyz_scale: float = 2.,
            xyz_offset: float = -1.,
    ):
        """Build the multi-resolution hash grid encoding, and optionally
        concat the xyz coordinates to the encoding.

        Args:
            n_input_dims: the dimension of the input vector (typically 3)
            device: the device to put the grid on
            include_xyz: whether to include xyz coordinates in the output
                encoding as the first 3 dims
            xyz_scale: scale the xyz coordinates by this factor
            xyz_offset: offset the xyz coordinates by this factor after scaling
        """
        super().__init__()
        # create grid on the device
        with torch.cuda.device(device):
            self.hash_grid = tcnn.Encoding(
                n_input_dims,
                encoding_config={
                    'otype': 'HashGrid',
                    'n_levels': 16,
                    'n_features_per_level': 2,
                    'log2_hashmap_size': 19,
                    'base_resolution': 16,
                    'per_level_scale': 1.3819129,  # (2048 / 16) ** (1 / 15)
                }
            )
        self.include_xyz = include_xyz
        self.xyz_scale = xyz_scale
        self.xyz_offset = xyz_offset
        self.n_output_dims = (self.hash_grid.n_output_dims +
                              (n_input_dims if include_xyz else 0))
    
    def forward(self, xyz: torch.Tensor):
        """Perform the encoding on the input tensor.

        Args:
            xyz: (n_pts, n_input_dims) the input position to be encoded, should
                be normalized to [0, 1].

        Returns:
            torch.Tensor: (n_pts, n_output_dims) the encoded tensor.
        """
        encoded = self.hash_grid(xyz)
        if self.include_xyz:
            xyz = xyz * self.xyz_scale + self.xyz_offset
            # important: first xyz, then encoded, as MLP assumes xyz to be the
            # first 3 dims
            encoded = torch.cat([xyz, encoded], dim=-1)
        return encoded


class SHDirectionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        # the input of sh_encoding should be normalized directions, and
        # transformed from [-1, 1] to [0, 1] (x -> (x + 1) / 2)
        self.sh_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'SphericalHarmonics',
                'degree': 4,
            }
        )
        self.n_output_dims = self.sh_encoding.n_output_dims
    
    def forward(self, dirs: torch.Tensor):
        """Perform SH encoding on the input tensor (on the last dim).

        Return the SH encoding of the input tensor, which is the evaluation of
        SH bases at the given directions.

        Args:
            dirs: (..., 3) the input directions to be encoded.

        Returns:
            torch.Tensor: (..., n_degree ** 2) the encoded tensor.
        """
        dirs = (dirs + 1.) / 2.  # (-1, 1) => (0, 1)
        encoded = self.sh_encoding(dirs)
        return encoded
