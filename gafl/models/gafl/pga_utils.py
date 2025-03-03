# Copyright (c) 2024 HITS gGmbH
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# Licensed under the BSD-3-Clause License.

import torch
import numpy as np

from functools import lru_cache

from gatr.primitives.linear import reverse, _compute_reversal, NUM_PIN_LINEAR_BASIS_ELEMENTS, equi_linear
from gatr.primitives.bilinear import _load_bilinear_basis, geometric_product
from gatr.primitives.dual import _compute_efficient_join
from gatr.utils.einsum import cached_einsum
from gatr.interface.rotation import embed_rotation
from gatr.interface.translation import embed_translation
from gatr.primitives.invariants import compute_inner_product_mask
from gatr.interface.scalar import embed_scalar
from gatr.layers.linear import EquiLinear
from gatr.primitives.invariants import norm

import math
import torch.nn as nn
from typing import Optional

from gafl.data.so3_utils import rotmat_to_rotquat



MOTOR_DIMS = [0,5,6,7,8,9,10,15]

def embed_frames(R, t):
    """
    Computes the frame embedding in PGA of a euclidean transformation T = (R, t).
    Args:
        R: Rotation matrix, shape: [*, N_res, 3, 3]
        t: Translation vector, shape: [*, N_res, 3]
    Returns:
        v: Frame embedding, shape: [*, N_res, 16]
    """
    
    rotquat = rotmat_to_rotquat(R)
    # frameflow lib uses a different quaternion convention than gatr
    rotmv = embed_rotation(rotquat[...,[1,2,3,0]])
    transmv = embed_translation(t) #gatr has a strange "-" sign in its implementation

    return geometric_product(transmv, rotmv)

def embed_quat_frames(Q, t):
    """
    Computes the frame embedding in PGA of a euclidean transformation T = (Q, t).
    Args:
        Q: Quaternion, shape: [*, N_res, 4]
        t: Translation vector, shape: [*, N_res, 3]
    Returns:
        v: Frame embedding, shape: [*, N_res, 16]
    """
    
    rotmv = embed_rotation(Q[...,[1,2,3,0]])
    transmv = embed_translation(t)

    return geometric_product(transmv, rotmv)

def reverse_versor(v):
    return _compute_reversal(device=v.device, dtype=v.dtype)[MOTOR_DIMS] * v

def apply_versor(x, v):

    """
    Applies a versor to a multivector.
    Args:
        x: Multivector to be transformed, shape: [*, 16]
        v: Versor, shape: [*, 16]
    Returns:
        y: Transformed multivector, shape: [*, 16]
    """

    v_inv = reverse(v)
    y = geometric_product(v, geometric_product(x, v_inv))

    return y


def apply_inverse_versor(x, v):
    
    """
    Applies the inverse of a versor to a multivector.
    Args:
        x: Multivector to be transformed, shape: [*, 16]
        v: Versor, shape: [*, 16], Make sure that this is properly normalized!
    Returns:
        y: Transformed multivector, shape: [*, 16]
    """

    v_inv = reverse(v)
    y = geometric_product(v_inv, geometric_product(x, v))

    return y

@lru_cache()
def load_point_versor_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:

    gp = _load_bilinear_basis("gp", device=device, dtype=dtype)
    c = cached_einsum("ijk, klm -> ijlm", gp[11:15, MOTOR_DIMS, :], gp[:, 11:15, MOTOR_DIMS])
    r = _compute_reversal(device=device, dtype=dtype)[MOTOR_DIMS]
    return c * r

def apply_point_versor(x, v, threshold=1e-3):

    mv = torch.empty(x.shape[:-1] + (4,), device=x.device, dtype=x.dtype)
    mv[..., 3] = 1.0
    mv[..., 2] = -x[..., 0]  # x-coordinate embedded in x_023
    mv[..., 1] = x[..., 1]  # y-coordinate embedded in x_013
    mv[..., 0] = -x[..., 2]  # z-coordinate embedded in x_012

    mv = cached_einsum("i j k l, ... j, ... k, ... l -> ... i", load_point_versor_kernel(device=x.device, dtype=x.dtype), v, mv, v)
    return mv


def cross_gp(x: torch.Tensor, y: torch.Tensor):
    """
    Parameters
    ----------
    x : torch.Tensor with shape (..., N_res, 16)

    y : torch.Tensor with shape (..., N_res, 16)

    Returns
    -------
    outputs : torch.Tensor with shape (..., N_res, N_res, 16)
    """

    # Select kernel on correct device
    gp = _load_bilinear_basis("gp", device=x.device, dtype=x.dtype)

    # Compute geometric product
    outputs = cached_einsum("i j k, ... nj, ... mk -> ... nmi", gp, x, y)

    return outputs

def relative_global_frame_transformation(v):
      
    v_inv = reverse(v)
    v_global = cross_gp(v, v_inv)
    
    return v_global

def relative_frame_transformation(v):
    """
    Computes invariant and equivariant representations of relative frame transformations between all pairs of frames.
    Args:
        v: Frame representations for all residues, shape: [*, N_res, 16]
        Assume normalized frames v ~v = 1

    Returns:
        v_local: Invariant relative frame transformations, shape: [*, N_res, N_res, 16]
    """
        
    v_inv = reverse(v)
    v_local = cross_gp(v_inv, v)
    
    return v_local

# When working with local frames, we do not need to ensure equivariance of the join using a reference multivector
# Also we only want SE(3) equivariance and not E(3) equivariance
def efficient_join(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Computes the join, using the efficient implementation.

    Parameters
    ----------
    x : torch.Tensor
        Left input multivector.
    y : torch.Tensor
        Right input multivector.

    Returns
    -------
    outputs : torch.Tensor
        Equivariant join result.
    """

    kernel = _compute_efficient_join(x.device, x.dtype)
    return cached_einsum("i j k , ... j, ... k -> ... i", kernel, x, y)

def extract_motor(v):
    """
    Extracts the motor from a frame embedding in PGA.
    Args:

        v: Frame embedding, shape: [*, 16]
    Returns:
        m: Motor, shape: [*, 8]
    """
    return v[..., MOTOR_DIMS]

def embed_motor(m):
    """
    Embeds a motor in a frame embedding in PGA.
    Args:
        m: Motor, shape: [*, 8]
    Returns:
        v: Frame embedding, shape: [*, 16]
    """

    v = torch.zeros(m.shape[:-1] + (16,), device=m.device, dtype=m.dtype)
    v[..., MOTOR_DIMS] = m
    return v

@lru_cache()
def compute_inf_norm_mask(device=torch.device("cpu")) -> torch.Tensor:
    # Invert boolean inner product mask
    return ~compute_inner_product_mask(device=device)

def inf_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x[..., compute_inf_norm_mask(device=x.device)], dim=-1, keepdim=True)


from typing import Tuple
from torch import nn

def equi_layer_norm(
    x: torch.Tensor, channel_dim: int = -2, epsilon: float = 0.01
) -> torch.Tensor:

    # Compute mean_channels |inputs|^2
    squared_norms = x[..., compute_inner_product_mask(device=x.device)].pow(2).sum(dim=-1, keepdim=True)
    squared_norms = torch.mean(squared_norms, dim=channel_dim, keepdim=True)

    # Insure against low-norm tensors (which can arise even when `x.var(dim=-1)` is high b/c some
    # entries don't contribute to the inner product / GP norm!)
    squared_norms = torch.clamp(squared_norms, epsilon)

    # Rescale inputs
    outputs = x / torch.sqrt(squared_norms)

    return outputs

class EquiLayerNorm(nn.Module):

    def __init__(self, mv_channel_dim=-2, epsilon: float = 0.01):
        super().__init__()
        self.mv_channel_dim = mv_channel_dim
        self.epsilon = epsilon

    def forward(
        self, multivectors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        outputs_mv = equi_layer_norm(
            multivectors, channel_dim=self.mv_channel_dim, epsilon=self.epsilon
        )

        return outputs_mv
    
class MVLinear(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, splits=1):
        super().__init__()
        assert out_channels % splits == 0

        weights = []
        for i in range(splits):
            weights.append(
                torch.empty((out_channels // splits, in_channels, NUM_PIN_LINEAR_BASIS_ELEMENTS))
            )

        mv_component_factors, mv_factor = self._compute_init_factors(1.0, 1.0 / np.sqrt(3.0), True)

        # Let us fist consider the multivector outputs.
        fan_in = in_channels
        bound = mv_factor / np.sqrt(fan_in)
        for i in range(splits):
            for j, factor in enumerate(mv_component_factors):
                nn.init.uniform_(weights[i][..., j], a=-factor * bound, b=factor * bound)

        concatenated_weights = torch.cat(weights, dim=0)
        self.weight = nn.Parameter(concatenated_weights)
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_channels, 1)))
        else:
            self.bias = None

    def forward(
        self, multivectors: torch.Tensor
    ):
        outputs_mv = equi_linear(multivectors, self.weight)  # (..., out_channels, 16)

        if self.bias is not None:
            bias = embed_scalar(self.bias)
            outputs_mv = outputs_mv + bias

        return outputs_mv, None

    @staticmethod
    def _compute_init_factors(gain, additional_factor, use_mv_heuristics):
        """Computes prefactors for the initialization.

        See self.reset_parameters().
        """
        mv_factor = gain * additional_factor * np.sqrt(3)

        # Individual factors for each multivector component
        if use_mv_heuristics:
            mv_component_factors = torch.sqrt(
                torch.Tensor([1.0, 4.0, 6.0, 2.0, 0.5, 0.5, 1.5, 1.5, 0.5])
            )
        else:
            mv_component_factors = torch.ones(NUM_PIN_LINEAR_BASIS_ELEMENTS)
        return mv_component_factors, mv_factor

class PairwiseGeometricBilinear(nn.Module):
    """Geometric bilinear layer between two different sets of multivectors.

    Pin-equivariant map between multivector tensors that constructs new geometric features via
    geometric products and the equivariant join (based on a reference vector).

    Parameters
    ----------
    in_mv_channels : int
        Input multivector channels of `x`
    out_mv_channels : int
        Output multivector channels
    hidden_mv_channels : int or None
        Hidden MV channels. If None, uses out_mv_channels.
    in_s_channels : int or None
        Input scalar channels of `x`. If None, no scalars are expected nor returned.
    out_s_channels : int or None
        Output scalar channels. If None, no scalars are expected nor returned.
    """

    def __init__(
        self,
        in_channels: int,
        
        out_channels: int,
        hidden_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Default options
        if hidden_channels is None:
            hidden_channels = out_channels

        out_channels_each = hidden_channels // 2
        assert (
            out_channels_each * 2 == hidden_channels
        ), "GeometricBilinear needs even channel number"

        self.linear_l = MVLinear(in_channels, 2 * out_channels_each, splits=2)
        self.linear_r = MVLinear(in_channels, 2 * out_channels_each, splits=2)
        self.out_channels_each = out_channels_each

        # Output linear projection
        self.linear_out = EquiLinear(
            hidden_channels, out_channels
        )

    def forward(
        self,
        mv1: torch.Tensor,
        mv2: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        mv1 : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors argument 1
        mv2 : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors argument 2

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., self.out_mv_channels, 16)
            Output multivectors
        """
        
        l, _ = self.linear_l(mv1)
        r, _ = self.linear_r(mv2)

        gp_out = geometric_product(l[..., :self.out_channels_each, :], r[..., :self.out_channels_each, :])
        join_out = efficient_join(l[..., self.out_channels_each:, :], r[..., self.out_channels_each:, :])

        # Output linear
        out = torch.cat((gp_out, join_out), dim=-2)
        out, _ = self.linear_out(out)

        return out

# Return list of slices, where each element in the list sclices the corresponding grade
def _grade_to_slice(dim):
    grade_to_slice = list()
    subspaces = torch.as_tensor([math.comb(dim, i) for i in range(dim + 1)])
    for grade in range(dim + 1):
        index_start = subspaces[:grade].sum()
        index_end = index_start + math.comb(4, grade)
        grade_to_slice.append(slice(index_start, index_end))
    return grade_to_slice

@lru_cache()
def bilinear_product_paths(type='gmt'):
    dim = 4
    grade_to_slice = _grade_to_slice(dim)
    gp_paths = torch.zeros((dim + 1, dim + 1, dim + 1), dtype=bool)

    if type == 'gmt':
        mt = _load_bilinear_basis('gp')
    elif type == 'jmt':
        mt = _compute_efficient_join()

    for i in range(dim + 1):
        for j in range(dim + 1):
            for k in range(dim + 1):
                s_i = grade_to_slice[i]
                s_j = grade_to_slice[j]
                s_k = grade_to_slice[k]

                m = mt[s_i, s_j, s_k]
                gp_paths[i, j, k] = (m != 0).any()

    return gp_paths

# Geometric many body contraction layer currently only implemeted for n=3
class GMBCLayer(nn.Module):

    def __init__(self, num_features):
        """
        Args:
            num_features:
                Number of output features
        """

        super(GMBCLayer, self).__init__()
        self.register_buffer("subspaces", torch.as_tensor([math.comb(4, i) for i in range(4 + 1)]))
        self.num_features = num_features

        self.register_buffer("gp", bilinear_product_paths('gmt'))
        self.register_buffer("jp", bilinear_product_paths('jmt'))

        self.gp_weights = nn.Parameter(torch.empty(num_features, self.gp.sum()))
        self.jp_weights = nn.Parameter(torch.empty(num_features, self.jp.sum()))

        self.linear = MVLinear(num_features, 2*num_features, splits=2)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.gp_weights, std=1 / (math.sqrt(4 + 1)))
        torch.nn.init.normal_(self.jp_weights, std=1 / (math.sqrt(4 + 1)))

    def _get_weight(self):
        gp_weights = torch.zeros(
            self.num_features,
            *self.gp.size(),
            dtype=self.gp_weights.dtype,
            device=self.gp_weights.device,
        )
        gp_weights[:, self.gp] = self.gp_weights
        gp_weights_repeated = (
            gp_weights.repeat_interleave(self.subspaces, dim=-3)
            .repeat_interleave(self.subspaces, dim=-2)
            .repeat_interleave(self.subspaces, dim=-1)
        )

        jp_weights = torch.zeros(
            self.num_features,
            *self.jp.size(),
            dtype=self.jp_weights.dtype,
            device=self.jp_weights.device,
        )
        jp_weights[:, self.jp] = self.jp_weights
        jp_weights_repeated = (
            jp_weights.repeat_interleave(self.subspaces, dim=-3)
            .repeat_interleave(self.subspaces, dim=-2)
            .repeat_interleave(self.subspaces, dim=-1)
        )
        return _load_bilinear_basis('gp', dtype=self.gp_weights.dtype, device=self.gp_weights.device) * gp_weights_repeated + _compute_efficient_join(dtype=self.gp_weights.dtype, device=self.gp_weights.device) * jp_weights_repeated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [*, C_v] single representation
        Returns:
            [*, C_s] single representation update
        """
        W = self._get_weight()
        x, _ = self.linear(x)
        return cached_einsum('nijk,...nj,...nk->...ni', W, x[..., :self.num_features, :], x[..., self.num_features:, :]) + x[..., self.num_features:, :]

class point_trafo(nn.Module):

    def __init__(self):
        super(point_trafo, self).__init__()

    def forward(self, x, T):
        return apply_point_versor(x, T)

class ga_norm(nn.Module):

    def __init__(self):
        super(ga_norm, self).__init__()

    def forward(self, x):
        n = norm(x)[..., 0]
        inf_n = inf_norm(x)[..., 0]
        return n, inf_n

class ga_versor(nn.Module):

    def __init__(self):
        super(ga_versor, self).__init__()

    def forward(self, x, T):
        return apply_versor(x, T)
    
class ga_inverse_versor(nn.Module):

    def __init__(self):
        super(ga_inverse_versor, self).__init__()

    def forward(self, x, T):
        return apply_inverse_versor(x, T)
    
class global_rel_trafo(nn.Module):
    
    def __init__(self):
        super(global_rel_trafo, self).__init__()

    def forward(self, T):
        return relative_global_frame_transformation(T)
    
class local_rel_trafo(nn.Module):
    
    def __init__(self):
        super(local_rel_trafo, self).__init__()

    def forward(self, T):
        return relative_frame_transformation(T)