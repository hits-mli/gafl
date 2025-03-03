# Based on the code for IPA in gafl/models/ipa_pytorch.py, which is licenced under:
# --------------------------------------------------------------------------
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------

# Applies to changes to the code for IPA as defined in gafl/models/ipa_pytorch.py:
# --------------------------------------------------------------------------
# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# --------------------------------------------------------------------------


"""Neural network architecture for the flow model."""
import numpy as np
import torch
import math
from scipy.stats import truncnorm
import torch.nn as nn
from typing import Optional, Callable, List, Sequence

from gatr.layers.linear import EquiLinear

from gafl.data import all_atom
from gafl.models.gafl.pga_utils import extract_motor, embed_quat_frames, PairwiseGeometricBilinear, GMBCLayer, point_trafo, ga_norm, ga_versor, ga_inverse_versor, local_rel_trafo
from gafl.models.gafl import pga_utils as pu


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def gfa_point_weights_init_(weights):
    with torch.no_grad():
        weights.fill_(-2.0) # This is an empirical value which has proven to work well in practice

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out

def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def compute_angles(ca_pos, pts):
    batch_size, num_res, num_heads, num_pts, _ = pts.shape
    calpha_vecs = (ca_pos[:, :, None, :] - ca_pos[:, None, :, :]) + 1e-10
    calpha_vecs = torch.tile(calpha_vecs[:, :, :, None, None, :], (1, 1, 1, num_heads, num_pts, 1))
    gfa_pts = pts[:, :, None, :, :, :] - torch.tile(ca_pos[:, :, None, None, None, :], (1, 1, num_res, num_heads, num_pts, 1))
    phi_angles = all_atom.calculate_neighbor_angles(
        calpha_vecs.reshape(-1, 3),
        gfa_pts.reshape(-1, 3)
    ).reshape(batch_size, num_res, num_res, num_heads, num_pts)
    return  phi_angles

class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class StructureModuleTransition(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransition, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)

        return s

class EdgeTransition(nn.Module):
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], axis=-1)
        edge_embed = torch.cat(
            [edge_embed, edge_bias], axis=-1).reshape(
                batch_size * num_res**2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(
            batch_size, num_res, num_res, -1
        )
        return edge_embed
        
class GeometricFrameAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        gfa_conf,
        geometric_input=True,
        geometric_output=True,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(GeometricFrameAttention, self).__init__()
        self._gfa_conf = gfa_conf

        self.c_s = gfa_conf.c_s
        self.c_z = gfa_conf.c_z
        self.c_hidden = gfa_conf.c_hidden
        self.no_heads = gfa_conf.no_heads
        self.no_qk_points = gfa_conf.no_qk_points
        self.no_v_points = gfa_conf.no_v_points
        self.geometric_input = geometric_input
        self.geometric_output = geometric_output
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpk = self.no_heads * self.no_qk_points * 3
        self.linear_k_points = Linear(self.c_s, hpk)

        if self.geometric_input:
            self.linear_v_g = Linear(self.c_s, self.no_v_points * 16)
            self.merge_geometric = EquiLinear(2 * self.no_v_points, self.no_heads * self.no_v_points)
        else:
            self.linear_v_g = Linear(self.c_s, self.no_heads * self.no_v_points * 16)

        self.merge_rel = EquiLinear(self.no_v_points + 1, self.no_v_points)

        self.bilinear_v = PairwiseGeometricBilinear(
            self.no_v_points, self.no_v_points
        )

        self.mbc_v = GMBCLayer(self.no_heads * self.no_v_points)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((gfa_conf.no_heads)))
        gfa_point_weights_init_(self.head_weights)

        self.softmax_weights = nn.Parameter(torch.ones((gfa_conf.no_heads)))

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + 8 + self.no_v_points * 18 # 16 + 1 + 1
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        if self.geometric_output:
            self.geometric_out = EquiLinear(self.no_heads * self.no_v_points, self.no_v_points)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        self.point_trafo = point_trafo()
        self.ga_norm = ga_norm()
        self.ga_versor = ga_versor()
        self.ga_inverse_versor = ga_inverse_versor()
        self.local_rel_trafo = local_rel_trafo()

    def forward(
        self,
        s: torch.Tensor,
        g: torch.Tensor,
        z: Optional[torch.Tensor],
        T: torch.Tensor,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            T:
                [*, N_res, 16] versors representing the frames
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]
        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        T8 = pu.extract_motor(T)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)

        q_pts = self.point_trafo(q_pts, T8.unsqueeze(-2))

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 4)
        )

        # [*, N_res, H * P_q * 3]
        k_pts = self.linear_k_points(s)

        # [*, N_res, H * P_q, 3]
        k_pts = torch.split(k_pts, k_pts.shape[-1] // 3, dim=-1)
        k_pts = torch.stack(k_pts, dim=-1)

        k_pts = self.point_trafo(k_pts, T8.unsqueeze(-2))

        # [*, N_res, H, P_q, 3]
        k_pts = k_pts.view(k_pts.shape[:-2] + (self.no_heads, -1, 4))

        v_g = self.linear_v_g(s) 

        if self.geometric_input:
            v_g = v_g.view(*v_g.shape[:-1], self.no_v_points, 16)
            v_g, _ = self.merge_geometric(torch.cat([v_g, g], dim=-2))
            v_g = v_g.view(*v_g.shape[:-2], self.no_heads, self.no_v_points, 16)
        else:
            v_g = v_g.view(*v_g.shape[:-1], self.no_heads, self.no_v_points, 16)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])
        
        if(_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4)[..., :3] - k_pts.unsqueeze(-5)[..., :3]
        pt_att = pt_displacement ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = torch.sum(pt_att, dim=-1)

        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )

        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        
        softmax_weights = self.softmax_weights.view(1, -1, 1, 1)

        a = a + pt_att 
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a * softmax_weights)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        o_pt = self.ga_versor(v_g, T.unsqueeze(-2).unsqueeze(-3))

        o_pt = torch.sum(
        (
            a[..., None, :, :, None]
            * permute_final_dims(o_pt, (1, 3, 0, 2))[..., None, :, :]
        ),
        dim=-2,
        )
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))

        o_pt = self.ga_inverse_versor(o_pt, T.unsqueeze(-2).unsqueeze(-3))

        v_rel = self.local_rel_trafo(T)
        v_rel = torch.matmul(a.transpose(-2, -3), v_rel)

        o_pt, _ = self.merge_rel(torch.cat([o_pt, v_rel.unsqueeze(-2)], dim=-2))

        o_pt = self.bilinear_v(v_g, o_pt)

        # [*, N_res, H * P_v, 16]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 16)

        o_pt = self.mbc_v(o_pt)

        post_mbc_norm, post_mbc_infnorm = self.ga_norm(o_pt)

        if(_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, N_res, C_z // 4]
        pair_z = self.down_z(z[0])
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        v_rel  = extract_motor(v_rel)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), *torch.unbind(v_rel, dim=-1), post_mbc_norm, post_mbc_infnorm, o_pair]

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            )
        )

        if self.geometric_output:
            g, _ = self.geometric_out(o_pt)
        else:
            g = None
        
        return s, g, v_rel

class BackboneUpdate(nn.Module):


    def __init__(self, c_s, c_g, c_T):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s
        self.c_g = c_g
        self.c_T = c_T

        self.linear_hidden = Linear(self.c_s + 16 * self.c_g + 8 * self.c_T, 64)
        self.gelu = nn.GELU()

        # Choose initialization such that t=0 and R=1, i.e. the identiy map as update
        self.linear_final = Linear(64, 6, init="final")

    def forward(self, s: torch.Tensor, g: torch.Tensor, T_rel:torch.Tensor, T: torch.Tensor, R: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            T:
                [*, N_res, c_T, 8] motors representing the frames
            g: 
                [*, N_res, c_g, 16] versors representing the frames
        Returns:
            [*, N_res, 6] update vector 
        """
        T_rel = T_rel.reshape(*T_rel.shape[:-2], -1)
        g = g.reshape(*g.shape[:-2], -1)

        in_ = torch.cat([s, g, T_rel], dim=-1)
        if mask is not None:
            in_ = in_ * mask

        update = self.linear_hidden(in_)
        update = self.gelu(update)
        update = self.linear_final(update)

        # Predict new frames both in the LA and PGA formulation as this avoids unneccessary conversions which would lead to numerical instabilities and nan losses
        new_rigid = R.compose_q_update_vec(update, mask)
        new_frame = embed_quat_frames(new_rigid.get_rots().get_quats(), new_rigid.get_trans())

        return new_frame, new_rigid
