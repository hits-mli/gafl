# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import torch
from torch.nn import functional as F
import numpy as np
import importlib

from gafl.data import utils as du


def load_module(object):
    module, object = object.rsplit(".", 1)

    # backwards compatibility (there used to be no gafl package):
    if not module.startswith("gafl."):
        module = "gafl." + module

    module = importlib.import_module(module)
    fn = getattr(module, object)
    return fn

def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram

def log_bin_values(values, x_min, x_max, n_bins, device='cpu'):
    assert x_min > 1, "x_min must be greater than 1"
    assert x_max > x_min, "x_max must be greater than x_min"
    # assert torch.all(values >= x_min), "All values must be greater than or equal to x_min"
    # assert torch.all(values <= x_max), "All values must be less than or equal to x_max"
    
    # clip values to be within the range
    values_ = torch.clamp(values, x_min, x_max)

    # Generate logarithmic bins
    bins = torch.logspace(torch.log10(torch.tensor(x_min-1, device=device)),
                          torch.log10(torch.tensor(x_max, device=device)),
                          n_bins + 1, device=device)
    
    # Bucketize values into bins
    bin_indices = torch.bucketize(values_, bins, right=True) - 1
    
    return bin_indices

def get_length_embedding(indices, embed_size, max_len=2056):
    """
    Creates an embedding of total lengths using log-space binning.
    
    Args:
        indices: shape [..., N_res]
        embed_size: dimension of the embeddings to create
        max_len: maximum length.

    Returns:
        embedding of shape [..., N_res, embed_size]
    """

    # additional length embedding using one-hot encoding of binned lengths (logarithmic bins)
    ##############################
    num_res = indices.shape[-1]
    num_res = torch.tensor((num_res), device=indices.device).float()

    # log binning
    binned_values = log_bin_values(values=num_res, x_min=60, x_max=max_len, n_bins=embed_size, device=indices.device)
    binned_values = F.one_hot(binned_values, num_classes=embed_size).float()

    # Shape adjustment to match input shape
    expanded_shape = list(indices.shape) + [1]
    binned_values = binned_values.repeat(*expanded_shape)
    return binned_values

def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding

def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def t_stratified_loss(batch_t, batch_loss, num_bins=4, t_interval=[0.0, 1.0], loss_name=None):
    """Stratify loss by binning t."""
    batch_t = du.to_numpy(batch_t)
    batch_loss = du.to_numpy(batch_loss)
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(t_interval[0], t_interval[1] + 1e-3, num_bins+1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = 'loss'
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin+1]
        t_range = f'{loss_name} t=[{bin_start:.2f},{bin_end:.2f})'
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses    