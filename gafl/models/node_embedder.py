# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Neural network for embedding node features."""
import torch
from torch import nn

from gafl.models.utils import get_index_embedding, get_time_embedding, get_length_embedding


class NodeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(NodeEmbedder, self).__init__()
        self._cfg = module_cfg
                
        if "permutation_equivariance" in self._cfg:
            self.permutation_equivariance = self._cfg.permutation_equivariance
        else:
            self.permutation_equivariance = False

        if "total_length_emb_dim" in self._cfg:
            self.total_length_emb_dim = self._cfg.total_length_emb_dim
        else:
            self.total_length_emb_dim = 0

        if "embed_breaks" in self._cfg:
            self.embed_breaks = self._cfg.embed_breaks
        else:
            self.embed_breaks = False
        
        if self.embed_breaks:
            self.c_breaks = 2
        else:
            self.c_breaks = 0

        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.linear = nn.Linear(
            self._cfg.c_pos_emb + self._cfg.c_timestep_emb, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, timesteps, mask, res_idx=None, breaks=None):
        # s: [b]

        b, num_res, device = mask.shape[0], mask.shape[1], mask.device

        if res_idx is None:
            pos = torch.arange(num_res, dtype=torch.float32).to(device).unsqueeze(0)
        else:
            pos = res_idx.float()

        if self.permutation_equivariance:
            # NOTE: remove positional encoding, except for first and last residue
            pos[:, 1:-1] = torch.ones_like(pos[:, 1:-1])
            # set the first residue to 0 and the last to 2, then multiply by half of max len
            pos[:, 0] = 0
            pos[:, -1] = 2
            pos = pos * 1028

        pos_emb = get_index_embedding(
            pos, self.c_pos_emb-self.total_length_emb_dim-self.c_breaks, max_len=2056
        )

        # [b, n_res, c_pos_emb]
        if res_idx is None:
            pos_emb = pos_emb.repeat([b, 1, 1])

        if self.embed_breaks:
            if breaks is None:
                breaks = torch.zeros([b, num_res], device=device, dtype=torch.float32)
            pos_emb = torch.cat([pos_emb, breaks.unsqueeze(-1), (1 - breaks).unsqueeze(-1)], dim=-1)

        if self.total_length_emb_dim > 0:
            length_embedding = get_length_embedding(pos, embed_size=self.total_length_emb_dim, max_len=2056)
            length_embedding = length_embedding.repeat([b, 1, 1])
            pos_emb = torch.cat([pos_emb, length_embedding], dim=-1)
        
        pos_emb = pos_emb * mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [pos_emb]
        # timesteps are between 0 and 1. Convert to integers.
        input_feats.append(self.embed_t(timesteps, mask))
        return self.linear(torch.cat(input_feats, dim=-1))