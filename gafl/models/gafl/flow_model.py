# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Neural network architecture for the flow model."""
import torch
import torch.nn as nn

from gafl.models.node_embedder import NodeEmbedder
from gafl.models.edge_embedder import EdgeEmbedder
from gafl.models import ipa_pytorch
from gafl.data import utils as du
from gafl.models.gafl.pga_utils import embed_frames, EquiLayerNorm
from gafl.models.gafl.cfa import GeometricFrameAttention, Linear, BackboneUpdate

class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()

        self._model_conf = model_conf
        self._gfa_conf = model_conf.gfa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_embedder = NodeEmbedder(model_conf.node_features)
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._gfa_conf.num_blocks):
            self.trunk[f'gfa_{b}'] = GeometricFrameAttention(self._gfa_conf, geometric_input=bool(b), geometric_output=True)
            self.trunk[f'gfa_ln_{b}'] = nn.LayerNorm(self._gfa_conf.c_s)
            self.trunk[f'g_ln_{b}'] = EquiLayerNorm()
            tfmr_in = self._gfa_conf.c_s
            tfmr_layer = nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._gfa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = nn.TransformerEncoder(
                tfmr_layer, self._gfa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = Linear(
                tfmr_in, self._gfa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._gfa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(
                self._gfa_conf.c_s,
                self._gfa_conf.no_v_points,
                self._gfa_conf.no_heads,)

            if b < self._gfa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._gfa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

    def forward(self, input_feats):
        # During sampling res_idx is not provided
        if 'res_idx' not in input_feats:
            res_idx = None
        else:
            res_idx = input_feats['res_idx']

        if 'breaks' in input_feats:
            breaks = input_feats['breaks']
        else:
            breaks = None
        
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        continuous_t = input_feats['t']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']

        # Initialize node and edge embeddings
        init_node_embed = self.node_embedder(continuous_t, node_mask, res_idx=res_idx, breaks=breaks)
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']
        init_edge_embed = self.edge_embedder(
            init_node_embed, trans_t, trans_sc, edge_mask, res_idx=res_idx, breaks=breaks)

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t,)
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        curr_frames = embed_frames(rotmats_t, trans_t * du.ANG_TO_NM_SCALE)

        # Main trunk
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        g = None
        for b in range(self._gfa_conf.num_blocks):
            gfa_embed, g_gfa, gfa_rel = self.trunk[f'gfa_{b}'](
                node_embed,
                g,
                edge_embed,
                curr_frames,
                node_mask)
            gfa_embed *= node_mask[..., None]
            gfa_rel = gfa_rel * node_mask[..., None, None]
            node_embed = self.trunk[f'gfa_ln_{b}'](node_embed + gfa_embed)
            
            if g is not None:
                g = self.trunk[f'g_ln_{b}'](g + g_gfa)
            else:
                g = self.trunk[f'g_ln_{b}'](g_gfa)
            g =  g * node_mask[..., None, None]
                
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)    
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            curr_frames, curr_rigids = self.trunk[f'bb_update_{b}'](
                node_embed,
                g,
                gfa_rel,
                curr_frames,
                curr_rigids,
                node_mask[..., None],)

            if b < self._gfa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()

        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
        }