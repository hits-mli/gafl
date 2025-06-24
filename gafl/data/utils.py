# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Any
from openfold.utils import rigid_utils as ru
import numpy as np
import collections
import string
import pickle
import os
import torch
import mdtraj as md
from torch_scatter import scatter_add, scatter
from Bio.PDB.Chain import Chain
import dataclasses
from Bio import PDB
import biotite.structure as struc
import io
import pandas as pd
from pathlib import Path
import warnings

from gafl.data import protein
from gafl.data import residue_constants


Rigid = ru.Rigid
Protein = protein.Protein

PICKLE_EXTENSIONS = ['.pkl', '.pickle', '.pck', '.db', '.pck']

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
	chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
	i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

CHAIN_FEATS = [
	'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'b_factors'
]

to_numpy = lambda x: x.detach().cpu().numpy()
aatype_to_seq = lambda aatype: ''.join([
		residue_constants.restypes_with_x[x] for x in aatype])


class CPU_Unpickler(pickle.Unpickler):
	"""Pytorch pickle loading workaround.

	https://github.com/pytorch/pytorch/issues/16797
	"""
	def find_class(self, module, name):
		if module == 'torch.storage' and name == '_load_from_bytes':
			return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
		else: return super().find_class(module, name)


def create_rigid(rots, trans):
	rots = ru.Rotation(rot_mats=rots)
	return Rigid(rots=rots, trans=trans)


def batch_align_structures(pos_1, pos_2, mask=None):
	if pos_1.shape != pos_2.shape:
		raise ValueError('pos_1 and pos_2 must have the same shape.')
	if pos_1.ndim != 3:
		raise ValueError(f'Expected inputs to have shape [B, N, 3]')
	num_batch = pos_1.shape[0]
	device = pos_1.device
	batch_indices = (
		torch.ones(*pos_1.shape[:2], device=device, dtype=torch.int64) 
		* torch.arange(num_batch, device=device)[:, None]
	)
	flat_pos_1 = pos_1.reshape(-1, 3)
	flat_pos_2 = pos_2.reshape(-1, 3)
	flat_batch_indices = batch_indices.reshape(-1)
	if mask is None:
		aligned_pos_1, aligned_pos_2, align_rots = align_structures(
			flat_pos_1, flat_batch_indices, flat_pos_2)
		aligned_pos_1 = aligned_pos_1.reshape(num_batch, -1, 3)
		aligned_pos_2 = aligned_pos_2.reshape(num_batch, -1, 3)
		return aligned_pos_1, aligned_pos_2, align_rots

	flat_mask = mask.reshape(-1).bool()
	_, _, align_rots = align_structures(
		flat_pos_1[flat_mask],
		flat_batch_indices[flat_mask],
		flat_pos_2[flat_mask]
	)
	aligned_pos_1 = torch.bmm(
		pos_1,
		align_rots
	)
	return aligned_pos_1, pos_2, align_rots


def adjust_oxygen_pos(
	atom_37: torch.Tensor, pos_is_known = None
) -> torch.Tensor:
	"""
	Imputes the position of the oxygen atom on the backbone by using adjacent frame information.
	Specifically, we say that the oxygen atom is in the plane created by the Calpha and C from the
	current frame and the nitrogen of the next frame. The oxygen is then placed c_o_bond_length Angstrom
	away from the C in the current frame in the direction away from the Ca-C-N triangle.

	For cases where the next frame is not available, for example we are at the C-terminus or the
	next frame is not available in the data then we place the oxygen in the same plane as the
	N-Ca-C of the current frame and pointing in the same direction as the average of the
	Ca->C and Ca->N vectors.

	Args:
		atom_37 (torch.Tensor): (N, 37, 3) tensor of positions of the backbone atoms in atom_37 ordering
								which is ['N', 'CA', 'C', 'CB', 'O', ...]
		pos_is_known (torch.Tensor): (N,) mask for known residues.
	"""

	N = atom_37.shape[0]
	assert atom_37.shape == (N, 37, 3)

	# Get vectors to Carbonly from Carbon alpha and N of next residue. (N-1, 3)
	# Note that the (N,) ordering is from N-terminal to C-terminal.

	# Calpha to carbonyl both in the current frame.
	calpha_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[:-1, 1, :]) / (
		torch.norm(atom_37[:-1, 2, :] - atom_37[:-1, 1, :], keepdim=True, dim=1) + 1e-7
	)
	# For masked positions, they are all 0 and so we add 1e-7 to avoid division by 0.
	# The positions are in Angstroms and so are on the order ~1 so 1e-7 is an insignificant change.

	# Nitrogen of the next frame to carbonyl of the current frame.
	nitrogen_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[1:, 0, :]) / (
		torch.norm(atom_37[:-1, 2, :] - atom_37[1:, 0, :], keepdim=True, dim=1) + 1e-7
	)

	carbonyl_to_oxygen: torch.Tensor = calpha_to_carbonyl + nitrogen_to_carbonyl  # (N-1, 3)
	carbonyl_to_oxygen = carbonyl_to_oxygen / (
		torch.norm(carbonyl_to_oxygen, dim=1, keepdim=True) + 1e-7
	)

	atom_37[:-1, 4, :] = atom_37[:-1, 2, :] + carbonyl_to_oxygen * 1.23

	# Now we deal with frames for which there is no next frame available.

	# Calpha to carbonyl both in the current frame. (N, 3)
	calpha_to_carbonyl_term: torch.Tensor = (atom_37[:, 2, :] - atom_37[:, 1, :]) / (
		torch.norm(atom_37[:, 2, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-7
	)
	# Calpha to nitrogen both in the current frame. (N, 3)
	calpha_to_nitrogen_term: torch.Tensor = (atom_37[:, 0, :] - atom_37[:, 1, :]) / (
		torch.norm(atom_37[:, 0, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-7
	)
	carbonyl_to_oxygen_term: torch.Tensor = (
		calpha_to_carbonyl_term + calpha_to_nitrogen_term
	)  # (N, 3)
	carbonyl_to_oxygen_term = carbonyl_to_oxygen_term / (
		torch.norm(carbonyl_to_oxygen_term, dim=1, keepdim=True) + 1e-7
	)

	# Create a mask that is 1 when the next residue is not available either
	# due to this frame being the C-terminus or the next residue is not
	# known due to pos_is_known being false.

	if pos_is_known is None:
		pos_is_known = torch.ones((atom_37.shape[0],), dtype=torch.int64, device=atom_37.device)

	next_res_gone: torch.Tensor = ~pos_is_known.bool()  # (N,)
	next_res_gone = torch.cat(
		[next_res_gone, torch.ones((1,), device=pos_is_known.device).bool()], dim=0
	)  # (N+1, )
	next_res_gone = next_res_gone[1:]  # (N,)

	atom_37[next_res_gone, 4, :] = (
		atom_37[next_res_gone, 2, :]
		+ carbonyl_to_oxygen_term[next_res_gone, :] * 1.23
	)

	return atom_37


def write_pkl(
		save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
	"""Serialize data into a pickle file."""
	if create_dir:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
	if use_torch:
		torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
	else:
		with open(save_path, 'wb') as handle:
			pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
	"""Read data from a pickle file."""
	try:
		if use_torch:
			return torch.load(read_path, map_location=map_location)
		else:
			with open(read_path, 'rb') as handle:
				return pickle.load(handle)
	except Exception as e:
		try:
			with open(read_path, 'rb') as handle:
				return CPU_Unpickler(handle).load()
		except Exception as e2:
			if verbose:
				print(f'Failed to read {read_path}. First error: {e}\n Second error: {e2}')
			raise(e)
	  
def read_npz(npz:dict):
	"""
	Converts dict of np arrays to a biotite AtomArray object.
	Returns: a struc.AtomArray (biotite) object
	"""

	coords = npz['coords']
	chain_id = npz['chain_id']
	res_id = npz['res_id']
	res_name = npz['res_name']
	atom_name = npz['atom_name']
	element = npz['element']

	if 'b_factor' in npz and npz['b_factor'].shape !=(0,):
		b_factor = npz['b_factor']
	else:
		b_factor = np.zeros(coords.shape[0])

	# some shape checks:
	num_bb_atoms = coords.shape[0]

	assert chain_id.shape[0] == num_bb_atoms, f'chain_id shape {chain_id.shape} does not match coords shape {coords.shape}'
	assert res_id.shape[0] == num_bb_atoms, f'res_id shape {res_id.shape} does not match coords shape {coords.shape}'
	assert res_name.shape[0] == num_bb_atoms, f'res_name shape {res_name.shape} does not match coords shape {coords.shape}'
	assert atom_name.shape[0] == num_bb_atoms, f'atom_name shape {atom_name.shape} does not match coords shape {coords.shape}'
	assert element.shape[0] == num_bb_atoms, f'element shape {element.shape} does not match coords shape {coords.shape}'
	assert b_factor.shape[0] == num_bb_atoms, f'b_factor shape {b_factor.shape} does not match coords shape {coords.shape}'

	# construct a biotite AtomArray
	chain_feats = struc.AtomArray(len(coords))
	chain_feats.coord = coords
	chain_feats.chain_id = chain_id
	chain_feats.res_id = res_id
	chain_feats.res_name = res_name
	chain_feats.atom_name = atom_name
	chain_feats.element = element
	chain_feats.b_factor = b_factor
	return chain_feats

def chain_str_to_int(chain_str: str):
	chain_int = 0
	if len(chain_str) == 1:
		return CHAIN_TO_INT[chain_str]
	for i, chain_char in enumerate(chain_str):
		chain_int += CHAIN_TO_INT[chain_char] + (i * len(ALPHANUMERIC))
	return chain_int


def parse_chain_feats(chain_feats, scale_factor=1.):
	ca_idx = residue_constants.atom_order['CA']
	chain_feats['bb_mask'] = chain_feats['atom_mask'][:, ca_idx]
	bb_pos = chain_feats['atom_positions'][:, ca_idx]
	bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats['bb_mask']) + 1e-5)
	centered_pos = chain_feats['atom_positions'] - bb_center[None, None, :]
	scaled_pos = centered_pos / scale_factor
	chain_feats['atom_positions'] = scaled_pos * chain_feats['atom_mask'][..., None]
	chain_feats['bb_positions'] = chain_feats['atom_positions'][:, ca_idx]
	return chain_feats


def concat_np_features(
		np_dicts: List[Dict[str, np.ndarray]], add_batch_dim: bool):
	"""Performs a nested concatenation of feature dicts.

	Args:
		np_dicts: list of dicts with the same structure.
			Each dict must have the same keys and numpy arrays as the values.
		add_batch_dim: whether to add a batch dimension to each feature.

	Returns:
		A single dict with all the features concatenated.
	"""
	combined_dict = collections.defaultdict(list)
	for chain_dict in np_dicts:
		for feat_name, feat_val in chain_dict.items():
			if add_batch_dim:
				feat_val = feat_val[None]
			combined_dict[feat_name].append(feat_val)
	# Concatenate each feature
	for feat_name, feat_vals in combined_dict.items():
		combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
	return combined_dict


def center_zero(pos: torch.Tensor, batch_indexes: torch.LongTensor) -> torch.Tensor:
	"""
	Move the molecule center to zero for sparse position tensors.

	Args:
		pos: [N, 3] batch positions of atoms in the molecule in sparse batch format.
		batch_indexes: [N] batch index for each atom in sparse batch format.

	Returns:
		pos: [N, 3] zero-centered batch positions of atoms in the molecule in sparse batch format.
	"""
	assert len(pos.shape) == 2 and pos.shape[-1] == 3, "pos must have shape [N, 3]"

	means = scatter(pos, batch_indexes, dim=0, reduce="mean")
	return pos - means[batch_indexes]


@torch.no_grad()
def align_structures(
	batch_positions: torch.Tensor,
	batch_indices: torch.Tensor,
	reference_positions: torch.Tensor,
	broadcast_reference: bool = False,
):
	"""
	Align structures in a ChemGraph batch to a reference, e.g. for RMSD computation. This uses the
	sparse formulation of pytorch geometric. If the ChemGraph is composed of a single system, then
	the reference can be given as a single structure and broadcasted. Returns the structure
	coordinates shifted to the geometric center and the batch structures rotated to match the
	reference structures. Uses the Kabsch algorithm (see e.g. [kabsch_align1]_). No permutation of
	atoms is carried out.

	Args:
		batch_positions (Tensor): Batch of structures (e.g. from ChemGraph) which should be aligned
		  to a reference.
		batch_indices (Tensor): Index tensor mapping each node / atom in batch to the respective
		  system (e.g. batch attribute of ChemGraph batch).
		reference_positions (Tensor): Reference structure. Can either be a batch of structures or a
		  single structure. In the second case, broadcasting is possible if the input batch is
		  composed exclusively of this structure.
		broadcast_reference (bool, optional): If reference batch contains only a single structure,
		  broadcast this structure to match the ChemGraph batch. Defaults to False.

	Returns:
		Tuple[torch.Tensor, torch.Tensor]: Tensors containing the centered positions of batch
		  structures rotated into the reference and the centered reference batch.

	References
	----------
	.. [kabsch_align1] Lawrence, Bernal, Witzgall:
	   A purely algebraic justification of the Kabsch-Umeyama algorithm.
	   Journal of research of the National Institute of Standards and Technology, 124, 1. 2019.
	"""
	# Minimize || Q @ R.T - P ||, which is the same as || Q - P @ R ||
	# batch_positions     -> P [BN x 3]
	# reference_positions -> Q [B / BN x 3]

	if batch_positions.shape[0] != reference_positions.shape[0]:
		if broadcast_reference:
			# Get number of systems in batch and broadcast reference structure.
			# This assumes, all systems in the current batch correspond to the reference system.
			# Typically always the case during evaluation.
			num_molecules = int(torch.max(batch_indices) + 1)
			reference_positions = reference_positions.repeat(num_molecules, 1)
		else:
			raise ValueError("Mismatch in batch dimensions.")

	# Center structures at origin (takes care of translation alignment)
	batch_positions = center_zero(batch_positions, batch_indices)
	reference_positions = center_zero(reference_positions, batch_indices)

	# Compute covariance matrix for optimal rotation (Q.T @ P) -> [B x 3 x 3].
	cov = scatter_add(
		batch_positions[:, None, :] * reference_positions[:, :, None], batch_indices, dim=0
	)

	# Perform singular value decomposition. (all [B x 3 x 3])
	u, _, v_t = torch.linalg.svd(cov)
	# Convenience transposes.
	u_t = u.transpose(1, 2)
	v = v_t.transpose(1, 2)

	# Compute rotation matrix correction for ensuring right-handed coordinate system
	# For comparison with other sources: det(AB) = det(A)*det(B) and det(A) = det(A.T)
	sign_correction = torch.sign(torch.linalg.det(torch.bmm(v, u_t)))
	# Correct transpose of U: diag(1, 1, sign_correction) @ U.T
	u_t[:, 2, :] = u_t[:, 2, :] * sign_correction[:, None]

	# Compute optimal rotation matrix (R = V @ diag(1, 1, sign_correction) @ U.T).
	rotation_matrices = torch.bmm(v, u_t)

	# Rotate batch positions P to optimal alignment with Q (P @ R)
	batch_positions_rotated = torch.bmm(
		batch_positions[:, None, :],
		rotation_matrices[batch_indices],
	).squeeze(1)

	return batch_positions_rotated, reference_positions, rotation_matrices


def parse_pdb_feats(
		pdb_name: str,
		pdb_path: str,
		scale_factor=1.,
		# TODO: Make the default behaviour read all chains.
		chain_id='A',
		calc_dssp:bool=False
	):
	"""
	Args:
		pdb_name: name of PDB to parse.
		pdb_path: path to PDB file to read.
		scale_factor: factor to scale atom positions.
		mean_center: whether to mean center atom positions.
		calc_dssp: whether to compute DSSP features.
	Returns:
		Dict with CHAIN_FEATS features extracted from PDB with specified
		preprocessing.
	"""
	parser = PDB.PDBParser(QUIET=True)
	structure = parser.get_structure(pdb_name, pdb_path)
	struct_chains = {
		chain.id: chain
		for chain in structure.get_chains()}

	def _process_chain_id(x):
		if calc_dssp:
			chain_prot = process_chain(struct_chains[x], x, pdb_loc=pdb_path, calc_dssp=calc_dssp)
		else:
			chain_prot = process_chain(struct_chains[x], x)
		chain_dict = dataclasses.asdict(chain_prot)

		# Process features
		feat_dict = {x: chain_dict[x] for x in (CHAIN_FEATS + ['dssp'] if calc_dssp else CHAIN_FEATS)}
		return parse_chain_feats(
			feat_dict, scale_factor=scale_factor)

	if isinstance(chain_id, str):
		return _process_chain_id(chain_id)
	elif isinstance(chain_id, list):
		return {
			x: _process_chain_id(x) for x in chain_id
		}
	elif chain_id is None:
		return {
			x: _process_chain_id(x) for x in struct_chains
		}
	else:
		raise ValueError(f'Unrecognized chain list {chain_id}')
	
def parse_npz_feats(npz_feats:struc.AtomArray, 
					scale_factor=1.,
					calc_dssp:bool=False):
	'''
	 Args:
		npz_feats: struc.AtomArray extracted from a npz file from read_npz()
		scale_factor: factor to scale atom positions.
		calc_dssp: whether to compute DSSP features.
	Returns:
		Dict with CHAIN_FEATS features extracted from the compressed PDB with specified
		preprocessing.
	'''

	def _process_chain_id(npz_feats):
		chain_prot = process_chain_npz(npz_feats, dssp=calc_dssp)
		
		chain_dict = dataclasses.asdict(chain_prot)

		feat_dict = {x: chain_dict[x] for x in CHAIN_FEATS}
		return parse_chain_feats(feat_dict, 
								 scale_factor=scale_factor)

	if isinstance(npz_feats, struc.AtomArray):
		return _process_chain_id(npz_feats)
	elif isinstance(npz_feats, list):
		return {
			x: _process_chain_id(x) for x in npz_feats
		}
	elif npz_feats is None:
		return None
	else:
		raise ValueError("Invalid input type")

def rigid_transform_3D(A, B, verbose=False):
	# Transforms A to look like B
	# https://github.com/nghiaho12/rigid_transform_3D
	assert A.shape == B.shape
	A = A.T
	B = B.T

	num_rows, num_cols = A.shape
	if num_rows != 3:
		raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

	num_rows, num_cols = B.shape
	if num_rows != 3:
		raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

	# find mean column wise
	centroid_A = np.mean(A, axis=1)
	centroid_B = np.mean(B, axis=1)

	# ensure centroids are 3x1
	centroid_A = centroid_A.reshape(-1, 1)
	centroid_B = centroid_B.reshape(-1, 1)

	# subtract mean
	Am = A - centroid_A
	Bm = B - centroid_B

	H = Am @ np.transpose(Bm)

	# sanity check
	#if linalg.matrix_rank(H) < 3:
	#    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

	# find rotation
	U, S, Vt = np.linalg.svd(H)
	R = Vt.T @ U.T
	
	# special reflection case
	reflection_detected = False
	if np.linalg.det(R) < 0:
		if verbose:
			print("det(R) < R, reflection detected!, correcting for it ...")
		Vt[2,:] *= -1
		R = Vt.T @ U.T
		reflection_detected = True

	t = -R @ centroid_A + centroid_B
	optimal_A = R @ A + t

	return optimal_A.T, R, t, reflection_detected


def process_chain(chain: Chain, chain_id: str, pdb_loc=None, calc_dssp=None) -> Protein:
	"""Convert a PDB chain object into a AlphaFold Protein instance.

	Forked from alphafold.common.protein.from_pdb_string

	WARNING: All non-standard residue types will be converted into UNK. All
		non-standard atoms will be ignored.

	Took out lines 94-97 which don't allow insertions in the PDB.
	Sabdab uses insertions for the chothia numbering so we need to allow them.

	Took out lines 110-112 since that would mess up CDR numbering.
	
	Addded: pdb_loc:str and calc_dssp:bool to compute the dssp

	Args:
		chain: Instance of Biopython's chain class.

	Returns:
		Protein object with protein features.
	"""
	atom_positions = []
	aatype = []
	atom_mask = []
	residue_index = []
	b_factors = []
	chain_ids = []
	# modified here
	if calc_dssp:
		assert pdb_loc is not None, 'pdb_loc must be provided to compute dssp'
		t = md.load(pdb_loc) # only possible to compute for the refolded designs since they contain [N, CA, C] needed for dihedrals
		dssp = md.compute_dssp(t) 
		dssp = dssp[0]
	else:
		dssp = None

	for res in chain:
		res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
		restype_idx = residue_constants.restype_order.get(
			res_shortname, residue_constants.restype_num)
		pos = np.zeros((residue_constants.atom_type_num, 3))
		mask = np.zeros((residue_constants.atom_type_num,))
		res_b_factors = np.zeros((residue_constants.atom_type_num,))
		for atom in res:
			if atom.name not in residue_constants.atom_types:
				continue
			pos[residue_constants.atom_order[atom.name]] = atom.coord
			mask[residue_constants.atom_order[atom.name]] = 1.
			res_b_factors[residue_constants.atom_order[atom.name]
						  ] = atom.bfactor
		aatype.append(restype_idx)
		atom_positions.append(pos)
		atom_mask.append(mask)
		residue_index.append(res.id[1])
		b_factors.append(res_b_factors)
		chain_ids.append(chain_id)

	return Protein(
		atom_positions=np.array(atom_positions),
		atom_mask=np.array(atom_mask),
		aatype=np.array(aatype),
		residue_index=np.array(residue_index),
		chain_index=np.array(chain_ids),
		b_factors=np.array(b_factors),
		dssp=dssp)

def process_chain_npz(npz_feats: struc.AtomArray, dssp=None) -> Protein:
	'''
	npz_feats: struc.AtomArray extracted from a npz file from read_npz()
	dssp: None, cannot be computed for the feats extracted from .npz without writing creating a .pdb file
	'''
	residues = npz_feats.res_name[::3]
	atoms = npz_feats.atom_name
	coords = npz_feats.coord
	bfac = npz_feats.b_factor
	chain_id = npz_feats.chain_id
	res_ids = npz_feats.res_id[::3]

	if len(bfac) == 0: 
    		bfac = np.zeros_like(atoms, dtype=np.float32)
	
	atom_positions = []
	aatype = []
	atom_mask = []
	residue_index = []
	b_factors = []
	chain_ids = []

	assert len(residues) > 0, 'No residues found in the npz file'

	for i in range(len(residues)):

		res_shortname = residue_constants.restype_3to1.get(residues[i], 'X')
		restype_idx = residue_constants.restype_order.get(
		res_shortname, residue_constants.restype_num)
		pos = np.zeros((residue_constants.atom_type_num, 3))
		mask = np.zeros((residue_constants.atom_type_num,))
		res_b_factors = np.zeros((residue_constants.atom_type_num,))

		for index in range(len(atoms[i*3:(i+1)*3])):
			if atoms[index] not in residue_constants.atom_types:
				continue
			pos[residue_constants.atom_order[atoms[index]]] = coords[i*3:(i+1)*3][index]
			mask[residue_constants.atom_order[atoms[index]]] = 1.
			res_b_factors[residue_constants.atom_order[atoms[index]]
						] = bfac[i*3:(i+1)*3][index]
			
		aatype.append(restype_idx)
		atom_positions.append(pos)
		atom_mask.append(mask)
		residue_index.append(res_ids[i])
		b_factors.append(res_b_factors)
		chain_ids.append(chain_id[i])
	
	return Protein(
		atom_positions=np.array(atom_positions),
		atom_mask=np.array(atom_mask),
		aatype=np.array(aatype),
		residue_index=np.array(residue_index),
		chain_index=np.array(chain_ids),
		b_factors=np.array(b_factors), 
		dssp=dssp)


def metadata_naming_convention(df: pd.DataFrame)->pd.DataFrame:
	'''
	Renames certain cols in the dataframe to match the naming convention of metadata files
	'''

	# dictionary that maps conventional col name to list of possible col names in the dataframe
	RENAME_DICT = {
		"pdb_name": ["name_pdb",],
		"processed_path": ["location",],
		"breaks": ["break",],
		"coil_percent": ["pct_coil",],
		"helix_percent": ["pct_helix",],
		"strand_percent": ["pct_strand",],
		"seq_len": ["length",],
	}

	for col, possible_names in RENAME_DICT.items():
		for name in possible_names:
			if name in df.columns:
				df.rename(columns={name: col}, inplace=True)
				break

	# for length -> modeled_seq_len, we also need to check that there are no breaks in the proteins:
	if "modeled_seq_len" not in df.columns:
		if "seq_len" in df.columns:
			assert "breaks" in df.columns, "breaks must be in the dataframe to compute modeled_seq_len"
			# assert the entry in breaks is always false:
			# assert df["breaks"].apply(lambda x: x == False).all(), "breaks must be False for all entries"
			df["modeled_seq_len"] = df["seq_len"]
		else:
			raise ValueError("modeled_seq_len or length must be in the dataframe")
		
	return df


# some filtering utils:
def has_breaks(chain_feats:dict, max_ca_ca_distance=4.5):
	'''
	Checks if the entry of PdbDataset has breaks in the backbone based on the ca ca distance (in angstrom, canonical distance is 3.8, 4.5 includes some tolerance).
	'''
	# check ca_ca distance:
	xyz = chain_feats['trans_1']
	if isinstance(xyz, torch.Tensor):
		xyz = xyz.detach().cpu().numpy()
	elif isinstance(xyz, list):
		xyz = np.array(xyz)
	elif not isinstance(xyz, np.ndarray):
		raise ValueError(f"xyz must be a torch.Tensor, list or np.ndarray, not {type(xyz)}")
	
	# calculate distances between subsequent CA atoms:
	subsequent_dists = np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)
	if not np.all(subsequent_dists < max_ca_ca_distance):
		return True
	else:
		return False
	
def has_inconstistent_indexing(chain_feats:dict):
	'''
	Checks if the entry of PdbDataset has inconsistent indexing of the residues.
	'''
	res_idxs = chain_feats['res_idx']
	if isinstance(res_idxs, torch.Tensor):
		res_idxs = res_idxs.detach().cpu().numpy()
	elif isinstance(res_idxs, list):
		res_idxs = np.array(res_idxs)
	elif not isinstance(res_idxs, np.ndarray):
		raise ValueError(f"res_idxs must be a torch.Tensor, list or np.ndarray, not {type(res_idxs)}")

	num_res = len(res_idxs)
	# check whether idxs are continuous:
	if not np.array_equal(res_idxs, np.arange(num_res)+1):
		return True
	else:
		return False
	
def dist_breaks(chain_feats:dict, max_ca_ca_distance=4.5):
	'''
	Returns the indices of the residues where the ca-ca distance is greater than max_ca_ca_distance
	'''
	# check ca_ca distance:
	xyz = chain_feats['trans_1']
	if isinstance(xyz, torch.Tensor):
		xyz = xyz.detach().cpu().numpy()
	elif isinstance(xyz, list):
		xyz = np.array(xyz)
	elif not isinstance(xyz, np.ndarray):
		raise ValueError(f"xyz must be a torch.Tensor, list or np.ndarray, not {type(xyz)}")
	
	# calculate distances between subsequent CA atoms:
	subsequent_dists = np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)
	return np.where(subsequent_dists >= max_ca_ca_distance)[0]
	
def idx_breaks(chain_feats:dict):
	'''
	Returns the indices of the residues where the res_idx is not continuous
	'''
	res_idxs = chain_feats['res_idx']
	if isinstance(res_idxs, torch.Tensor):
		res_idxs = res_idxs.detach().cpu().numpy()
	elif isinstance(res_idxs, list):
		res_idxs = np.array(res_idxs)
	elif not isinstance(res_idxs, np.ndarray):
		raise ValueError(f"res_idxs must be a torch.Tensor, list or np.ndarray, not {type(res_idxs)}")

	return np.where(res_idxs[1:] - np.roll(res_idxs, 1)[1:] != 1)[0]


def get_processed_feats(path, extra_feats:List[str]=None):
	"""
	Reads the processed features from a file. The file can be either a pickle or npz file.
	"""
	path_extension = Path(path).suffix

	if path_extension in PICKLE_EXTENSIONS:
		processed_feats = read_pkl(path)
		processed_feats = parse_chain_feats(processed_feats)
		
	elif path_extension == '.npz':
		npz_dict = dict(np.load(path))
		biotite_struct = read_npz(npz_dict)
		processed_feats = parse_npz_feats(npz_feats=biotite_struct)
		processed_feats['modeled_idx'] = processed_feats['residue_index']
		if extra_feats is not None:
			num_bb_atoms = processed_feats['atom_positions'].shape[0]
			for k in npz_dict.keys():
				if k not in processed_feats.keys() and k in extra_feats:

					# check shape:
					if npz_dict[k].shape[0] == 3*num_bb_atoms:
						# only use every third atom:
						npz_dict[k] = npz_dict[k][::3]
						warnings.warn(f"Shape of {k} was 3 times the atom_positions shape, using every third atom")
					elif npz_dict[k].shape[0] != num_bb_atoms:
						raise ValueError(f"Shape of {k} does not match atom_positions shape")
				
					processed_feats[k] = npz_dict[k]
	else:
		raise ValueError(f'Unknown file extension {path_extension}')
	

	
	return processed_feats
