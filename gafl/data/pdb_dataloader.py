# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""PDB data loader."""
import math
import torch
import tree
import numpy as np
import torch
import pandas as pd
import ast
import logging
from tqdm import tqdm

from openfold.data import data_transforms
from openfold.utils import rigid_utils

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler, dist
from omegaconf import OmegaConf
from pathlib import Path
import os

from gafl.data import utils as du
from gafl.analysis import utils as au
from gafl.analysis.metrics import calc_mdtraj_metrics

class PdbDataModule(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.dataset_cfg = data_cfg.dataset

        OmegaConf.set_struct(self.dataset_cfg, False)
        # Handle missing train_valid_test_split argument
        if not hasattr(self.dataset_cfg, 'train_valid_test_split'):
            self.dataset_cfg.train_valid_test_split = [1.0, 0.0, 0.0]
        else:
            assert sum(self.dataset_cfg.train_valid_test_split) == 1, 'train_valid_test_split must sum to 1'
            assert len(self.dataset_cfg.train_valid_test_split) == 3, 'train_valid_test_split must have 3 elements'

        # Handle missing generate_valid_samples argument
        if not hasattr(self.dataset_cfg, 'generate_valid_samples'):
            self.dataset_cfg.generate_valid_samples = True

        if not hasattr(self.dataset_cfg, 'calc_dssp'):
            self.dataset_cfg.calc_dssp = False

        if not hasattr(self.dataset_cfg, 'target_sec_content'):
            self.dataset_cfg.target_sec_content = OmegaConf.create()
            self.dataset_cfg.target_sec_content.helix_percent = 0.32
            self.dataset_cfg.target_sec_content.strand_percent = 0.27
        else:
            if not hasattr(self.dataset_cfg.target_sec_content, 'helix_percent'):
                self.dataset_cfg.target_sec_content.helix_percent = 0.32
            if not hasattr(self.dataset_cfg.target_sec_content, 'strand_percent'):
                self.dataset_cfg.target_sec_content.strand_percent = 0.27

        OmegaConf.set_struct(self.dataset_cfg, True)
            
        self.sampler_cfg = data_cfg.sampler
        OmegaConf.set_struct(self.sampler_cfg, False)
        if not hasattr(self.sampler_cfg, 'clustered'):
            self.sampler_cfg.clustered = False
        OmegaConf.set_struct(self.sampler_cfg, True)

    def setup(self, stage: str):
        if self.dataset_cfg.calc_dssp:
            pdb_csv = pd.read_csv(self.dataset_cfg.csv_path)
            # Check if csc has columns helix_percent, strand_percent, coil_percent
            if not ('helix_percent' in pdb_csv.columns and 'strand_percent' in pdb_csv.columns):
                logging.info("Calculating DSSP values for dataset")
                #Iterate over csv column 'processed_path'
                helix_pct = []
                strand_pct = []
                coil_pct = []

                for path in tqdm(pdb_csv['processed_path']):
                    try:
                        processed_feats = du.get_processed_feats(path)
                        
                    except Exception as e:
                            raise ValueError(f'Error in processing {path}') from e
                    
                    atom_pos = processed_feats['atom_positions']
                    os.makedirs(os.path.join('tmp'), exist_ok=True)
                    au.write_prot_to_pdb(atom_pos, os.path.join('tmp', 'dssp_sample.pdb'), overwrite=True, no_indexing=True)

                    #Calculate dssp
                    dssp = calc_mdtraj_metrics(os.path.join('tmp', 'dssp_sample.pdb'))

                    helix_pct.append(dssp['helix_percent'])
                    strand_pct.append(dssp['strand_percent'])
                    coil_pct.append(dssp['coil_percent'])
            
                #Append dssp values to csv
                pdb_csv['helix_percent'] = helix_pct
                pdb_csv['strand_percent'] = strand_pct
                pdb_csv['coil_percent'] = coil_pct
                pdb_csv.to_csv(self.dataset_cfg.csv_path, index=False)

        self._train_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            is_training=True,
        )
        self._valid_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            is_training=False,
        )
        self._valid_sample_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            is_training=False,
            sample_dataset=True,
        )

    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        batch_sampler = LengthBatcher(
                sampler_cfg=self.sampler_cfg,
                metadata_csv=self._train_dataset.csv,
                rank=rank,
                num_replicas=num_replicas,
                num_batches=get_num_batches(self.sampler_cfg, self._train_dataset.csv),
            )
        dataloader = DataLoader(
            self._train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
        )

        return dataloader

    def val_dataloader(self):
        if self.dataset_cfg.generate_valid_samples:
            valid_sample_loader =  DataLoader(
                                self._valid_sample_dataset,
                                sampler=DistributedSampler(self._valid_sample_dataset, shuffle=False),
                                num_workers=2,
                                prefetch_factor=2,
                                persistent_workers=True,
                            )
        else:
            valid_sample_loader = DataLoader(EmptyDataset(), batch_size=1)
        
        if self.dataset_cfg.train_valid_test_split[1] == 0:
            valid_loader = DataLoader(EmptyDataset(), batch_size=1)
        else:
            num_workers = self.loader_cfg.num_workers
            valid_loader = DataLoader(
                                self._valid_dataset,
                                batch_sampler=LengthBatcher(
                                    sampler_cfg=self.sampler_cfg,
                                    metadata_csv=self._valid_dataset.csv,
                                    rank=None,
                                    num_replicas=None,
                                    num_batches=get_num_batches(self.sampler_cfg, self._valid_dataset.csv),
                                ),
                                num_workers=num_workers,
                                prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
                                pin_memory=False,
                                persistent_workers=True if num_workers > 0 else False,
                            )

        return [valid_sample_loader, valid_loader]

class PdbDataset(Dataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            sample_dataset=False, # Determines if validation dataset should be used as template for sampling backbones in the validation step
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self.sample_dataset = sample_dataset
        self._dataset_cfg = dataset_cfg

        if not hasattr(self._dataset_cfg, 'filter_breaks'):
            OmegaConf.set_struct(self._dataset_cfg, False)
            self._dataset_cfg.filter_breaks = False
            OmegaConf.set_struct(self._dataset_cfg, True)

        if not hasattr(self._dataset_cfg, 'label_breaks'):
            OmegaConf.set_struct(self._dataset_cfg, False)
            self._dataset_cfg.label_breaks = False
            OmegaConf.set_struct(self._dataset_cfg, True)

        if not hasattr(self._dataset_cfg, 'use_res_idx'):
            OmegaConf.set_struct(self._dataset_cfg, False)
            self._dataset_cfg.use_res_idx = False
            OmegaConf.set_struct(self._dataset_cfg, True)
                                 
        if not hasattr(self._dataset_cfg, 'break_csv_path'):
            OmegaConf.set_struct(self._dataset_cfg, False)
            self._dataset_cfg.break_csv_path = self.dataset_cfg.csv_path.replace('metadata.csv', 'breaks.csv')
            OmegaConf.set_struct(self._dataset_cfg, True)   
        
        if not hasattr(self._dataset_cfg, 'filter_scrmsd'):
            OmegaConf.set_struct(self._dataset_cfg, False)
            self._dataset_cfg.filter_scrmsd = "inf"
            OmegaConf.set_struct(self._dataset_cfg, True)

        if not hasattr(self._dataset_cfg, 'max_coil_pct'):
            OmegaConf.set_struct(self._dataset_cfg, False)
            self._dataset_cfg.max_coil_pct = 1.
            OmegaConf.set_struct(self._dataset_cfg, True)

        if not hasattr(self._dataset_cfg, 'min_num_res_eval'):
            OmegaConf.set_struct(self._dataset_cfg, False)
            self._dataset_cfg.min_num_res_eval = 60
            OmegaConf.set_struct(self._dataset_cfg, True)

        if not hasattr(self._dataset_cfg, 'allowed_oligomers'):
            OmegaConf.set_struct(self._dataset_cfg, False)
            self._dataset_cfg.allowed_oligomers = None
            OmegaConf.set_struct(self._dataset_cfg, True)

        if not hasattr(self._dataset_cfg, 'apply_clustering'):
            OmegaConf.set_struct(self._dataset_cfg, False)
            self._dataset_cfg.apply_clustering = False
            OmegaConf.set_struct(self._dataset_cfg, True)

        if not hasattr(self._dataset_cfg, 'target_sec_content'):
            OmegaConf.set_struct(self._dataset_cfg, False)
            self._dataset_cfg.target_sec_content = OmegaConf.create()
            self._dataset_cfg.target_sec_content.helix_percent = 0.32
            self._dataset_cfg.target_sec_content.strand_percent = 0.27
            OmegaConf.set_struct(self._dataset_cfg, True)


        self._init_metadata()
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg

    def _init_metadata(self):
        """Initialize metadata."""

        # Process CSV with different filtering criterions.
        pdb_csv = pd.read_csv(self.dataset_cfg.csv_path)

        # apply naming convention
        pdb_csv = du.metadata_naming_convention(pdb_csv)

        if self.dataset_cfg.apply_clustering:
            if hasattr(self.dataset_cfg, 'cluster_path_framediff'):

                # assume that the dataset type is framediff and read the clusters from the cluster file
                logging.info("Reading and clusters from framdiff-like-cluster file and storing to metadata...\nIf you want to use the cluster present in the metadata, set cluster_path_framediff to null.")
                pdb_to_cluster = _get_pdb_to_cluster_dict_framediff(self.dataset_cfg)
                # store the cluster information in the metadata
                cluster = []
                for pdb_name in pdb_csv['pdb_name']:
                    cluster.append(pdb_to_cluster[pdb_name] if pdb_name in pdb_to_cluster else "nan")

                unique_clusters = set(list(cluster))
                logging.info(f"Found {len(unique_clusters)} unique clusters.\n")
                pdb_csv['cluster'] = cluster

        # define a filter mask that is used to filter the dataset
        ###############################
        filter_mask = [True] * len(pdb_csv)

        # filter for breaks (which we define as non-continuous residue indices)
        if self.dataset_cfg.filter_breaks:
            # Check if column "breaks" exists
            if 'dist_breaks' not in pdb_csv.columns:
                breaks = []
                for path in tqdm(pdb_csv['processed_path'], desc='Saving dist breaks in metadata...'):
                    chain_feats = self._process_csv_row(path)

                    breaks.append(du.has_breaks(chain_feats) or du.has_inconstistent_indexing(chain_feats))
                pdb_csv['dist_breaks'] = breaks
                pdb_csv.to_csv(self.dataset_cfg.csv_path, index=False)
            
            filter_mask_breaks = pdb_csv['dist_breaks'] == False
            logging.info(f'Filtering for breaks: Removed {len(pdb_csv) - sum(filter_mask_breaks)} of {len(pdb_csv)} examples. {sum(filter_mask_breaks)} remaining.')
            filter_mask = np.logical_and(filter_mask, filter_mask_breaks)

        # filter for scrmsd
        if self.dataset_cfg.filter_scrmsd not in ["inf", "nan", float('inf'), float('nan'), None]:
            max_scrmsd = float(self.dataset_cfg.filter_scrmsd)
            # Check if column "scrmsd" exists or whether a csv file with scrmsd values is provided, then write the scrmsd values to the csv.
            if 'scrmsd' not in pdb_csv.columns or hasattr(self.dataset_cfg, 'scrmsd_csv_path'):
                assert hasattr(self.dataset_cfg, 'scrmsd_csv_path'), 'scrmsd_csv_path must be provided in the config if scrmsd column is not present in the csv.'
                if not Path(self.dataset_cfg.scrmsd_csv_path).exists():
                    raise FileNotFoundError(f"File {self.dataset_cfg.scrmsd_csv_path} not found. Set filter_scrmsd to 'inf' to ignore this error.")
                scrmsd_csv = pd.read_csv(self.dataset_cfg.scrmsd_csv_path)
                scrmsd_dict = {scrmsd_csv['pdb'][i]: scrmsd_csv['scrmsd'][i] for i in range(len(scrmsd_csv))}
                scrmsd_values = []
                for pdb_name in pdb_csv['pdb_name']:
                    scrmsd_values.append(scrmsd_dict[pdb_name] if pdb_name in scrmsd_dict else "nan")
                pdb_csv['scrmsd'] = scrmsd_values
                pdb_csv.to_csv(self.dataset_cfg.csv_path, index=False)
            
            scrmsd_values = [float(v) if v != "nan" else float('inf') for v in pdb_csv['scrmsd']]
            filter_mask_scrmsd = (np.array(scrmsd_values) <= max_scrmsd).tolist()
            now_filtered_out = [filter_mask[i] and not filter_mask_scrmsd[i] for i in range(len(filter_mask))]
            currently_remaining = sum(filter_mask)
            filter_mask = np.logical_and(filter_mask, filter_mask_scrmsd)
            
            logging.info(f'Filtering for scrmsd < {max_scrmsd}. Removed {sum(now_filtered_out)} of {currently_remaining} examples. {sum(filter_mask)} remaining.')

        if self.dataset_cfg.max_coil_pct < 1.:
            # Check if column "coil_pct" exists
            assert 'coil_percent' in pdb_csv.columns, 'Column "coil_percent" must be present in the metadata csv. Cols: ' + str(pdb_csv.columns)
            filter_mask_coil = pdb_csv['coil_percent'] <= self.dataset_cfg.max_coil_pct

            now_filtered_out = [filter_mask[i] and not filter_mask_coil[i] for i in range(len(filter_mask))]
            currently_remaining = sum(filter_mask)
            filter_mask = np.logical_and(filter_mask, filter_mask_coil)
            logging.info(f'Filtering for coil_pct <= {self.dataset_cfg.max_coil_pct}. Removed {sum(now_filtered_out)} of {currently_remaining} examples. {sum(filter_mask)} remaining.')

        # oligomer filter (if not in metadata cols, assume all are monomeric)
        if self.dataset_cfg.allowed_oligomers is not None:
            if self.dataset_cfg.allowed_oligomers != ['monomeric'] and not 'oligomeric_detail' in pdb_csv.columns:
                raise ValueError('Column "oligomeric_detail" must be present in the metadata csv if allowed_oligomers is set.')
            allowed_oligomers = self.dataset_cfg.allowed_oligomers
            if 'oligomeric_detail' in pdb_csv.columns:
                filter_mask_oligomers = pdb_csv['oligomeric_detail'].isin(allowed_oligomers)
                now_filtered_out = [filter_mask[i] and not filter_mask_oligomers[i] for i in range(len(filter_mask))]
                currently_remaining = sum(filter_mask)
                filter_mask = np.logical_and(filter_mask, filter_mask_oligomers)
                logging.info(f'Filtering for allowed oligomers: Removed {sum(now_filtered_out)} of {currently_remaining} examples. {sum(filter_mask)} remaining.')
        ###############################


        # Process information of breaks (unmodelled residues) in the pdb files
        ###############################
        # be default, assume that the break path is metadata_path.parent/breaks.csv:
        if self.dataset_cfg.label_breaks or self.dataset_cfg.use_res_idx:
            if self.dataset_cfg.break_csv_path is None:
                self.dataset_cfg.break_csv_path = str(Path(self.dataset_cfg.csv_path).parent/'breaks.csv')
                logging.info(f'data.dataset.break_csv_path is None. Setting it to data.dataset.csv_path.parent/breaks.csv, i.e. {self.dataset_cfg.break_csv_path}')
            if os.path.exists(self.dataset_cfg.break_csv_path):
                logging.info(f'Loading break information from self.dataset_cfg.break_csv_path={self.dataset_cfg.break_csv_path}')
                self.break_csv = pd.read_csv(self.dataset_cfg.break_csv_path)
            else:
                logging.info('Found no break information at self.dataset_cfg.break_csv_path')
                logging.info(f'Calculating breaks for dataset and storing them at {self.dataset_cfg.break_csv_path}...')

                breaks_dict = {
                    'pdb_name': [],
                    'idx_breaks': [],
                    'dist_breaks': [],
                    'merged_idx': [],
                    'consistent_breaks': []
                }

                for i in tqdm(range(len(pdb_csv)), desc='Calculating positions of breaks'):
                    csv_row = pdb_csv.iloc[i]
                    data = self._process_csv_row(csv_row['processed_path'])
                    idx1 = du.idx_breaks(data)
                    idx2 = du.dist_breaks(data)
                    merged_idx = np.unique(np.concatenate((idx1, idx2)))
                    # True if IDX and DIST criteriums for breaks are met at the same time
                    consistent_breaks = np.array_equal(idx1, idx2)
                    breaks_dict['pdb_name'].append(csv_row['pdb_name'])
                    breaks_dict['idx_breaks'].append(list(idx1))
                    breaks_dict['dist_breaks'].append(list(idx2))
                    breaks_dict['merged_idx'].append(list(merged_idx))
                    breaks_dict['consistent_breaks'].append(consistent_breaks)
                
                self.break_csv = pd.DataFrame(breaks_dict)
                self.break_csv.to_csv(self.dataset_cfg.break_csv_path, index=False)

        ###############################


        # Note: pdb_csv is the stored csv file with all samples and most relevant columns, self.csv will be a) filtered and b) appended by additional columns like break information etc. 
        self.csv = pdb_csv
        if self.dataset_cfg.use_res_idx or self.dataset_cfg.label_breaks:
            self.csv = self.csv.merge(self.break_csv, on='pdb_name')

        # apply filtering:
        if not isinstance(filter_mask, list):
            filter_mask = filter_mask.tolist()

        self.csv = self.csv[filter_mask]
        self.csv = self.csv.reset_index(drop=True)

        logging.info(f'Number of examples after filtering: {len(pdb_csv)}\n')
        # Filter for modeled sequence length.
        self.csv = self.csv[self.csv.modeled_seq_len <= self.dataset_cfg.max_num_res]
        self.csv = self.csv[self.csv.modeled_seq_len >= self.dataset_cfg.min_num_res]

        logging.info(f'Number of examples after filtering for modeled sequence length: {len(self.csv)}\n')

        if self.dataset_cfg.subset is not None:
            self.csv = self.csv.iloc[:self.dataset_cfg.subset]

        if self.dataset_cfg.calc_dssp:
            self.helix_percent = np.mean(self.csv['helix_percent'].to_numpy())
            self.strand_percent = np.mean(self.csv['strand_percent'].to_numpy())
        else:
            self.helix_percent = self.dataset_cfg.target_sec_content.helix_percent
            self.strand_percent = self.dataset_cfg.target_sec_content.strand_percent

        # Training or validation specific logic.
        if self.is_training:
            logging.info(f'Using {self.helix_percent} helix and {self.strand_percent} strand content as target secondary structure content')

            # Extract training set.
            if self.dataset_cfg.train_valid_test_split[0] != 1.0:
                self.csv = self.csv.groupby('modeled_seq_len')
                self.csv = self.csv.apply(lambda x: x.sample(frac=self.dataset_cfg.train_valid_test_split[0], replace=False, random_state=self.dataset_cfg.seed)).droplevel(0)
                self.csv = self.csv.sort_values('modeled_seq_len', ascending=False)
                self._log.info(
                    f'Training: {len(self.csv)} examples')
            else:
                self.csv = self.csv.sort_values('modeled_seq_len', ascending=False)
        else:
            if self.sample_dataset:
                mask = self.csv['modeled_seq_len'] >= self.dataset_cfg.min_num_res_eval
                mask &= self.csv['modeled_seq_len'] <= self.dataset_cfg.min_eval_length
                eval_csv = self.csv[mask]
                eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
                all_lengths = np.sort(eval_csv.modeled_seq_len.unique())
                length_indices = (len(all_lengths) - 1) * np.linspace(
                    0.0, 1.0, self.dataset_cfg.num_eval_lengths)
                length_indices = length_indices.astype(int)
                eval_lengths = all_lengths[length_indices]
                eval_csv = eval_csv[eval_csv.modeled_seq_len.isin(eval_lengths)]

                # Fix a random seed to get the same split each time.
                eval_csv = eval_csv.groupby('modeled_seq_len').sample(
                    self.dataset_cfg.samples_per_eval_length, replace=True, random_state=123)
                eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
                self.csv = eval_csv
                if self.dataset_cfg.generate_valid_samples:
                    self._log.info(
                        f'Generate {len(self.csv)} validation samples with lengths {eval_lengths}')
            else:
                # Extract validation set.
                if self.dataset_cfg.train_valid_test_split[1] > 0:
                    valid_frac = self.dataset_cfg.train_valid_test_split[1] / (self.dataset_cfg.train_valid_test_split[1] + self.dataset_cfg.train_valid_test_split[2])
                else:
                    self.csv = None
                    return

                train_csv = self.csv.groupby('modeled_seq_len')
                self.train_csv = train_csv.apply(lambda x: x.sample(frac=self.dataset_cfg.train_valid_test_split[0], replace=False, random_state=self.dataset_cfg.seed)).droplevel(0)
                self.csv = self.csv.drop(train_csv.index)
                self.csv = self.csv.groupby('modeled_seq_len')
                self.csv = self.csv.apply(lambda x: x.sample(frac=valid_frac, replace=False, random_state=self.dataset_cfg.seed)).droplevel(0)
                self.csv = self.csv.sort_values('modeled_seq_len', ascending=False)
                self._log.info(f'Validation: {len(self.csv)} examples')

    def _process_csv_row(self, processed_file_path):
        try:
            processed_feats = du.get_processed_feats(processed_file_path)
            modeled_idx = processed_feats['modeled_idx']
            if len(modeled_idx) == 0:
                raise ValueError(f'No modeled residues found in {processed_file_path}')

            # Filter out residues that are not modeled.
            min_idx = np.min(modeled_idx)
            max_idx = np.max(modeled_idx)
            processed_feats = tree.map_structure(
                    lambda x: x[min_idx:(max_idx+1)], processed_feats)

            # Run through OpenFold data transforms.
            chain_feats = {
                'aatype': torch.tensor(processed_feats['aatype']).long(),
                'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
                'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
            }
            chain_feats = data_transforms.atom37_to_frames(chain_feats)
            rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
            rotmats_1 = rigids_1.get_rots().get_rot_mats()
            trans_1 = rigids_1.get_trans()
            res_idx = processed_feats['residue_index']
            res_idx = res_idx - np.min(res_idx) + 1

        except Exception as e:
            raise ValueError(f'Error in processing {processed_file_path}') from e
        
        return {
            'aatype': chain_feats['aatype'],
            'res_idx': torch.tensor(res_idx),
            'rotmats_1': rotmats_1,
            'trans_1': trans_1,
            'res_mask': torch.tensor(processed_feats['bb_mask']).int(),
            'atom_positions': chain_feats['all_atom_positions'],
        }

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx, conf_idx:int=None):
        # Sample data example.
        example_idx = idx
        if isinstance(example_idx, list):
            example_idx = example_idx[0]

        csv_row = self.csv.iloc[example_idx]
        processed_file_path = csv_row['processed_path']
        chain_feats = self._process_csv_row(processed_file_path)
        chain_feats['csv_idx'] = torch.ones(1, dtype=torch.long) * idx

        if self.dataset_cfg.use_res_idx:
            # If there are inconsistencies in the different break definitions, consecutive residue indices are used
            if (not csv_row['consistent_breaks']):
                chain_feats['res_idx'] = torch.arange(chain_feats['res_mask'].shape[0], dtype=torch.long)
        else:
            chain_feats['res_idx'] = torch.arange(chain_feats['res_mask'].shape[0], dtype=torch.long)

        if self.dataset_cfg.label_breaks:
            break_mask = torch.zeros(chain_feats['res_mask'].shape[0], dtype=torch.float32)
            break_idc = csv_row['merged_idx']
            # if there are no breaks, break_idc is empty, so we can leave the mask as it is
            if len(break_idc) > 0:
                if isinstance(break_idc, str):
                    break_idc = ast.literal_eval(break_idc)
                break_idc = np.array(break_idc, dtype=np.int32)
                break_idc = np.append(break_idc, break_idc + 1)
                #Create tensor of length N with 1 at break positions
                break_mask[break_idc] = 1.0

            chain_feats['breaks'] = break_mask

        if not self.is_training:
            chain_feats['target_helix_percent'] = self.helix_percent
            chain_feats['target_strand_percent'] = self.strand_percent

        return chain_feats


class LengthBatcher:

    def __init__(
            self,
            *,
            sampler_cfg,
            metadata_csv,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
            num_batches=None,
        ):
        """
        If clustered: Assumes that there is an entry "cluster" in the metadata, for which a unique identifier is expected. Then, an epoch is defined as iteration over all clusters, where for each epoch, one random cluster member is picked.
        """
        # In FrameFlow, each epoch had n_data batches, so the data points were seen multiple times per epoch.
        # here, we define epoch as the iteration over all data points once.
        super().__init__()
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv
        self._cluster_init()
        self._data_csv['index'] = list(range(len(self._data_csv)))
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size =  self._sampler_cfg.max_batch_size
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}')

        # OUTCOMMENTED BECAUSE DDP REQUIRES SAME NUM BATCHES ACROSS ALL REPLICAS!
        ###############################
        # # set max num batches to be the number of batches in epoch 0.
        # self.num_batches = None
        # self._create_batches()
        # self.num_batches = len(self.sample_order)
        ###############################


        self.num_batches = num_batches
        if num_batches is not None:
            self.num_batches = num_batches//self.num_replicas
            self._log.info(f'Total number of batches: {self.num_batches}')
            self._log.info(f'Number of batches per replica: {self.num_batches}')

    def _cluster_init(self):
        self.clustered = self._sampler_cfg.clustered if hasattr(self._sampler_cfg, 'clustered') else False
        if self.clustered:
            if not 'cluster' in self._data_csv.columns:
                raise ValueError("Clustered sampling requires a 'cluster' column in the metadata.")
            
            def _cluster_unassigned(cluster_id):
                if cluster_id in ["", "None", "nan", None]:
                    return True
                if isinstance(cluster_id, float) or isinstance(cluster_id, int):
                    return np.isnan(cluster_id)
                return False
                
            # assign clusters to states that have no valid cluster entry in the metadata.csv:
            for i, row in self._data_csv.iterrows():
                if _cluster_unassigned(row['cluster']):
                    new_cluster = hash(str(i)) # assume that this does not occur twice
                    self._data_csv.at[i, 'cluster'] = new_cluster

                elif not isinstance(row['cluster'], str):
                    if float(row['cluster']).is_integer():
                        self._data_csv.at[i, 'cluster'] = int(row['cluster'])
                    else:
                        raise ValueError(f"Cluster must be a string or integer-like float. Found {row['cluster']} in {row['pdb_name']}.")
                    

            all_clusters = self._data_csv['cluster'].unique()
            has_nan_cluster = any([cluster == "nan" for cluster in all_clusters])
            if has_nan_cluster and len(all_clusters) == 1:
                raise RuntimeError("Internal error. All clusters are nan.")
            if has_nan_cluster:
                logging.warning(f"Found nan cluster among {len(all_clusters)} clusters. This will be treated as a separate cluster.")

            # dictionary that maps cluster to a list of csv indices that belong to that cluster
            self.cluster_dict = {cluster: self._data_csv[self._data_csv['cluster'] == cluster].index.tolist() for cluster in all_clusters}
            assert sum([len(self.cluster_dict[cluster]) for cluster in self.cluster_dict]) == len(self._data_csv), "Internal error. Not all indices are assigned to a cluster."

            logging.info(f'Applying clustered sampling: {len(all_clusters)} clusters, {len(self._data_csv)} examples.')


    def _get_replica_csv(self, rng:torch.Generator):
        replica_csv = self._data_csv

        if self.clustered:
            np_rng = np.random.default_rng(seed=int(rng.initial_seed()))
            # for each cluster, choose a random member and write it to replica_csv
            cluster_indices = []
            for cluster in self.cluster_dict:
                cluster_indices.append(np_rng.choice(self.cluster_dict[cluster], size=1)[0])

            replica_csv = replica_csv[replica_csv.index.isin(cluster_indices)]

        # indices are only used for randomly splitting the batch across devices
        indices = list(range(len(replica_csv)))

        if self.shuffle:
            indices = torch.randperm(len(indices), generator=rng).tolist()
        
        # replica csv is a reordered version (and for num_replicas > 1 also subset of) the original csv
        if len(indices) > self.num_replicas:
            replica_csv = replica_csv.iloc[
                indices[self.rank::self.num_replicas]
            ]

        return replica_csv
        

    def _replica_epoch_batches(self):
        """
        Returns the batch idxs for the current epoch.
        """
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)

        replica_csv = self._get_replica_csv(rng)
        
        # Each batch contains multiple proteins of the same length.
        sample_order = []
        for seq_len, len_df in replica_csv.groupby('modeled_seq_len'):
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // seq_len**2 + 1,
            )
            num_batches = math.ceil(len(len_df) / max_batch_size)
            for i in range(num_batches):
                batch_df = len_df.iloc[i*max_batch_size:(i+1)*max_batch_size]
                batch_indices = batch_df['index'].tolist()
                sample_order.append(batch_indices)
        
        # Remove any length bias.
        new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
        return [sample_order[i] for i in new_order]

    # def _create_batches(self):
    #     # Make sure all replicas have the same number of batches Otherwise leads to bugs.
    #     # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
    #     all_batches = []
    #     num_augments = -1
    #     while len(all_batches) < self._num_batches:
    #         all_batches.extend(self._replica_epoch_batches())
    #         num_augments += 1
    #         if num_augments > 1000:
    #             raise ValueError('Exceeded number of augmentations.')
    #     if len(all_batches) >= self._num_batches:
    #         all_batches = all_batches[:self._num_batches]
    #     self.sample_order = all_batches

    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        all_batches = self._replica_epoch_batches()
        if self.num_batches is not None:
            if len(all_batches) > self.num_batches:
                all_batches = all_batches[:self.num_batches]
            elif len(all_batches) < self.num_batches:
                # randomly duplicate batches to match the number of batches
                duplicates = np.random.choice(len(all_batches), size=self.num_batches - len(all_batches), replace=True)
                all_batches.extend([all_batches[d] for d in duplicates])
        else:
            raise ValueError("num_batches must be set before creating batches")
        self.sample_order = all_batches


    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return self.num_batches


def get_num_batches(sampler_cfg, metadata_csv, seed=123):
    """
    Function to infer the number of batches in an epoch. This can be used for automatically inferring the approx. (exact number is epoch-dependent) total number of batches for LengthBatcher. Has to be called before initializing the actual distributed samplers since the number of batches has to be the same across all replicas.
    """
    # create a dummy length batcher with num_replicas=1:
    batcher = LengthBatcher(
        sampler_cfg=sampler_cfg,
        metadata_csv=metadata_csv,
        seed=seed,
        shuffle=True,
        num_replicas=1,
        rank=0,
        num_batches=None
    )
    all_batches = batcher._replica_epoch_batches()
    return len(all_batches)


class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("This dataset is empty")
    


def _read_clusters_framediff(dataset_cfg):
    pdb_to_cluster = {}
    with open(dataset_cfg.cluster_path_framediff, "r") as f:
        for i,line in enumerate(f):
            for chain in line.split(' '):
                pdb = chain.split('_')[0]
                pdb_to_cluster[pdb.lower()] = str(i)
    return pdb_to_cluster

def _get_pdb_to_cluster_dict_framediff(dataset_cfg):
    pdb_to_cluster = _read_clusters_framediff(dataset_cfg)
    return pdb_to_cluster
