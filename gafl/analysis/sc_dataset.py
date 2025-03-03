# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import numpy as np
import torch
import subprocess
from typing import Optional
from biotite.sequence.io import fasta
import pandas as pd
# import esm
# from esm.esmfold.v1.esmfold import ESMFold
from tqdm import tqdm
import esm
import pickle
from analysis import utils as au
from gafl.data import utils as du
from gafl.analysis import metrics

def filter_trainset(path: str):
    for file in tqdm(os.listdir(path)):
        if file.endswith(".pkl"):
            with open(os.path.join(path, file), 'rb') as f:
                structure = pickle.load(f)

            #Get number of unique residues
            length = len(structure['residue_index'])

            if length > 128:
                continue

            atom_pos = structure['atom_positions']
            os.makedirs(os.path.join(path, 'self_consistency', file.replace('.pkl', '')), exist_ok=True)
            au.write_prot_to_pdb(atom_pos, os.path.join(path, 'self_consistency', file.replace('.pkl', ''), "sample.pdb"), overwrite=True, no_indexing=True)

def run_self_consistency_batch(
        path: str,
        pdb_count = None,
        max_esm_batch = 10,
        length_interval = None,
        motif_mask: Optional[np.ndarray]=None,
        calc_non_coil_rmsd: bool = False
    ):
    """
    """

    sequences_per_sample = 8
    folding_model = esm.pretrained.esmfold_v1().eval()
    folding_model = folding_model.cuda()

    # os.makedirs(os.path.join(path, "self_consistency"), exist_ok=True)

    with torch.no_grad():
        fasta_headers = []
        fasta_seqs = []
        processed_pdbs = []
        print("Running ProteinMPNN on filtered training samples...")
        # for sample_id in tqdm(sample_ids, desc="ProteinMPNN for samples"):

        files = os.listdir(path)
        if pdb_count is not None:
            files = files[:pdb_count]

        for file in tqdm(files):
            if not os.path.isdir(os.path.join(path, file)):
                continue
            if length_interval is not None:
                pdb_length = au.get_pdb_length(os.path.join(path, file, "sample.pdb"))
                if pdb_length < length_interval[0] or pdb_length > length_interval[1]:
                    continue

            processed_pdbs.append(file)
            decoy_pdb_dir = os.path.join(path, file)
            reference_pdb_path = os.path.join(decoy_pdb_dir, 'sample.pdb')

            # Run PorteinMPNN
            output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
            # print(output_path)
            process = subprocess.Popen([
                'python',
                f'./ProteinMPNN/helper_scripts/parse_multiple_chains.py',
                f'--input_path={decoy_pdb_dir}',
                f'--output_path={output_path}',
            ])
            # print(process)
            _ = process.wait()
            num_tries = 0
            ret = -1
            pmpnn_args = [
                'python',
                f'./ProteinMPNN/protein_mpnn_run.py',
                '--out_folder',
                decoy_pdb_dir,
                '--jsonl_path',
                output_path,
                '--num_seq_per_target',
                str(sequences_per_sample),
                '--sampling_temp',
                '0.1',
                '--seed',
                '38',
                '--batch_size',
                '1',
                '--device',
                '0'
            ]
            # print(pmpnn_args)
            while ret < 0:
                try:
                    process = subprocess.Popen(
                        pmpnn_args,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT
                    )
                    ret = process.wait()
                except Exception as e:
                    num_tries += 1
                    print(f'Failed ProteinMPNN. Attempt {num_tries}/5')
                    torch.cuda.empty_cache()
                    if num_tries > 4:
                        raise e
            mpnn_fasta_path = os.path.join(
                decoy_pdb_dir,
                'seqs',
                os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
            )
            # print(mpnn_fasta_path)

            fasta_file = fasta.FastaFile.read(mpnn_fasta_path)
            fasta_headers.extend(list(fasta_file.keys())[1:])
            fasta_seqs.extend(list(fasta_file.values())[1:])

        slice_size = sequences_per_sample * max_esm_batch
        slices = [(i,i+slice_size) for i in range(0, len(fasta_seqs), slice_size)]

        print(f"Run ESM on predicted sequences")
        for j, s in tqdm(enumerate(slices), total=len(slices)):
            # Run ESMFold on each ProteinMPNN sequence
            all_esm_output = folding_model.infer(fasta_seqs[s[0]:s[1]])
            all_esm_pdbs = folding_model.output_to_pdb(all_esm_output)
            # print(all_esm_output.keys())

            # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
            mpnn_results = {
                'tm_score': [],
                'sample_path': [],
                'header': [],
                'sequence': [],
                'rmsd': [],
            }
            if calc_non_coil_rmsd:
                mpnn_results['non_coil_rmsd'] = []
            if motif_mask is not None:
                # Only calculate motif RMSD if mask is specified.
                mpnn_results['motif_rmsd'] = []
            # for i in tqdm(range(len(fasta_seqs)), desc="Calc metrics and save"):
            for i in range(len(fasta_seqs[s[0]:s[1]])):
                header = fasta_headers[s[0]:s[1]][i]
                sequence = fasta_seqs[s[0]:s[1]][i]
                # esm_output = all_esm_output[i]
                esm_pdb = all_esm_pdbs[i]
                decoy_pdb_dir = os.path.join(path, processed_pdbs[i//sequences_per_sample + j*max_esm_batch])
                reference_pdb_path = os.path.join(decoy_pdb_dir, 'sample.pdb')

                esmf_dir = os.path.join(decoy_pdb_dir, 'esmf')
                os.makedirs(esmf_dir, exist_ok=True)

                esmf_sample_path = os.path.join(esmf_dir, f'sample_{i%sequences_per_sample}.pdb')
                with open(esmf_sample_path, "w") as f:
                    f.write(esm_pdb)

                sample_feats = du.parse_pdb_feats('sample', reference_pdb_path)
                esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path, calc_dssp=calc_non_coil_rmsd)
                sample_seq = du.aatype_to_seq(sample_feats['aatype'])

                # Calculate scTM of ESMFold outputs with reference protein
                _, tm_score = metrics.calc_tm_score(
                    sample_feats['bb_positions'], esmf_feats['bb_positions'],
                    sample_seq, sample_seq)
                rmsd = metrics.calc_aligned_rmsd(
                    sample_feats['bb_positions'], esmf_feats['bb_positions'])
                
                if calc_non_coil_rmsd:
                    # calculate the rmsd if coils in the refolded structure are ignored:
                    non_coil_idxs = np.where(esmf_feats['dssp'] != 'C')[0]
                    if len(non_coil_idxs) == 0:
                        non_coil_rmsd = np.nan
                    else:
                        non_coil_rmsd = metrics.calc_aligned_rmsd(
                            sample_feats['bb_positions'][non_coil_idxs],
                            esmf_feats['bb_positions'][non_coil_idxs])
                
                if motif_mask is not None:
                    sample_motif = sample_feats['bb_positions'][motif_mask]
                    of_motif = esmf_feats['bb_positions'][motif_mask]
                    motif_rmsd = metrics.calc_aligned_rmsd(
                        sample_motif, of_motif)
                    mpnn_results['motif_rmsd'].append(motif_rmsd)
                mpnn_results['rmsd'].append(rmsd)
                if calc_non_coil_rmsd:
                    mpnn_results['non_coil_rmsd'].append(non_coil_rmsd)
                mpnn_results['tm_score'].append(tm_score)
                mpnn_results['sample_path'].append(esmf_sample_path)
                mpnn_results['header'].append(header)
                mpnn_results['sequence'].append(sequence)


                if (i+1) % sequences_per_sample == 0:
                    if i > 0:
                        # Save results to CSV
                        csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
                        mpnn_results = pd.DataFrame(mpnn_results)
                        mpnn_results.to_csv(csv_path)
                    mpnn_results = {
                        'tm_score': [],
                        'sample_path': [],
                        'header': [],
                        'sequence': [],
                        'rmsd': [],
                    }
                    if calc_non_coil_rmsd:
                        mpnn_results['non_coil_rmsd'] = []