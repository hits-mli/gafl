# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""DDP inference script."""
import os
import time
import numpy as np
import hydra
import torch
import GPUtil
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from gafl.models.flow_module import FlowModule
from gafl import experiment_utils as eu



torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)

class Sampler:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'

        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up directories to write results to
        self._ckpt_name = '/'.join(ckpt_path.replace('.ckpt', '').split('/')[-3:])
        self._output_dir = os.path.join(
            self._infer_cfg.output_dir,
            self._ckpt_name,
            self._infer_cfg.name,
        )
        os.makedirs(self._output_dir, exist_ok=True)
        self.config_path = os.path.join(self._output_dir, 'config.yaml')
        with open(self.config_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f)

        # Read checkpoint and initialize module.
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
        )
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        if not hasattr(self._infer_cfg, 'max_res_per_esm_batch'):
            self._infer_cfg.max_res_per_esm_batch = 2056
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._output_dir = self._output_dir

    def run_sampling(self):
        devices = GPUtil.getAvailable(
            order='memory', limit = 8)[:self._infer_cfg.num_gpus]
        log.info(f"Using devices: {devices}")
        # eval_dataset = eu.LengthDataset(self._samples_cfg)
        eval_dataset = eu.LengthDatasetBatch(self._samples_cfg)
        dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=1, shuffle=False, drop_last=False)
        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=devices,
        )
        self._log_information_on_start()
        trainer.predict(self._flow_module, dataloaders=dataloader)

    def _log_information_on_start(self):
        min_len = self._samples_cfg.min_length
        max_len = self._samples_cfg.max_length
        samples_per_length = self._samples_cfg.samples_per_length
        length_step = self._samples_cfg.length_step
        num_timesteps = self._infer_cfg.interpolant.sampling.num_timesteps

        sampled_lengths = list(range(min_len, max_len + 1, length_step))
        
        MAX_PRINTED_LENS = 5
        if len(sampled_lengths) <= MAX_PRINTED_LENS:
            sampled_len_str = '[' + ', '.join(map(str, sampled_lengths)) + ']'
        else:
            sampled_len_str = '[' + ', '.join(map(str, sampled_lengths[:MAX_PRINTED_LENS-1])) + ', ..., ' + str(sampled_lengths[-1]) + ']'
            
        logstr = f'\n\nSAMPLING CONFIG:\n{"="*80}'
        logstr += f'\nSaving results to {self._output_dir}'
        logstr += f'\nNumber of timesteps: {num_timesteps}'
        logstr += f'\nGenerating {samples_per_length} samples for each length in {sampled_len_str}'
        logstr += f'\n{"="*80}\n'
        
        log.info(logstr)


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def run(cfg: DictConfig) -> None:

    # Read model checkpoint.
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = Sampler(cfg)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time

    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
