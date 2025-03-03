# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any
import torch
import time
import os
import random
import wandb
import math
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
import torch.distributed as dist
import shutil
from omegaconf import OmegaConf
import warnings

from gafl.analysis import metrics 
from gafl.analysis import utils as au
from gafl.models.flow_model import FlowModel
from gafl.models import utils as mu
from gafl.data.interpolant import Interpolant 
from gafl.data import utils as du
from gafl.data import all_atom
from gafl.data import so3_utils
from gafl.data import residue_constants
from gafl.data.protein import from_pdb_string
from gafl import experiment_utils as eu
# Suppress only the specific PyTorch Lightning user warnings about sync_dist, which are triggered although the world size is 1.
warnings.filterwarnings("ignore", message=".*sync_dist=True.*")

class FlowModule(LightningModule):

    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model and interpolant
        self.create_model()

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        self.training_epoch_metrics = []

        self.validation_epoch_metrics = []
        self.validation_sample_epoch_metrics = []
        self.validation_epoch_samples = []

        self._time_bins = torch.linspace(0.0, 1.0, 101)
        self._time_histogram = torch.zeros(100, dtype=torch.int64)

        self.nan_steps = 0

        if hasattr(self._exp_cfg, 'warmup_lr'):
            self.warmup_lr = self._exp_cfg.warmup_lr
        else:
            self.warmup_lr = False

        if hasattr(self._exp_cfg, 'warmup_lr_factor'):
            self.warmup_lr_factor = self._exp_cfg.warmup_lr_factor
        else:
            self.warmup_lr_factor = 0.1

        OmegaConf.set_struct(self._exp_cfg, False)
        if not hasattr(self._exp_cfg, 'scheduler'):
            self._exp_cfg.scheduler = OmegaConf.create()
            self._exp_cfg.scheduler.enabled=False
        OmegaConf.set_struct(self._exp_cfg, True)

        if hasattr(self._exp_cfg, 'reset_optimizer_on_load'):
            self.reset_optimizer_on_load = self._exp_cfg.reset_optimizer_on_load
        else:
            self.reset_optimizer_on_load = False

        self.save_hyperparameters()

    def create_model(self):
        if "module" in self._model_cfg:
            # backwards compatibility:
            if 'gafl_pdb_new' in self._model_cfg.module:
                self._model_cfg.module = str(self._model_cfg.module).replace('gafl_pdb_new', 'flow_model')
            model_module = mu.load_module(self._model_cfg.module)
            self.model = model_module(self._model_cfg)
        else:
            self.model = FlowModel(self._model_cfg)
        self.interpolant = Interpolant(self._interpolant_cfg)

    def on_train_start(self):
        self._epoch_start_time = time.time()
        
    def on_train_epoch_end(self):
        if not self.training_epoch_metrics:
            logging.warning('No training metrics to log')
            self.training_epoch_metrics.clear()
            self._epoch_start_time = time.time()
            return
        
        train_epoch_metrics = pd.concat(self.training_epoch_metrics)

        train_epoch_dict = train_epoch_metrics.mean().to_dict()

        for metric_name,metric_val in train_epoch_dict.items():
            self._log_scalar(
                f'train_epoch/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(train_epoch_metrics),
            )

        self.training_epoch_metrics.clear()

        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self._log_scalar(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def loss_fn(self, noisy_batch: Any, model_output: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask']
        if training_cfg.min_plddt_mask is not None:
            plddt_mask = noisy_batch['res_plddt'] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3] 

        # Timestep used for normalization.
        t = noisy_batch['t']
        norm_scale = 1 - torch.min(
            t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        # Model output predictions.
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        # loss denom may not contain any zeros:
        assert (loss_denom == 0).sum() == 0, 'Loss denom contains zeros'

        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / norm_scale * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss) * (
            t[:, 0] > training_cfg.aux_loss_t_pass
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        se3_vf_loss += auxiliary_loss
        # if torch.isnan(se3_vf_loss).any():
            # raise ValueError('NaN loss encountered')
        return {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss
        }

    def model_step(self, noisy_batch: Any):
        model_output = self.model(noisy_batch)
        
        losses = self.loss_fn(noisy_batch, model_output)
        return losses

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        if batch is None:
            return
        
        if dataloader_idx == 0:
            res_mask = batch['res_mask']
            self.interpolant.set_device(res_mask.device)
            num_batch, num_res = res_mask.shape
            
            samples = self.interpolant.sample(
                num_batch,
                num_res,
                self.model,
            )[0][-1].numpy()

            batch_metrics = []
            for i in range(num_batch):

                # Write out sample to PDB file
                final_pos = samples[i]
                saved_path = au.write_prot_to_pdb(
                    final_pos,
                    os.path.join(
                        self._sample_write_dir,
                        f'sample_{i}_idx_{batch_idx}_len_{num_res}.pdb'),
                    no_indexing=True
                )
                if isinstance(self.logger, WandbLogger):
                    with open(saved_path, 'r') as f:
                        atom37 = from_pdb_string(f.read()).atom_positions
                    N = atom37.shape[0]
                    backbone = atom37[:,:5,:]
                    colors = np.zeros((N, 5, 3))
                    colors[:,0,:] = np.array([0.0, 0.0, 1.0])
                    colors[:,1,:] = np.array([0.25, 0.75, 0.75])
                    colors[:,2,:] = np.array([0.25, 0.75, 0.75])
                    colors[:,3,:] = np.array([0.25, 0.75, 0.75])
                    colors[:,4,:] = np.array([1.0, 0.0, 0.0])
                    backbone = np.concatenate([backbone, colors*255], axis=-1).reshape(N*5, 6)
                    Ca_atoms = atom37[:,1,:]
                    self.validation_epoch_samples.append(
                        [saved_path, self.global_step, wandb.Molecule(saved_path), wandb.Object3D(backbone), wandb.Object3D(Ca_atoms)]
                    )

                mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
                ca_idx = residue_constants.atom_order['CA']
                ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
                batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

            batch_metrics = pd.DataFrame(batch_metrics)
            self.validation_sample_epoch_metrics.append(batch_metrics)

            if batch_idx == 0:
                self.target_helix_percent = batch['target_helix_percent']
                self.target_strand_percent = batch['target_strand_percent']
        
        if dataloader_idx == 1:
            self.interpolant.set_device(batch['res_mask'].device)
            noisy_batch = self.interpolant.corrupt_batch(batch)
            if self._interpolant_cfg.self_condition and random.random() > 0.5:
                with torch.no_grad():
                    model_sc = self.model(noisy_batch)
                    noisy_batch['trans_sc'] = model_sc['pred_trans']
            batch_losses = self.model_step(noisy_batch)

            batch_metrics = {}
            batch_metrics.update({k: [torch.mean(v).cpu().item()] for k,v in batch_losses.items()})

            # Losses to track. Stratified across t.
            t = torch.squeeze(noisy_batch['t'])
            
            if self._exp_cfg.training.t_bins > 1:
                for loss_name, loss_dict in batch_losses.items():
                    stratified_losses = mu.t_stratified_loss(
                        t, loss_dict, num_bins=self._exp_cfg.training.t_bins, t_interval=self._interpolant_cfg.t_interval, loss_name=loss_name)
                    batch_metrics.update({k: [v] for k,v in stratified_losses.items()})
        
            batch_metrics = pd.DataFrame(batch_metrics)
            self.validation_epoch_metrics.append(batch_metrics)


    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "Protein", "Backbone", "C-alpha"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()
        
        if len(self.validation_sample_epoch_metrics) > 0:
            val_epoch_metrics = pd.concat(self.validation_sample_epoch_metrics)

            val_epoch_dict = val_epoch_metrics.mean().to_dict()
            # Calculate deviation of mean helix percent and mean strand percent from dataset
            # this quantity is actually the mean of the mean of the residue-level helix and strand percent! (which is fine, otherwise small proteins would have less weight)
            helix_deviation = val_epoch_metrics['helix_percent'].mean() - self.target_helix_percent
            strand_deviation = val_epoch_metrics['strand_percent'].mean() - self.target_strand_percent
            # sec_deviation = math.sqrt(helix_deviation**2 + strand_deviation**2)
            sec_deviation = abs(helix_deviation) + abs(strand_deviation)
            self._log_scalar(
                'valid/sec_deviation',
                sec_deviation,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics)
            )

            for metric_name,metric_val in val_epoch_dict.items():
                self._log_scalar(
                    f'valid/{metric_name}',
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=len(val_epoch_metrics),
                )
            self.validation_sample_epoch_metrics.clear()

        
        if len(self.validation_epoch_metrics) > 0:
            val_epoch_metrics = pd.concat(self.validation_epoch_metrics)

            val_epoch_dict = val_epoch_metrics.mean().to_dict()

            for metric_name,metric_val in val_epoch_dict.items():
                self._log_scalar(
                    f'valid/{metric_name}',
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=len(val_epoch_metrics),
                )
        

            self.validation_epoch_metrics.clear()

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist is None:
            sync_dist = self.trainer.world_size > 1
        if rank_zero_only is None:
            rank_zero_only = not sync_dist
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def training_step(self, batch: Any, stage: int):
        batchsize = batch['res_mask'].shape[0]
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = model_sc['pred_trans']
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }

        train_loss = (
            total_losses[self._exp_cfg.training.loss]
            +  total_losses['auxiliary_loss']
        )

        nan_detected = torch.isnan(train_loss).float()
        
        # Ensure all processes are aware of NaNs
        if self.trainer.world_size > 1:  # Check if using DDP
            dist.all_reduce(nan_detected, op=dist.ReduceOp.SUM)
        
        # If any process detected NaN, skip the step across all GPUs
        if nan_detected > 0:
            self.nan_steps += 1
            if self.nan_steps > 5:
                raise ValueError('NaN loss encountered too many times in a row')
            if self.global_rank == 0:
                logging.info(f"NaN detected in loss (epoch {self.current_epoch}). Skipping this step.")
            return None  # Skip the step
        else:
            self.nan_steps = 0

        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        

        batch_metrics = {}
        batch_metrics.update({k: [v.cpu().item()] for k,v in total_losses.items()})

        # Losses to track. Stratified across t.
        t = torch.squeeze(noisy_batch['t'])
        self._log_scalar(
            "train/t",
            np.mean(du.to_numpy(t)),
            prog_bar=False, batch_size=num_batch)
        
        if self._exp_cfg.training.t_bins > 1:
            for loss_name, loss_dict in batch_losses.items():
                stratified_losses = mu.t_stratified_loss(
                    t, loss_dict, num_bins=self._exp_cfg.training.t_bins, t_interval=self._interpolant_cfg.t_interval, loss_name=loss_name)
                batch_metrics.update({k: [v] for k,v in stratified_losses.items()})

                for k,v in stratified_losses.items():
                    self._log_scalar(
                        f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        batch_metrics = pd.DataFrame(batch_metrics)
        self.training_epoch_metrics.append(batch_metrics)

        # Training throughput
        self._log_scalar(
            "train/length", float(batch['res_mask'].shape[1]), prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", float(num_batch), prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar(
            "train/examples_per_second", num_batch / step_time)
        # train_loss = (
        #     total_losses[self._exp_cfg.training.loss]
        #     +  total_losses['auxiliary_loss']
        # ) # This double counts the auxiliary loss
        # train_loss = total_losses[self._exp_cfg.training.loss]
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)
        
        if self._exp_cfg.use_wandb:
            wandb_logs = {}
            time_indices = torch.bucketize(noisy_batch["t"].detach().cpu(), self._time_bins)
            time_indices_unique, counts = torch.unique(time_indices-1, return_counts=True)
            time_indices_unique[time_indices_unique < 0] = 0
            self._time_histogram[time_indices_unique] += counts
            wandb_logs["sampled_time_cumulative"] = wandb.Histogram(
                np_histogram=((
                    (self._time_histogram/self._time_histogram.sum()).numpy(),
                    self._time_bins.numpy()
                ))
            )
            time_histogram_step = torch.zeros(self._time_histogram.shape)
            time_histogram_step[time_indices_unique] += counts
            wandb_logs["sampled_time_per_step"] = wandb.Histogram(
                np_histogram=((
                    (time_histogram_step/time_histogram_step.sum()).numpy(),
                    self._time_bins.numpy()
                ))
            )
            self.logger.experiment.log(wandb_logs)


        return train_loss

    def lr_lambda(self, epoch):
        if epoch < self._exp_cfg.scheduler.warmup_epochs:
            # Warmup phase: Linear increase
            warmup_factor = epoch / self._exp_cfg.scheduler.warmup_epochs
            return  warmup_factor + (1 - warmup_factor) * self._exp_cfg.scheduler.warmup_lr / self._exp_cfg.optimizer.lr
        elif epoch < self._exp_cfg.scheduler.warmup_epochs + self._exp_cfg.scheduler.constant_epochs:
            # Constant LR phase
            return 1.0
        elif epoch < self._exp_cfg.scheduler.min_epoch:
            # Decay phase: Cosine annealing
            decay_epochs = epoch - (self._exp_cfg.scheduler.warmup_epochs + self._exp_cfg.scheduler.constant_epochs)
            decay_factor = 0.5 * (1 + math.cos(torch.pi * decay_epochs / (self._exp_cfg.scheduler.min_epoch - self._exp_cfg.scheduler.warmup_epochs - self._exp_cfg.scheduler.constant_epochs)))
            return self._exp_cfg.scheduler.lr_min / self._exp_cfg.optimizer.lr * (1 - decay_factor) + decay_factor
        else:
            return self._exp_cfg.scheduler.lr_min / self._exp_cfg.optimizer.lr

    def configure_optimizers(self):
        if not self._exp_cfg.scheduler.enabled:
            return torch.optim.AdamW(
                params=self.model.parameters(),
                **self._exp_cfg.optimizer
            )
        else:
            optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                **self._exp_cfg.optimizer
            )
            # if self.warmup_lr_factor == 0:
            #     return optimizer
            
            # # train the first epoch with a smaller learning rate:
            # small_lr = self.warmup_lr_factor * self._exp_cfg.optimizer.lr
            # this_epoch = self.trainer.current_epoch

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=self.lr_lambda
            )
            return [optimizer], [scheduler]
    
    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step()
    
    def on_load_checkpoint(self, *args, **kwargs):
        output = super().on_load_checkpoint(*args, **kwargs)
        if self.reset_optimizer_on_load:
            self.configure_optimizers()
        return output


    def predict_step(self, batch, batch_idx):
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()
        diffuse_mask = torch.ones(1, sample_length)
        all_sample_ids = batch['sample_id'][0].cpu().numpy()

        batch_size = self._infer_cfg.samples.batch_size
        sample_ids_batches = np.split(
            all_sample_ids,
            np.arange(batch_size, len(all_sample_ids), batch_size)
        )

        for sample_ids in sample_ids_batches:
            length_dir = os.path.join(self._output_dir, f'length_{sample_length}')

            for i, sample_id in enumerate(sample_ids):
                sample_dir = os.path.join(length_dir, f'sample_{sample_id}')
                os.makedirs(sample_dir, exist_ok=True)

            atom37_traj, model_traj, _ = interpolant.sample(
                len(sample_ids), sample_length, self.model, save_path=length_dir
            )

            atom37_traj = du.to_numpy(torch.stack(atom37_traj, dim=1))
            model_traj = du.to_numpy(torch.stack(model_traj, dim=1))

            for i, sample_id in enumerate(sample_ids):
                sample_dir = os.path.join(length_dir, f'sample_{sample_id}')
                paths = eu.save_traj(
                    atom37_traj[i, -1],
                    np.flip(atom37_traj[i], axis=0),
                    np.flip(model_traj[i], axis=0),
                    du.to_numpy(diffuse_mask)[0],
                    output_dir=sample_dir,
                )