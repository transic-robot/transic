from typing import Any, List, Union, Tuple

import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import LightningModule
from rl_games.common import env_configurations
from rl_games.algos_torch.model_builder import register_network, register_model
import transic_envs
from transic_envs.envs.core.pcd_base import (
    sample_points,
    apply_pc_aug_random_trans,
    apply_pc_aug_jitter,
    TRANSICEnvPCD,
)

from transic.learn.policy import BasePolicy
from transic.utils.datadict import any_to_datadict
from transic.utils.config_utils import omegaconf_to_dict
from transic.utils.tree_utils import unstack_sequence_fields
from transic.utils.array import (
    torch_device,
    get_batch_size,
    any_concat,
    any_to_torch_tensor,
)
from transic.rl.models import ModelA2CContinuousLogStd
from transic.rl.network_builder import DictObsBuilder

register_model("my_continuous_a2c_logstd", ModelA2CContinuousLogStd)
register_network("dict_obs_actor_critic", DictObsBuilder)


class _DistillationBaseModule(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._on_train_once = False

    def training_step(self, *args, **kwargs):
        self.on_train_start()
        loss, log_dict, batch_size = self.distillation_training_step(*args, **kwargs)
        log_dict = {f"train/{k}": v for k, v in log_dict.items()}
        log_dict["train/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, *args, **kwargs):
        loss, log_dict = self.distillation_test_step(*args, **kwargs)
        log_dict = {f"val/{k}": v for k, v in log_dict.items()}
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        return loss

    def test_step(self, *args, **kwargs):
        self.env = self.create_env()
        loss, log_dict = self.distillation_test_step(*args, **kwargs)
        log_dict = {f"test/{k}": v for k, v in log_dict.items()}
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        return loss

    def on_train_start(self):
        if not self._on_train_once:
            self.env = self.create_env()
            self._on_train_once = True

    def configure_optimizers(self):
        """
        Get optimizers, which are subsequently used to train.
        """
        raise NotImplementedError

    def distillation_training_step(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def distillation_test_step(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def create_env(self) -> Any:
        """
        Create vectorized distributed envs.
        """
        raise NotImplementedError


class DistillationModule(_DistillationBaseModule):
    def __init__(
        self,
        *,
        # ====== policies ======
        prop_obs_keys: List[str],
        pcd_sample_points: int = 1024,
        student_policy: Union[BasePolicy, DictConfig],
        # ====== learning ======
        lr: float,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        # ====== env creation ======
        rlg_task_cfg: Union[OmegaConf, dict],
        num_envs: int,
        display: bool,
        # ====== training data augmentation ======
        enable_pcd_augmentation: bool,
        pcd_aug_apply_prob: float,
        pcd_aug_random_trans_high: Tuple[float, float, float],
        pcd_aug_random_trans_low: Tuple[float, float, float],
        pcd_aug_jitter_ratio: float,
        pcd_aug_jitter_sigma: float,
        pcd_aug_jitter_low: float,
        pcd_aug_jitter_high: float,
        enable_prop_augmentation: bool,
        prop_aug_scale_sigma: float,
        prop_aug_scale_low: float,
        prop_aug_scale_high: float,
        # ====== eval ======
        n_eval_episodes: int,
        # ====== pcd regularization ======
        enable_pcd_matched_scenes_regularization: bool,
        pcd_matched_scenes_reg_weight: float,
        # ====== device ======
        sim_device: int,
        rl_device: int,
        graphics_device_id: int,
    ):
        super().__init__()
        self.prop_obs_keys = prop_obs_keys
        self.pcd_sample_points = pcd_sample_points
        if isinstance(student_policy, DictConfig):
            student_policy = instantiate(student_policy)
        self.student_policy = student_policy

        self._num_envs = num_envs
        self._env_display = display
        self._env_config = omegaconf_to_dict(rlg_task_cfg)

        self.enable_pcd_augmentation = enable_pcd_augmentation
        self.pcd_aug_apply_prob = pcd_aug_apply_prob
        self.pcd_aug_random_trans_high = torch.tensor(pcd_aug_random_trans_high).view(
            1, 1, 3
        )
        self.pcd_aug_random_trans_low = torch.tensor(pcd_aug_random_trans_low).view(
            1, 1, 3
        )
        self.pcd_aug_jitter_ratio = pcd_aug_jitter_ratio
        self.pcd_aug_jitter_sigma = pcd_aug_jitter_sigma
        self.pcd_aug_jitter_low = pcd_aug_jitter_low
        self.pcd_aug_jitter_high = pcd_aug_jitter_high
        self.pcd_aug_jitter_dist = None
        self.enable_prop_augmentation = enable_prop_augmentation
        self.prop_aug_scale_sigma = prop_aug_scale_sigma
        self.prop_aug_scale_low = prop_aug_scale_low
        self.prop_aug_scale_high = prop_aug_scale_high

        self.sim_device = str(torch_device(sim_device))
        self.rl_device = str(torch_device(rl_device))
        self.graphics_device_id = graphics_device_id

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay

        self.n_eval_episodes = n_eval_episodes
        self._eval_rewards = None

        self.enable_pcd_matched_scenes_regularization = (
            enable_pcd_matched_scenes_regularization
        )
        self.pcd_matched_scenes_reg_weight = pcd_matched_scenes_reg_weight

    def distillation_training_step(self, *args, **kwargs):
        if self.student_policy.is_sequence_policy:
            return self.distillation_training_step_seq_policy(*args, **kwargs)
        else:
            return self.distillation_training_step_non_seq_policy(*args, **kwargs)

    def distillation_test_step(self, *args, **kwargs):
        if self.student_policy.is_sequence_policy:
            return self.distillation_test_step_seq_policy(*args, **kwargs)
        else:
            return self.distillation_test_step_non_seq_policy(*args, **kwargs)

    def distillation_training_step_seq_policy(self, batch, batch_idx):
        if self.enable_pcd_matched_scenes_regularization:
            main_data, (
                (real_pcds, real_pcd_ee_masks),
                (sim_pcds, sim_pcd_ee_masks),
            ) = batch
        else:
            main_data = batch
        B, T = main_data["actions"].shape[1:3]
        # main data is dict of (N_chunks, B, ctx_len, ...)
        # we loop over chunk dim
        main_data = unstack_sequence_fields(
            main_data, batch_size=get_batch_size(main_data, strict=True)
        )
        all_loss, all_l1 = [], []
        real_batch_size = []
        for i, main_data_chunk in enumerate(main_data):
            policy_state = self.student_policy.get_initial_state(B, T)
            # get padding mask
            pad_mask = main_data_chunk.pop("pad_mask")
            # flatten first two dims in favor of recover pcd
            main_data_chunk = {
                k: v.reshape(-1, *v.shape[2:]) for k, v in main_data_chunk.items()
            }
            # recover point cloud
            pcd, ee_mask = self.env.recover_pcd_from_offline_data(
                state_dict=main_data_chunk,
            )  # (B * T, N_points, 3), (B * T, N_points)
            pcd, ee_mask = sample_points(pcd, ee_mask, self.pcd_sample_points)
            if self.enable_pcd_augmentation:
                pcd, ee_mask = apply_pc_aug_random_trans(
                    pcd,
                    ee_mask,
                    self.device,
                    self.pcd_aug_apply_prob,
                    self.pcd_aug_random_trans_high.to(self.device),
                    self.pcd_aug_random_trans_low.to(self.device),
                )
                if np.random.rand() < self.pcd_aug_apply_prob:
                    jitter_points = int(
                        self.pcd_aug_jitter_ratio * self.pcd_sample_points
                    )
                    if self.pcd_aug_jitter_dist is None:
                        jitter_std = torch.tensor(
                            [self.pcd_aug_jitter_sigma] * 3,
                            dtype=torch.float32,
                            device=self.device,
                        ).view(1, 3)
                        # repeat along n_points
                        jitter_std = jitter_std.repeat(jitter_points, 1)
                        jitter_mean = torch.zeros_like(jitter_std)
                        self.pcd_aug_jitter_dist = torch.distributions.normal.Normal(
                            jitter_mean, jitter_std
                        )
                    jitter_value = self.pcd_aug_jitter_dist.sample()
                    pcd, ee_mask = apply_pc_aug_jitter(
                        pcd,
                        ee_mask,
                        jitter_value,
                        jitter_points,
                        self.pcd_aug_jitter_low,
                        self.pcd_aug_jitter_high,
                    )
            # recover first two dims
            main_data_chunk = {
                k: v.reshape(B, T, *v.shape[1:]) for k, v in main_data_chunk.items()
            }
            pcd = pcd.reshape(B, T, *pcd.shape[1:])
            ee_mask = ee_mask.reshape(B, T, *ee_mask.shape[1:])

            main_data_chunk["pointcloud"] = pcd
            main_data_chunk["ee_mask"] = ee_mask
            main_data_chunk["cos_q"], main_data_chunk["sin_q"] = torch.cos(
                main_data_chunk["q"]
            ), torch.sin(main_data_chunk["q"])

            prop_obs = any_concat(
                [
                    any_to_torch_tensor(
                        main_data_chunk[k], device=self.device, dtype=torch.float32
                    )
                    for k in self.prop_obs_keys
                ],
                dim=-1,
            )
            if self.enable_prop_augmentation:
                prop_aug_scale_dist = torch.distributions.normal.Normal(
                    torch.zeros_like(prop_obs),
                    torch.ones_like(prop_obs) * self.prop_aug_scale_sigma,
                )
                prop_scale_value = prop_aug_scale_dist.sample()
                prop_scale_value = torch.clamp(
                    prop_scale_value,
                    min=self.prop_aug_scale_low,
                    max=self.prop_aug_scale_high,
                )
                prop_obs = prop_obs + prop_obs * prop_scale_value
            obs = {
                "pcd/coordinate": any_to_torch_tensor(
                    main_data_chunk["pointcloud"],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "pcd/ee_mask": any_to_torch_tensor(
                    main_data_chunk["ee_mask"],
                    device=self.device,
                    dtype=torch.long,
                ),
                "proprioception": prop_obs,
            }
            gt_q_actions, gt_gripper_actions = (
                main_data_chunk["actions"][..., :-1],
                main_data_chunk["actions"][..., -1:],
            )
            # clip q into valid range
            gt_q_actions = torch.clamp(
                gt_q_actions,
                min=self.franka_dof_lower_limits,
                max=self.franka_dof_upper_limits,
            )
            # normalize q to [-1, 1]
            gt_q_actions = (gt_q_actions - self.franka_dof_lower_limits) / (
                self.franka_dof_upper_limits - self.franka_dof_lower_limits
            ) * 2 - 1
            gt_gripper_actions = torch.clamp(gt_gripper_actions, min=-1.0, max=1.0)
            gt_actions = any_concat(
                [gt_q_actions, gt_gripper_actions], dim=-1
            )  # (B, L, 8)
            pi, policy_state = self.student_policy(obs, policy_state)
            action_loss = pi.imitation_loss(gt_actions, reduction="none").reshape(
                pad_mask.shape
            )
            # reduce the loss according to the action mask
            # "True" indicates should calculate the loss
            action_loss = action_loss * pad_mask
            all_loss.append(action_loss)
            # minus because imitation_accuracy returns negative l1 distance
            l1 = -pi.imitation_accuracy(gt_actions, pad_mask)
            all_l1.append(l1)
            real_batch_size.append(pad_mask.sum())
        real_batch_size = torch.sum(torch.stack(real_batch_size)).item()
        action_loss = torch.sum(torch.stack(all_loss)) / real_batch_size
        l1 = torch.mean(torch.stack(all_l1))
        log_dict = {"action_loss": action_loss, "l1": l1}

        if self.enable_pcd_matched_scenes_regularization:
            # (B, n_poses, n_points, 3) -> (B * n_poses, n_points, 3)
            real_pcds = real_pcds.reshape(-1, real_pcds.shape[-2], 3)
            sim_pcds = sim_pcds.reshape(-1, sim_pcds.shape[-2], 3)
            # (B, n_poses, n_points) -> (B * n_poses, n_points)
            real_pcd_ee_masks = real_pcd_ee_masks.reshape(
                -1, real_pcd_ee_masks.shape[-1]
            )
            sim_pcd_ee_masks = sim_pcd_ee_masks.reshape(-1, sim_pcd_ee_masks.shape[-1])
            real_pcd_features = self.student_policy.feature_extractor._extractors[
                "pcd"
            ](
                {"coordinate": real_pcds, "ee_mask": real_pcd_ee_masks}
            )  # (N, D)
            sim_pcd_features = self.student_policy.feature_extractor._extractors["pcd"](
                {"coordinate": sim_pcds, "ee_mask": sim_pcd_ee_masks}
            )
            reg_loss = torch.linalg.norm(
                real_pcd_features - sim_pcd_features, dim=-1
            ).mean()
            loss = action_loss + self.pcd_matched_scenes_reg_weight * reg_loss
            log_dict["reg_loss"] = reg_loss
        else:
            loss = action_loss
        return loss, log_dict, real_batch_size

    def distillation_training_step_non_seq_policy(self, batch, batch_idx):
        if self.enable_pcd_matched_scenes_regularization:
            main_data, (
                (real_pcds, real_pcd_ee_masks),
                (sim_pcds, sim_pcd_ee_masks),
            ) = batch
        else:
            main_data = batch
        # recover point cloud
        pcd, ee_mask = self.env.recover_pcd_from_offline_data(
            state_dict=main_data,
        )
        pcd, ee_mask = sample_points(pcd, ee_mask, self.pcd_sample_points)
        if self.enable_pcd_augmentation:
            pcd, ee_mask = apply_pc_aug_random_trans(
                pcd,
                ee_mask,
                self.device,
                self.pcd_aug_apply_prob,
                self.pcd_aug_random_trans_high.to(self.device),
                self.pcd_aug_random_trans_low.to(self.device),
            )
            if np.random.rand() < self.pcd_aug_apply_prob:
                jitter_points = int(self.pcd_aug_jitter_ratio * self.pcd_sample_points)
                if self.pcd_aug_jitter_dist is None:
                    jitter_std = torch.tensor(
                        [self.pcd_aug_jitter_sigma] * 3,
                        dtype=torch.float32,
                        device=self.device,
                    ).view(1, 3)
                    # repeat along n_points
                    jitter_std = jitter_std.repeat(jitter_points, 1)
                    jitter_mean = torch.zeros_like(jitter_std)
                    self.pcd_aug_jitter_dist = torch.distributions.normal.Normal(
                        jitter_mean, jitter_std
                    )
                jitter_value = self.pcd_aug_jitter_dist.sample()
                pcd, ee_mask = apply_pc_aug_jitter(
                    pcd,
                    ee_mask,
                    jitter_value,
                    jitter_points,
                    self.pcd_aug_jitter_low,
                    self.pcd_aug_jitter_high,
                )

        main_data["pointcloud"] = pcd
        main_data["ee_mask"] = ee_mask
        main_data["cos_q"], main_data["sin_q"] = torch.cos(main_data["q"]), torch.sin(
            main_data["q"]
        )

        prop_obs = any_concat(
            [
                any_to_torch_tensor(
                    main_data[k], device=self.device, dtype=torch.float32
                )
                for k in self.prop_obs_keys
            ],
            dim=-1,
        )
        if self.enable_prop_augmentation:
            prop_aug_scale_dist = torch.distributions.normal.Normal(
                torch.zeros_like(prop_obs),
                torch.ones_like(prop_obs) * self.prop_aug_scale_sigma,
            )
            prop_scale_value = prop_aug_scale_dist.sample()
            prop_scale_value = torch.clamp(
                prop_scale_value,
                min=self.prop_aug_scale_low,
                max=self.prop_aug_scale_high,
            )
            prop_obs = prop_obs + prop_obs * prop_scale_value
        obs = {
            "pcd/coordinate": any_to_torch_tensor(
                main_data["pointcloud"], device=self.device, dtype=torch.float32
            ),
            "pcd/ee_mask": any_to_torch_tensor(
                main_data["ee_mask"],
                device=self.device,
                dtype=torch.long,
            ),
            "proprioception": prop_obs,
        }
        gt_q_actions, gt_gripper_actions = (
            main_data["actions"][..., :-1],
            main_data["actions"][..., -1:],
        )
        # clip q into valid range
        gt_q_actions = torch.clamp(
            gt_q_actions,
            min=self.franka_dof_lower_limits,
            max=self.franka_dof_upper_limits,
        )
        # normalize q to [-1, 1]
        gt_q_actions = (gt_q_actions - self.franka_dof_lower_limits) / (
            self.franka_dof_upper_limits - self.franka_dof_lower_limits
        ) * 2 - 1
        gt_gripper_actions = torch.clamp(gt_gripper_actions, min=-1.0, max=1.0)
        gt_actions = any_concat([gt_q_actions, gt_gripper_actions], dim=-1)  # (..., 8)
        pi = self.student_policy(obs)
        action_loss = pi.imitation_loss(gt_actions, reduction="none")
        action_loss = action_loss.mean()
        log_dict = {"action_loss": action_loss}
        if self.enable_pcd_matched_scenes_regularization:
            # (B, n_poses, n_points, 3) -> (B * n_poses, n_points, 3)
            real_pcds = real_pcds.reshape(-1, real_pcds.shape[-2], 3)
            sim_pcds = sim_pcds.reshape(-1, sim_pcds.shape[-2], 3)
            # (B, n_poses, n_points) -> (B * n_poses, n_points)
            real_pcd_ee_masks = real_pcd_ee_masks.reshape(
                -1, real_pcd_ee_masks.shape[-1]
            )
            sim_pcd_ee_masks = sim_pcd_ee_masks.reshape(-1, sim_pcd_ee_masks.shape[-1])
            real_pcd_features = self.student_policy.feature_extractor._extractors[
                "pcd"
            ](
                {"coordinate": real_pcds, "ee_mask": real_pcd_ee_masks}
            )  # (N, D)
            sim_pcd_features = self.student_policy.feature_extractor._extractors["pcd"](
                {"coordinate": sim_pcds, "ee_mask": sim_pcd_ee_masks}
            )
            reg_loss = torch.linalg.norm(
                real_pcd_features - sim_pcd_features, dim=-1
            ).mean()
            loss = action_loss + self.pcd_matched_scenes_reg_weight * reg_loss
            log_dict["reg_loss"] = reg_loss
        else:
            loss = action_loss
        # minus because imitation_accuracy returns negative l1 distance
        l1 = -pi.imitation_accuracy(gt_actions)
        log_dict["l1"] = l1
        return loss, log_dict, len(main_data["q"])

    @torch.no_grad()
    def distillation_test_step_seq_policy(self, *args, **kwargs):
        self.env.disable_pointcloud_augmentation()
        self._eval_rewards = torch.zeros((self._num_envs,), device=self.device)
        step_count = torch.zeros((self._num_envs,), device=self.device)
        obs = self.env.reset()
        returns, episode_count = [], 0
        success, failures = [], []
        policy_state = self.student_policy.get_initial_state(self.env.num_envs, 1)
        # returned obs when done is True is actually terminal obs
        # so need to maintain prev_done to properly reset policy state
        prev_done = None
        while True:
            # handle the case where we need to reset RNN state based on its horizon length
            reset_state_indices = (
                step_count % self.student_policy.ctx_len == 0
            ).nonzero(as_tuple=False)[:, 0]
            if len(reset_state_indices) > 0:
                reset_state = self.student_policy.get_initial_state(
                    len(reset_state_indices), 1
                )
                policy_state = self.student_policy.update_state(
                    old_state=policy_state,
                    new_state=reset_state,
                    idxs=reset_state_indices,
                )

            if prev_done is not None and len(prev_done.nonzero(as_tuple=False)) > 0:
                prev_done_indices = prev_done.nonzero(as_tuple=False)[
                    :, 0
                ]  # (num_done, 1) -> (num_done,)
                zero_policy_state = self.student_policy.get_initial_state(
                    len(prev_done_indices), 1
                )
                policy_state = self.student_policy.update_state(
                    old_state=policy_state,
                    new_state=zero_policy_state,
                    idxs=prev_done_indices,
                )
                step_count[prev_done_indices] = 0
            student_policy_obs = any_to_datadict(
                {
                    k: v.clone()
                    for k, v in obs.items()
                    if k in {"pcd/coordinate", "pcd/ee_mask", "proprioception"}
                }
            ).to_torch_tensor(device=self.device)
            # add time dimension
            student_policy_obs = {
                k: v.unsqueeze(1) for k, v in student_policy_obs.items()
            }  # (B, 1, ...)
            actions, policy_state = self.student_policy.act(
                student_policy_obs, policy_state
            )  # (B, 8)
            step_count += 1
            actions = torch.clamp(actions, min=-1.0, max=1.0)
            q_actions, gripper_actions = actions[..., :7], actions[..., 7:]
            q_actions = (q_actions + 1) / 2 * (
                self.franka_dof_upper_limits - self.franka_dof_lower_limits
            ) + self.franka_dof_lower_limits
            actions = any_concat([q_actions, gripper_actions], dim=-1)
            obs, reward, done, _ = self.env.step(actions)
            prev_done = done.clone()
            self._eval_rewards += reward
            done_indices = done.nonzero(as_tuple=False)
            if len(done_indices) > 0:
                returns.append(self._eval_rewards[done_indices])
                success.append(self.env.success_buf[done_indices])
                failures.append(self.env.failure_buf[done_indices])
                self._eval_rewards[done_indices] = 0.0
                episode_count += len(done_indices)
            if episode_count >= self.n_eval_episodes:
                break
        returns = torch.cat(returns, dim=0)[: self.n_eval_episodes].mean()
        success = torch.cat(success, dim=0)[: self.n_eval_episodes].float().mean()
        failures = torch.cat(failures, dim=0)[: self.n_eval_episodes].float().mean()
        return torch.zeros_like(returns), {
            "return": returns,
            "success_rate": success,
            "failure_rate": failures,
        }

    @torch.no_grad()
    def distillation_test_step_non_seq_policy(self, *args, **kwargs):
        self.env.disable_pointcloud_augmentation()
        self._eval_rewards = torch.zeros((self._num_envs,), device=self.device)
        obs = self.env.reset()
        returns, episode_count = [], 0
        success, failures = [], []
        while True:
            student_policy_obs = any_to_datadict(
                {
                    k: v.clone()
                    for k, v in obs.items()
                    if k in {"pcd/coordinate", "pcd/ee_mask", "proprioception"}
                }
            ).to_torch_tensor(device=self.device)
            actions = self.student_policy.act(student_policy_obs)  # (B, 8)
            actions = torch.clamp(actions, min=-1.0, max=1.0)
            q_actions, gripper_actions = actions[..., :7], actions[..., 7:]
            q_actions = (q_actions + 1) / 2 * (
                self.franka_dof_upper_limits - self.franka_dof_lower_limits
            ) + self.franka_dof_lower_limits
            actions = any_concat([q_actions, gripper_actions], dim=-1)
            obs, reward, done, _ = self.env.step(actions)
            self._eval_rewards += reward
            done_indices = done.nonzero(as_tuple=False)
            if len(done_indices) > 0:
                returns.append(self._eval_rewards[done_indices])
                success.append(self.env.success_buf[done_indices])
                failures.append(self.env.failure_buf[done_indices])
                self._eval_rewards[done_indices] = 0.0
                episode_count += len(done_indices)
            if episode_count >= self.n_eval_episodes:
                break
        returns = torch.cat(returns, dim=0)[: self.n_eval_episodes].mean()
        success = torch.cat(success, dim=0)[: self.n_eval_episodes].float().mean()
        failures = torch.cat(failures, dim=0)[: self.n_eval_episodes].float().mean()
        return torch.zeros_like(returns), {
            "return": returns,
            "success_rate": success,
            "failure_rate": failures,
        }

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError
        return optimizer

    def create_env(self) -> TRANSICEnvPCD:
        self._eval_rewards = torch.zeros((self._num_envs,), device=self.device)

        def create_transic_env():
            envs = transic_envs.make(
                sim_device=str(self.device),
                rl_device=str(self.device),
                graphics_device_id=self.device.index,
                multi_gpu=False,
                cfg=self._env_config,
                display=self._env_display,
                record=False,
            )
            return envs

        env_configurations.register(
            "rlgpu",
            {
                "vecenv_type": "RLGPU",
                "env_creator": create_transic_env,
            },
        )
        env = env_configurations.configurations["rlgpu"]["env_creator"]()
        self.franka_dof_upper_limits = env.franka_dof_upper_limits[:7]
        self.franka_dof_lower_limits = env.franka_dof_lower_limits[:7]
        return env
