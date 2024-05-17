import numpy as np
import torch
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from hydra.utils import instantiate

from transic.learn.lr_schedule import CosineScheduleFunction
from transic.utils.array import any_to_torch_tensor

franka_dof_upper_limits = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
franka_dof_lower_limits = [
    -2.8973,
    -1.7628,
    -2.8973,
    -3.0718,
    -2.8973,
    -0.0175,
    -2.8973,
]


class ResidualPolicyModule(LightningModule):
    def __init__(
        self,
        *,
        # ====== policy ======
        residual_policy,
        include_robot_gripper_action_input: bool = True,
        learn_gripper_action: bool = True,
        # ====== learning ======
        lr: float,
        use_cosine_lr: bool = True,
        lr_warmup_steps: int,
        lr_cosine_steps: int,
        lr_cosine_min: float,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        intervention_pred_loss_weight: float = 1.0,
        # ====== pcd sampling ======
        pcd_downsample_N: int = None,
    ):
        super().__init__()
        if isinstance(residual_policy, DictConfig):
            residual_policy = instantiate(residual_policy)
        self.residual_policy = residual_policy
        self.include_robot_gripper_action_input = include_robot_gripper_action_input
        self.learn_gripper_action = learn_gripper_action

        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.intervention_loss_weight = intervention_pred_loss_weight

        # ====== pcd sampling ======
        self.pcd_downsample_N = pcd_downsample_N

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

        if self.use_cosine_lr:
            scheduler_kwargs = dict(
                base_value=1.0,  # anneal from the original LR value
                final_value=self.lr_cosine_min / self.lr,
                epochs=self.lr_cosine_steps,
                warmup_start_value=self.lr_cosine_min / self.lr,
                warmup_epochs=self.lr_warmup_steps,
                steps_per_epoch=1,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=CosineScheduleFunction(**scheduler_kwargs),
            )
            return (
                [optimizer],
                [{"scheduler": scheduler, "interval": "step"}],
            )

        return optimizer

    def training_step(self, *args, **kwargs):
        loss, bs, log_dict = self.residual_forward_step(*args, **kwargs, is_train=True)
        log_dict = {f"train/{k}": v for k, v in log_dict.items()}
        log_dict["train/loss"] = loss
        self.log_dict(
            log_dict, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs
        )
        return loss

    def validation_step(self, *args, **kwargs):
        with torch.no_grad():
            loss, bs, log_dict = self.residual_forward_step(
                *args, **kwargs, is_train=False
            )
        log_dict = {f"val/{k}": v for k, v in log_dict.items()}
        log_dict["val/loss"] = loss
        self.log_dict(
            log_dict, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs
        )
        return loss

    def residual_forward_step(self, batch, batch_idx, is_train: bool):
        """
        batch data are (n_chunks, B, ctx_len, ...)
        """

        batch["policy_obs_proprioception"] = any_to_torch_tensor(
            batch["policy_obs_proprioception"],
            device=self.device,
            dtype=torch.float32,
        )
        batch["policy_obs_pcd_coordinate"] = any_to_torch_tensor(
            batch["policy_obs_pcd_coordinate"],
            device=self.device,
            dtype=torch.float32,
        )
        batch["policy_obs_pcd_ee_mask"] = any_to_torch_tensor(
            batch["policy_obs_pcd_ee_mask"],
            device=self.device,
            dtype=torch.long,
        )

        robot_policy_action = batch["policy_action"]  # (n_chunks, B, T, A)
        # separate q action and gripper action
        robot_policy_action, robot_policy_gripper_action = (
            robot_policy_action[..., :-1],
            robot_policy_action[..., -1],
        )
        # rectify gripper action from [-1, 1] to {0, 1}
        robot_policy_gripper_action = torch.where(
            robot_policy_gripper_action >= 0, 1, 0
        )
        residual_q = batch["residual_q"]

        q_upper_limits = any_to_torch_tensor(
            franka_dof_upper_limits, device=self.device
        )
        q_lower_limits = any_to_torch_tensor(
            franka_dof_lower_limits, device=self.device
        )
        delta_q_upper_limits = q_upper_limits - q_lower_limits
        delta_q_lower_limits = q_lower_limits - q_upper_limits

        # clip residual_q_ into valid range
        residual_q = torch.clamp(
            residual_q,
            min=delta_q_lower_limits,
            max=delta_q_upper_limits,
        )
        # normalize residual_q to [-1, 1]
        residual_q = (residual_q - delta_q_lower_limits) / (
            delta_q_upper_limits - delta_q_lower_limits
        ) * 2 - 1
        if self.learn_gripper_action:
            # gripper change action
            gripper_change_action = batch["gripper_change_action"]
            # gripper change action is {0, 1}, we convert to {-1, 1} so we can use GMM to optimize both q and gripper action
            gripper_change_action = torch.where(gripper_change_action == 0, -1, 1)
            target_action = torch.cat([residual_q, gripper_change_action], dim=-1)
        else:
            target_action = residual_q

        # add data augmentation to residual policy input if necessary
        policy_obs_pcd_coordinate = batch[
            "policy_obs_pcd_coordinate"
        ]  # (n_chunks, B, ctx_len, n_points, 3)
        policy_obs_pcd_ee_mask = batch["policy_obs_pcd_ee_mask"]
        policy_obs_pcd_pad_mask = batch["policy_obs_pcd_pad_mask"]
        policy_obs_proprioception = batch["policy_obs_proprioception"]

        robot_policy_action_chunks = any_to_torch_tensor(
            robot_policy_action,
            device=self.device,
            dtype=torch.float32,
        )
        robot_policy_gripper_action_chunks = any_to_torch_tensor(
            robot_policy_gripper_action,
            device=self.device,
            dtype=torch.long,
        )
        human_intervention_mask_chunks = any_to_torch_tensor(
            batch["is_human_intervention"],
            device=self.device,
            dtype="bool",
        )
        pad_mask_chunks = any_to_torch_tensor(
            batch["pad_mask"],
            device=self.device,
            dtype="bool",
        )
        # we loop over chunks
        all_action_loss = []
        all_intervention_loss = []
        all_intervention_acc = []
        real_batch_size = []
        B, T = policy_obs_pcd_coordinate.shape[1:3]
        for i, (
            pcd_coordinate,
            pcd_ee_mask,
            pcd_pad_mask,
            proprioception,
            robot_action,
            robot_gripper_action,
            target_action_chunk,
            intervention_mask,
            pad_mask,
        ) in enumerate(
            zip(
                policy_obs_pcd_coordinate,
                policy_obs_pcd_ee_mask,
                policy_obs_pcd_pad_mask,
                policy_obs_proprioception,
                robot_policy_action_chunks,
                robot_policy_gripper_action_chunks,
                target_action,
                human_intervention_mask_chunks,
                pad_mask_chunks,
            )
        ):
            if not self.residual_policy.is_sequence_policy:
                pcd_coordinate = pcd_coordinate.view(
                    -1, *pcd_coordinate.shape[2:]
                )  # (B*T, ...)
                pcd_ee_mask = pcd_ee_mask.view(-1, *pcd_ee_mask.shape[2:])  # (B*T, ...)
                pcd_pad_mask = pcd_pad_mask.view(-1, *pcd_pad_mask.shape[2:])
                proprioception = proprioception.view(
                    -1, *proprioception.shape[2:]
                )  # (B*T, ...)
                robot_action = robot_action.view(
                    -1, *robot_action.shape[2:]
                )  # (B*T, ...)
                robot_gripper_action = robot_gripper_action.view(
                    -1, *robot_gripper_action.shape[2:]
                )  # (B*T, ...)
                target_action_chunk = target_action_chunk.view(
                    -1, *target_action_chunk.shape[2:]
                )  # (B*T, ...)
                intervention_mask = intervention_mask.view(
                    -1, *intervention_mask.shape[2:]
                )  # (B*T, ...)
                pad_mask = pad_mask.view(-1, *pad_mask.shape[2:])  # (B*T, ...)
                # only action valid when both intervention and pad mask are True
                action_valid_mask = intervention_mask & pad_mask
                # P_intervention << P_no_intervention
                # so we balance data
                intervention_pred_valid_mask = pad_mask.clone()
                if is_train:
                    intervention_pred_valid_mask[~intervention_mask] = torch.rand(
                        intervention_pred_valid_mask[~intervention_mask].shape,
                        device=self.device,
                    ) >= (1 - self.data_module.P_intervention)
                obs = {
                    "pcd/coordinate": pcd_coordinate,
                    "pcd/ee_mask": pcd_ee_mask,
                    "pcd/pad_mask": pcd_pad_mask,
                    "proprioception": proprioception,
                    "robot_policy_action": robot_action,
                }
                if self.include_robot_gripper_action_input:
                    obs["robot_policy_gripper_action"] = robot_gripper_action
                pi, intervention_dist = self.residual_policy(obs)
                raw_action_loss = pi.imitation_loss(
                    target_action_chunk, reduction="none"
                ).reshape(action_valid_mask.shape)
                action_loss = raw_action_loss * action_valid_mask
                all_action_loss.append(action_loss)
                real_batch_size.append(action_valid_mask.sum())
                raw_intervention_loss = intervention_dist.imitation_loss(
                    intervention_mask.long(), reduction="none"
                ).reshape(intervention_pred_valid_mask.shape)
                intervention_loss = raw_intervention_loss * intervention_pred_valid_mask
                all_intervention_loss.append(intervention_loss)
                intervention_pred_acc = intervention_dist.imitation_accuracy(
                    intervention_mask.long(),
                    mask=intervention_pred_valid_mask,
                )
                all_intervention_acc.append(intervention_pred_acc)
            else:
                # only valid when both intervention and pad mask are True
                action_valid_mask = intervention_mask & pad_mask
                # P_intervention << P_no_intervention
                # so we balance data
                intervention_pred_valid_mask = pad_mask.clone()
                if is_train:
                    intervention_pred_valid_mask[~intervention_mask] = torch.rand(
                        intervention_pred_valid_mask[~intervention_mask].shape,
                        device=self.device,
                    ) >= (1 - self.data_module.P_intervention)
                policy_state = self.residual_policy.get_initial_state(B, T)
                obs = {
                    "pcd/coordinate": pcd_coordinate,
                    "pcd/ee_mask": pcd_ee_mask,
                    "pcd/pad_mask": pcd_pad_mask,
                    "proprioception": proprioception,
                    "robot_policy_action": robot_action,
                }
                if self.include_robot_gripper_action_input:
                    obs["robot_policy_gripper_action"] = robot_gripper_action
                pi, intervention_dist, policy_state = self.residual_policy(
                    obs, policy_state
                )
                raw_action_loss = pi.imitation_loss(
                    target_action_chunk, reduction="none"
                ).reshape(action_valid_mask.shape)
                action_loss = raw_action_loss * action_valid_mask
                all_action_loss.append(action_loss)
                real_batch_size.append(action_valid_mask.sum())
                raw_intervention_loss = intervention_dist.imitation_loss(
                    intervention_mask.long(), reduction="none"
                ).reshape(intervention_pred_valid_mask.shape)
                intervention_loss = raw_intervention_loss * intervention_pred_valid_mask
                all_intervention_loss.append(intervention_loss)
                intervention_pred_acc = intervention_dist.imitation_accuracy(
                    intervention_mask.long(),
                    mask=intervention_pred_valid_mask,
                )
                all_intervention_acc.append(intervention_pred_acc)
        real_batch_size = torch.sum(torch.stack(real_batch_size)).item()
        action_loss = torch.sum(torch.stack(all_action_loss)) / real_batch_size
        intervention_loss = (
            torch.sum(torch.stack(all_intervention_loss)) / pad_mask_chunks.sum()
        )
        intervention_acc = np.mean(all_intervention_acc)
        loss = action_loss + self.intervention_loss_weight * intervention_loss
        log_dict = {
            "action_loss": action_loss,
            "intervention_loss": intervention_loss,
            "intervention_acc": intervention_acc,
        }
        return loss, real_batch_size, log_dict
