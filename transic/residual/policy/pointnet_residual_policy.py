import os

import torch

from transic.learn.policy import GMMHead, CategoricalNet
from transic.learn.policy import BasePolicy
from transic.nn.features import SimpleFeatureFusion, PointNet, Identity, Embedding
from transic.utils.torch_utils import load_state_dict, freeze_params


class PointNetResidualPolicy(BasePolicy):
    def __init__(
        self,
        *,
        point_channels: int = 3,
        subtract_point_mean: bool = False,
        add_ee_embd: bool = False,
        ee_embd_dim: int,
        pointnet_output_dim: int,
        pointnet_hidden_dim: int,
        pointnet_hidden_depth: int,
        pointnet_activation: str = "gelu",
        prop_input_dim: int,
        robot_policy_output_dim: int,
        include_robot_policy_gripper_action_input: bool,
        robot_policy_gripper_action_embd_dim: int,
        feature_fusion_hidden_depth: int = 1,
        feature_fusion_hidden_dim: int = 256,
        feature_fusion_output_dim: int = 256,
        feature_fusion_activation: str = "relu",
        feature_fusion_add_input_activation: bool = False,
        feature_fusion_add_output_activation: bool = False,
        action_dim: int,
        action_net_gmm_n_modes: int = 5,
        action_net_hidden_dim: int,
        action_net_hidden_depth: int,
        action_net_activation: str = "relu",
        intervention_head_hidden_dim: int,
        intervention_head_hidden_depth: int,
        intervention_head_activation: str = "relu",
        deterministic_inference: bool = True,
        gmm_low_noise_eval: bool = True,
        update_intervention_head_only: bool = False,
        ckpt_path_if_update_intervention_head_only: str = None,
    ):
        super().__init__()

        extractors = {
            "pcd": PointNet(
                n_coordinates=point_channels,
                add_ee_embd=add_ee_embd,
                ee_embd_dim=ee_embd_dim,
                output_dim=pointnet_output_dim,
                hidden_dim=pointnet_hidden_dim,
                hidden_depth=pointnet_hidden_depth,
                activation=pointnet_activation,
                subtract_mean=subtract_point_mean,
            ),
            "proprioception": Identity(prop_input_dim),
            "robot_policy_action": Identity(robot_policy_output_dim),
        }
        if include_robot_policy_gripper_action_input:
            extractors["robot_policy_gripper_action"] = Embedding(
                num_embeddings=2,  # open/close
                embedding_dim=robot_policy_gripper_action_embd_dim,
            )
        self.feature_extractor = SimpleFeatureFusion(
            extractors=extractors,
            hidden_depth=feature_fusion_hidden_depth,
            hidden_dim=feature_fusion_hidden_dim,
            output_dim=feature_fusion_output_dim,
            activation=feature_fusion_activation,
            add_input_activation=feature_fusion_add_input_activation,
            add_output_activation=feature_fusion_add_output_activation,
        )

        self.action_net = GMMHead(
            feature_fusion_output_dim,
            n_modes=action_net_gmm_n_modes,
            action_dim=action_dim,
            hidden_dim=action_net_hidden_dim,
            hidden_depth=action_net_hidden_depth,
            activation=action_net_activation,
            low_noise_eval=gmm_low_noise_eval,
        )
        self.intervention_head = CategoricalNet(
            feature_fusion_output_dim,
            action_dim=2,  # intervention or not
            hidden_dim=intervention_head_hidden_dim,
            hidden_depth=intervention_head_hidden_depth,
            activation=intervention_head_activation,
        )
        if update_intervention_head_only:
            assert os.path.exists(ckpt_path_if_update_intervention_head_only)
            ckpt = torch.load(
                ckpt_path_if_update_intervention_head_only, map_location="cpu"
            )

            feature_extractor_weighs = {
                k: v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("residual_policy.feature_extractor")
            }
            load_state_dict(
                self.feature_extractor,
                feature_extractor_weighs,
                strip_prefix="residual_policy.feature_extractor.",
                strict=True,
            )
            freeze_params(self.feature_extractor)

            action_net_weights = {
                k: v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("residual_policy.action_net")
            }
            load_state_dict(
                self.action_net,
                action_net_weights,
                strip_prefix="residual_policy.action_net.",
                strict=True,
            )
            freeze_params(self.action_net)

        self._deterministic_inference = deterministic_inference

    def forward(self, obs):
        feature = self.feature_extractor(obs)
        action_dist = self.action_net(feature)
        intervention_dist = self.intervention_head(feature)
        return action_dist, intervention_dist

    @torch.no_grad()
    def act(self, obs, deterministic=None):
        action_dist, intervention_dist = self.forward(obs)
        if deterministic is None:
            deterministic = self._deterministic_inference
        if deterministic:
            return action_dist.mode(), intervention_dist.mode()
        else:
            return action_dist.sample(), intervention_dist.sample()
