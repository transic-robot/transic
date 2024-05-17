import torch

from transic.learn.policy import GMMHead, BasePolicy
from transic.nn.features import SimpleFeatureFusion, PointNet, Identity


class PointNetPolicy(BasePolicy):
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
        deterministic_inference: bool = True,
        gmm_low_noise_eval: bool = True,
    ):
        super().__init__()

        self.feature_extractor = SimpleFeatureFusion(
            extractors={
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
            },
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
        self._deterministic_inference = deterministic_inference

    def forward(self, obs):
        return self.action_net(self.feature_extractor(obs))

    @torch.no_grad()
    def act(self, obs, deterministic=None):
        dist = self.forward(obs)
        if deterministic is None:
            deterministic = self._deterministic_inference
        if deterministic:
            return dist.mode()
        else:
            return dist.sample()
