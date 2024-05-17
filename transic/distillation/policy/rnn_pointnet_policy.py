from typing import Literal, Optional

import torch
import torch.nn as nn

from transic.learn.policy import GMMHead, BasePolicy
from transic.nn.features import SimpleFeatureFusion, PointNet, Identity
from transic.utils.array import get_batch_size, any_slice


RNN_CLS = {
    "lstm": nn.LSTM,
    "gru": nn.GRU,
}


class RNNPointNetPolicy(BasePolicy):
    is_sequence_policy = True

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
        rnn_type: Literal["lstm", "gru"],
        rnn_n_layers: int = 2,
        rnn_hidden_dim: int = 256,
        ctx_len: int,
        action_dim: int,
        action_net_gmm_n_modes: int = 5,
        action_net_hidden_dim: int,
        action_net_hidden_depth: int,
        action_net_activation: str = "relu",
        deterministic_inference: bool = True,
        gmm_low_noise_eval: bool = True,
    ):
        super().__init__()

        self.ctx_len = ctx_len

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

        assert rnn_type in ["lstm", "gru"]
        rnn_cls = RNN_CLS[rnn_type]
        self.rnn = rnn_cls(
            input_size=feature_fusion_output_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_n_layers,
            batch_first=True,
        )

        self.action_net = GMMHead(
            rnn_hidden_dim,
            n_modes=action_net_gmm_n_modes,
            action_dim=action_dim,
            hidden_dim=action_net_hidden_dim,
            hidden_depth=action_net_hidden_depth,
            activation=action_net_activation,
            low_noise_eval=gmm_low_noise_eval,
        )
        self._deterministic_inference = deterministic_inference

    def get_initial_state(self, batch_size: int, timesteps: Optional[int] = None):
        h_0 = torch.zeros(
            self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=self.device
        )
        if isinstance(self.rnn, nn.LSTM):
            c_0 = torch.zeros_like(h_0)
            return h_0, c_0
        return h_0

    def update_state(self, *, old_state, new_state, idxs):
        if isinstance(self.rnn, nn.LSTM):
            h_old, c_old = old_state
            h_new, c_new = new_state
            h_old[:, idxs] = h_new
            c_old[:, idxs] = c_new
            return h_old, c_old
        elif isinstance(self.rnn, nn.GRU):
            old_state[:, idxs] = new_state
            return old_state
        else:
            raise NotImplementedError(f"Unknown RNN type {type(self.rnn)}")

    def forward(self, obs, policy_state):
        """
        obs: dict of (B, L, ...)
        rnn_state: (h_0, c_0) or h_0
        """
        x = self.feature_extractor(obs)
        x, policy_state = self.rnn(x, policy_state)
        return self.action_net(x), policy_state

    @torch.no_grad()
    def act(self, obs, policy_state, deterministic=None):
        """
        obs: dict of (B, L=1, ...)
        rnn_state: (h_0, c_0) or h_0
        """
        assert get_batch_size(any_slice(obs, 0), strict=True) == 1, "Use L=1 for act"
        dist, policy_state = self.forward(obs, policy_state)
        if deterministic is None:
            deterministic = self._deterministic_inference
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        # action is (B, L=1, A), reduce to (B, A)
        action = action[:, 0]
        return action, policy_state
