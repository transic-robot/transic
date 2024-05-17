import torch
import torch.nn as nn

from transic.nn.mlp import build_mlp
from transic.learn.optimizer_group import default_optimizer_groups


class _PointNetSimplified(nn.Module):
    def __init__(
        self,
        *,
        point_channels: int = 3,
        output_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "gelu",
    ):
        super().__init__()
        self._mlp = build_mlp(
            input_dim=point_channels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
            activation=activation,
        )
        self.output_dim = output_dim

    def forward(self, x):
        """
        x: (..., points, point_channels)
        """
        x = self._mlp(x)  # (..., points, output_dim)
        x = torch.max(x, dim=-2)[0]  # (..., output_dim)
        return x


class PointNet(nn.Module):
    def __init__(
        self,
        *,
        n_coordinates: int = 3,
        add_ee_embd: bool = False,
        ee_embd_dim: int = 128,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_depth: int = 2,
        activation: str = "gelu",
        subtract_mean: bool = False,
    ):
        super().__init__()
        pn_in_channels = n_coordinates
        if add_ee_embd:
            pn_in_channels += ee_embd_dim
        if subtract_mean:
            pn_in_channels += n_coordinates
        self.pointnet = _PointNetSimplified(
            point_channels=pn_in_channels,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
        )
        self.ee_embd_layer = None
        if add_ee_embd:
            self.ee_embd_layer = nn.Embedding(2, embedding_dim=ee_embd_dim)
        self.add_ee_embd = add_ee_embd
        self.subtract_mean = subtract_mean
        self.output_dim = self.pointnet.output_dim

    def forward(self, x):
        """
        x["coordinate"]: (..., points, coordinates)
        """
        point = x["coordinate"]
        ee_mask = x.get("ee_mask", None)
        if self.subtract_mean:
            mean = torch.mean(point, dim=-2, keepdim=True)  # (..., 1, coordinates)
            mean = torch.broadcast_to(mean, point.shape)  # (..., points, coordinates)
            point = point - mean
            point = torch.cat([point, mean], dim=-1)  # (..., points, 2 * coordinates)
        if self.add_ee_embd:
            ee_mask = torch.tensor(ee_mask, dtype=torch.long)  # (..., points)
            ee_embd = self.ee_embd_layer(ee_mask)  # (..., points, ee_embd_dim)
            x = torch.concat(
                [point, ee_embd], dim=-1
            )  # (..., points, coordinates + ee_embd_dim)
        return self.pointnet(x)

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        pg, pids = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=["ee_embd_layer.*"],
        )
        return pg, pids
