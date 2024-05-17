import torch
import torch.nn as nn

from transic.nn.features.pointcloud.set_transformer.set_transformer import (
    PoolingSetAttention,
)


class SetXFPCDEncoder(nn.Module):
    def __init__(
        self,
        *,
        n_coordinates: int = 3,
        add_ee_embd: bool = False,
        ee_embd_dim: int = 128,
        hidden_dim: int = 512,
        subtract_mean: bool = False,
        set_xf_num_heads: int = 8,
        set_xf_num_queries: int = 8,
        set_xf_pool_type,
        set_xf_layer_norm: bool = False,
    ):
        super().__init__()
        pn_in_channels = n_coordinates
        if add_ee_embd:
            pn_in_channels += ee_embd_dim
        if subtract_mean:
            pn_in_channels += n_coordinates

        self.linear = nn.Linear(pn_in_channels, hidden_dim)
        self.num_queries = set_xf_num_queries
        self.set_xf = PoolingSetAttention(
            dim=hidden_dim,
            num_heads=set_xf_num_heads,
            num_queries=set_xf_num_queries,
            pool_type=set_xf_pool_type,
            layer_norm=set_xf_layer_norm,
        )
        self.ee_embd_layer = None
        if add_ee_embd:
            self.ee_embd_layer = nn.Embedding(2, embedding_dim=ee_embd_dim)
        self.add_ee_embd = add_ee_embd
        self.subtract_mean = subtract_mean
        if set_xf_pool_type == "concat":
            self.output_dim = hidden_dim * set_xf_num_queries
        else:
            self.output_dim = hidden_dim

    def forward(self, x):
        """
        x["coordinate"]: (..., points, coordinates)
        x["ee_mask"]: (..., points) if present
        x["pad_mask"]: (..., points) if present (for variable length point clouds)
        """
        point = x["coordinate"]
        leading_dims = point.shape[:-2]
        point = point.reshape(-1, *point.shape[-2:])
        ee_mask = x.get("ee_mask", None)
        if ee_mask is not None:
            ee_mask = ee_mask.reshape(-1, *ee_mask.shape[-1:])
        pad_mask = x.get("pad_mask", None)
        if pad_mask is not None:
            pad_mask = pad_mask.reshape(-1, *pad_mask.shape[-1:])
            pad_mask = pad_mask.to(dtype=torch.bool)
            pad_mask = pad_mask.unsqueeze(1)  # (..., 1, points)
            pad_mask = pad_mask.repeat(
                1, self.num_queries, 1
            )  # (..., num_queries, points)
        if self.subtract_mean:
            mean = torch.mean(point, dim=-2, keepdim=True)  # (..., 1, coordinates)
            mean = torch.broadcast_to(mean, point.shape)  # (..., points, coordinates)
            point = point - mean
            point = torch.cat([point, mean], dim=-1)  # (..., points, 2 * coordinates)
        if self.add_ee_embd:
            ee_mask = ee_mask.to(dtype=torch.long)  # (..., points)
            ee_embd = self.ee_embd_layer(ee_mask)  # (..., points, ee_embd_dim)
            point = torch.concat(
                [point, ee_embd], dim=-1
            )  # (..., points, coordinates + ee_embd_dim)
        point = self.linear(point)  # (..., points, hidden_dim)
        output = self.set_xf(point, mask=pad_mask)  # (..., self.output_dim)
        # recover leading dimensions
        output = output.reshape(*leading_dims, *output.shape[-1:])
        return output
