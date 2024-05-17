from __future__ import annotations

import torch
import torch.nn as nn

from transic.nn.mlp import build_mlp
from transic.learn.optimizer_group import default_optimizer_groups


class SimpleFeatureFusion(nn.Module):
    def __init__(
        self,
        extractors: dict[str, nn.Module],
        hidden_depth: int,
        hidden_dim: int,
        output_dim: int,
        activation,
        add_input_activation: bool,
        add_output_activation: bool,
    ):
        super().__init__()
        self._extractors = nn.ModuleDict(extractors)
        extractors_output_dim = sum(e.output_dim for e in extractors.values())
        self.output_dim = output_dim
        self._head = build_mlp(
            input_dim=extractors_output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            weight_init="orthogonal",
            bias_init="zeros",
            norm_type=None,
            add_input_activation=add_input_activation,
            add_input_norm=False,
            add_output_activation=add_output_activation,
            add_output_norm=False,
        )

        self._obs_groups = None
        self._obs_key_checked = False

    def _check_obs_key_match(self, obs: dict, strict: bool = False):
        if strict:
            assert set(self._extractors.keys()) == set(obs.keys())
        elif set(self._extractors.keys()) != set(obs.keys()):
            print(
                f"[warning] obs key mismatch: {set(self._extractors.keys())} != {set(obs.keys())}"
            )

    def forward(self, x):
        x = self._group_obs(x)
        if not self._obs_key_checked:
            self._check_obs_key_match(x, strict=False)
            self._obs_key_checked = True
        x = {k: v.forward(x[k]) for k, v in self._extractors.items()}
        x = torch.cat([x[k] for k in sorted(x.keys())], dim=-1)
        x = self._head(x)
        return x

    def _group_obs(self, obs):
        obs_keys = obs.keys()
        if self._obs_groups is None:
            # group by /
            obs_groups = {k.split("/")[0] for k in obs_keys}
            self._obs_groups = sorted(list(obs_groups))
        obs_rtn = {}
        for g in self._obs_groups:
            is_subgroup = any(k.startswith(f"{g}/") for k in obs_keys)
            if is_subgroup:
                obs_rtn[g] = {
                    k.split("/", 1)[1]: v
                    for k, v in obs.items()
                    if k.startswith(f"{g}/")
                }
            else:
                obs_rtn[g] = obs[g]
        return obs_rtn

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        extractors_pgs, extractor_pids = [], []
        for extractor in self._extractors.values():
            pg, pid = extractor.get_optimizer_groups(
                weight_decay=weight_decay,
                lr_layer_decay=lr_layer_decay,
                lr_scale=lr_scale,
            )
            extractors_pgs.extend(pg)
            extractor_pids.extend(pid)
        head_pg, head_pid = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            exclude_filter=lambda name, p: id(p) in extractor_pids,
        )
        return extractors_pgs + head_pg, extractor_pids + head_pid
