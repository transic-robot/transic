from __future__ import annotations

import torch.nn as nn
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from gym import spaces


class RunningMeanStdObs(nn.Module):
    def __init__(
        self,
        insize,
        epsilon=1e-05,
        per_channel=False,
        norm_only=False,
        exclude_keys: list | None = None,
    ):
        assert isinstance(insize, spaces.Dict)
        exclude_keys = exclude_keys or []
        super(RunningMeanStdObs, self).__init__()
        self.running_mean_std = nn.ModuleDict(
            {
                k: RunningMeanStd(v.shape, epsilon, per_channel, norm_only)
                for k, v in insize.items()
                if k not in exclude_keys
            }
        )
        self._exclude_keys = exclude_keys

    def forward(self, input, denorm=False):
        res = {
            k: self.running_mean_std[k](v, denorm) if k not in self._exclude_keys else v
            for k, v in input.items()
        }
        return res
