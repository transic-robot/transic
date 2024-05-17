import torch.nn as nn


class Identity(nn.Module):
    def __init__(
        self,
        input_dim: int,
    ):
        super().__init__()
        self._output_dim = input_dim

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x):
        return x

    def get_optimizer_groups(self, *args, **kwargs):
        return [], []
