from __future__ import annotations
from typing import Literal, Callable, Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transic.nn.mlp import build_mlp


class Categorical(torch.distributions.Categorical):
    """
    Mostly interface changes, add mode() function, no real difference from Categorical
    """

    def mode(self):
        return self.logits.argmax(dim=-1)

    def imitation_loss(self, actions, reduction="mean"):
        """
        actions: groundtruth actions from expert
        """
        assert actions.dtype == torch.long
        if self.logits.ndim == 3:
            assert actions.ndim == 2
            assert self.logits.shape[:2] == actions.shape
            return F.cross_entropy(
                self.logits.reshape(-1, self.logits.shape[-1]),
                actions.reshape(-1),
                reduction=reduction,
            )
        return F.cross_entropy(self.logits, actions, reduction=reduction)

    def imitation_accuracy(self, actions, mask=None, reduction="mean", scale_100=False):
        if self.logits.ndim == 3:
            assert actions.ndim == 2
            assert self.logits.shape[:2] == actions.shape
            if mask is not None:
                assert mask.ndim == 2
                assert self.logits.shape[:2] == mask.shape
            actions = actions.reshape(-1)
            if mask is not None:
                mask = mask.reshape(-1)
            return classify_accuracy(
                self.logits.reshape(-1, self.logits.shape[-1]),
                actions,
                mask=mask,
                reduction=reduction,
                scale_100=scale_100,
            )
        return classify_accuracy(
            self.logits, actions, mask=mask, reduction=reduction, scale_100=scale_100
        )

    def random_actions(self):
        """
        Generate a completely random action, NOT the same as sample(), more like
        action_space.sample()
        """
        return torch.randint(
            low=0,
            high=self.logits.size(-1),
            size=self.logits.size()[:-1],
            device=self.logits.device,
        )


class CategoricalHead(nn.Module):
    def forward(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=x)


class CategoricalNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        action_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        last_layer_gain: float | None = 0.01,
    ):
        """
        Use orthogonal initialization to initialize the MLP policy

        Args:
            last_layer_gain: orthogonal initialization gain for the last FC layer.
                you may want to set it to a small value (e.g. 0.01) to make the
                Categorical close to uniform random at the beginning.
                Set to None to use the default gain (dependent on the NN activation)
        """
        super().__init__()
        self.mlp = _build_mlp_distribution_net(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
            last_layer_gain=last_layer_gain,
        )
        self.head = CategoricalHead()

    def forward(self, x):
        return self.head(self.mlp(x))


class MixtureOfGaussian:
    def __init__(
        self,
        logits: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
        min_std: float = 0.0001,
        low_noise_eval: bool = True,
    ):
        """
        logits: (..., n_modes)
        means: (..., n_modes, dim)
        scales: (..., n_modes, dim)
        """
        assert logits.dim() + 1 == means.dim() == scales.dim()
        assert logits.shape[-1] == means.shape[-2] == scales.shape[-2]
        assert means.shape == scales.shape

        self._logits = logits
        self._means = torch.tanh(means)
        self._scales = scales
        self._min_std = min_std
        self._low_noise_eval = low_noise_eval

    def mode(self):
        # assume mode will only be called during eval
        if self._low_noise_eval:
            scales = torch.ones_like(self._means) * 1e-4
            component_distribution = torch.distributions.Normal(
                loc=self._means, scale=scales
            )
            component_distribution = torch.distributions.Independent(
                component_distribution, 1
            )
            dist = torch.distributions.MixtureSameFamily(
                mixture_distribution=torch.distributions.Categorical(
                    logits=self._logits
                ),
                component_distribution=component_distribution,
            )
            return dist.sample()
        else:
            # return the mean of the most probable component
            one_hot = F.one_hot(
                self._logits.argmax(dim=-1), self._logits.shape[-1]
            ).unsqueeze(-1)
            return (self._means * one_hot).sum(dim=-2)

    def imitation_loss(self, actions, reduction="mean"):
        """
        NLL loss
        actions: (..., dim)
        """

        batch_dims = self._logits.shape[:-1]
        logits = self._logits.reshape(-1, self._logits.shape[-1])
        means = self._means.reshape(-1, *self._means.shape[-2:])
        scales = self._scales.reshape(-1, *self._scales.shape[-2:])

        scales = F.softplus(scales) + self._min_std
        component_distribution = torch.distributions.Normal(loc=means, scale=scales)
        component_distribution = torch.distributions.Independent(
            component_distribution, 1
        )
        dist = torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(logits=logits),
            component_distribution=component_distribution,
        )
        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dist.batch_shape) == 1

        assert actions.shape[:-1] == batch_dims
        actions = actions.reshape(-1, actions.shape[-1])
        log_probs = dist.log_prob(actions)  # (...,), note that action dim is summed
        log_probs = log_probs.reshape(*batch_dims)
        if reduction == "mean":
            return -log_probs.mean()
        elif reduction == "sum":
            return -log_probs.sum()
        elif reduction == "none":
            return -log_probs

    def imitation_accuracy(self, actions, mask=None, reduction="mean"):
        """
        L1 distance between mode and actions

        actions: (..., dim)
        mask: (...,)
        """
        if mask is not None:
            assert mask.shape == actions.shape[:-1]

        scales = torch.ones_like(self._means) * 1e-4
        component_distribution = torch.distributions.Normal(
            loc=self._means, scale=scales
        )
        component_distribution = torch.distributions.Independent(
            component_distribution, 1
        )
        dist = torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(logits=self._logits),
            component_distribution=component_distribution,
        )
        mode = dist.mean  # (..., dim)
        loss = (actions - mode).abs().sum(-1)  # (...)
        # we want accuracy, higher is better, so we negate the loss
        loss = -loss
        if mask is not None:
            loss *= mask
        if reduction == "mean":
            if mask is not None:
                return loss.sum() / mask.sum()
            else:
                return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss


class GMMHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        n_modes: int = 5,
        action_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        mean_mlp_last_layer_gain: float | None = 0.01,
        low_noise_eval: bool = True,
    ):
        super().__init__()
        self._logits_mlp = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=n_modes,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
        )
        self._mean_mlp = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=n_modes * action_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
        )
        if mean_mlp_last_layer_gain is not None:
            assert mean_mlp_last_layer_gain > 0
            nn.init.orthogonal_(
                self._mean_mlp[-1].weight, gain=mean_mlp_last_layer_gain
            )
        self._scale_mlp = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=n_modes * action_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
        )
        self._n_modes = n_modes
        self._action_dim = action_dim

        self._low_noise_eval = low_noise_eval

    def forward(self, x: torch.Tensor):
        logits = self._logits_mlp(x)
        mean = self._mean_mlp(x)  # (..., n_modes * action_dim)
        scale = self._scale_mlp(x)  # (..., n_modes * action_dim)
        mean = mean.reshape(*mean.shape[:-1], self._n_modes, self._action_dim)
        scale = scale.reshape(*scale.shape[:-1], self._n_modes, self._action_dim)

        assert logits.shape[-1] == self._n_modes
        assert mean.shape[-2:] == (self._n_modes, self._action_dim)
        assert scale.shape[-2:] == (self._n_modes, self._action_dim)
        return MixtureOfGaussian(
            logits, mean, scale, low_noise_eval=self._low_noise_eval
        )

    @property
    def action_dim(self):
        return self._action_dim


def _build_mlp_distribution_net(
    input_dim: int,
    *,
    output_dim: int,
    hidden_dim: int,
    hidden_depth: int,
    activation: str | Callable = "relu",
    norm_type: Literal["batchnorm", "layernorm"] | None = None,
    last_layer_gain: float | None = 0.01,
):
    """
    Use orthogonal initialization to initialize the MLP policy

    Args:
        last_layer_gain: orthogonal initialization gain for the last FC layer.
            you may want to set it to a small value (e.g. 0.01) to have the
            Gaussian centered around 0.0 in the beginning.
            Set to None to use the default gain (dependent on the NN activation)
    """

    mlp = build_mlp(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        activation=activation,
        weight_init="orthogonal",
        bias_init="zeros",
        norm_type=norm_type,
    )
    if last_layer_gain:
        assert last_layer_gain > 0
        nn.init.orthogonal_(mlp[-1].weight, gain=last_layer_gain)
    return mlp


def classify_accuracy(
    output,
    target,
    topk: Union[int, List[int], Tuple[int]] = 1,
    mask=None,
    reduction="mean",
    scale_100=False,
):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    Accuracy is a float between 0.0 and 1.0

    Args:
        topk: if int, return a single acc. If tuple, return a tuple of accs
        mask: shape [batch_size,], binary mask of whether to include this sample or not
    """
    if isinstance(topk, int):
        topk = [topk]
        is_int = True
    else:
        is_int = False

    batch_size = target.size(0)
    assert output.size(0) == batch_size
    if mask is not None:
        assert mask.dim() == 1
        assert mask.size(0) == batch_size

    assert reduction in ["sum", "mean", "none"]
    if reduction != "mean":
        assert not scale_100, f"reduce={reduction} does not support scale_100=True"

    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        if mask is not None:
            correct = mask * correct

        mult = 100.0 if scale_100 else 1.0
        res = []
        for k in topk:
            correct_k = correct[:k].int().sum(dim=0)
            if reduction == "mean":
                if mask is not None:
                    # fmt: off
                    res.append(
                        float(correct_k.float().sum().mul_(mult / (mask.sum().item() + 1e-6)).item())
                    )
                    # fmt: on
                else:
                    res.append(
                        float(correct_k.float().sum().mul_(mult / batch_size).item())
                    )
            elif reduction == "sum":
                res.append(int(correct_k.sum().item()))
            elif reduction == "none":
                res.append(correct_k)
            else:
                raise NotImplementedError(f"Unknown reduce={reduction}")

    if is_int:
        assert len(res) == 1, "INTERNAL"
        return res[0]
    else:
        return res
