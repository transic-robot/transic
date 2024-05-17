from typing import Callable, Union, List, Tuple

import torch
import torch.nn as nn

from transic.utils.misc_utils import match_patterns

FilterType = Union[
    Callable[[str, torch.Tensor], bool], List[str], Tuple[str], str, None
]


def default_optimizer_groups(
    model: nn.Module,
    weight_decay: float,
    lr_scale: float = 1.0,
    no_decay_filter: FilterType = None,
    exclude_filter: FilterType = None,
):
    """
    lr_scale is only effective when using with enlight.learn.lr_schedule.LambdaLRWithScale

    Returns:
        [{'lr_scale': 1.0, 'weight_decay': weight_decay, 'params': decay_group},
         {'lr_scale': 1.0, 'weight_decay': 0.0, 'params': no_decay_group}],
        list of all param_ids processed
    """
    no_decay_filter = _transform_filter(no_decay_filter)
    exclude_filter = _transform_filter(exclude_filter)
    decay_group = []
    no_decay_group = []
    all_params_id = []
    for n, p in model.named_parameters():
        all_params_id.append(id(p))
        if not p.requires_grad or exclude_filter(n, p):
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or no_decay_filter(n, p):
            no_decay_group.append(p)
        else:
            decay_group.append(p)
    return [
        {"weight_decay": weight_decay, "params": decay_group, "lr_scale": lr_scale},
        {"weight_decay": 0.0, "params": no_decay_group, "lr_scale": lr_scale},
    ], all_params_id


def _transform_filter(filter: FilterType):
    """
    Filter can be:
        - None: always returns False
        - function(name, p) -> True to activate, False to deactivate
        - list of strings to match, can have wildcard
    """
    if filter is None:
        return lambda name, p: False
    elif callable(filter):
        return filter
    elif isinstance(filter, (str, list, tuple)):
        if isinstance(filter, str):
            filter = [filter]
        return lambda name, p: match_patterns(name, include=filter)
    else:
        raise ValueError(f"Invalid filter: {filter}")
