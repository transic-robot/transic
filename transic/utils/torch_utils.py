import tree
import torch
import torch.nn as nn

from transic.utils.tree_utils import tree_value_at_path


def load_state_dict(objects, states, strip_prefix=None, strict=False):
    """
    Args:
        strict: objects and states must match exactly
        strip_prefix: only match the keys that have the prefix, and strip it
    """

    def _load(paths, obj):
        if not _implements_method(obj, "load_state_dict"):
            raise ValueError(
                f"Object {type(obj)} does not support load_state_dict() method"
            )
        try:
            state = tree_value_at_path(states, paths)
        except ValueError:  # paths do not exist in `states` structure
            if strict:
                raise
            else:
                return
        if strip_prefix:
            assert isinstance(strip_prefix, str)
            state = {
                k[len(strip_prefix) :]: v
                for k, v in state.items()
                if k.startswith(strip_prefix)
            }
        if isinstance(obj, nn.Module):
            return obj.load_state_dict(state, strict=strict)
        else:
            return obj.load_state_dict(state)

    return tree.map_structure_with_path(_load, objects)


def _implements_method(object, method: str):
    """
    Returns:
        True if object implements a method
    """
    return hasattr(object, method) and callable(getattr(object, method))


def set_requires_grad(model, requires_grad):
    if torch.is_tensor(model):
        model.requires_grad = requires_grad
    else:
        for param in model.parameters():
            param.requires_grad = requires_grad


def freeze_params(model):
    set_requires_grad(model, False)
    if not torch.is_tensor(model):
        model.eval()
