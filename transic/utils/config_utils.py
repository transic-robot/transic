import collections
from omegaconf import OmegaConf


def is_sequence(obj):
    """
    Returns:
      True if the sequence is a collections.Sequence and not a string.
    """
    return isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str)


def is_mapping(obj):
    """
    Returns:
      True if the sequence is a collections.Mapping
    """
    return isinstance(obj, collections.abc.Mapping)


def omegaconf_to_dict(cfg, resolve: bool = True, enum_to_str: bool = False):
    """
    Convert arbitrary nested omegaconf objects to primitive containers

    WARNING: cannot use tree lib because it gets confused on DictConfig and ListConfig
    """
    kw = dict(resolve=resolve, enum_to_str=enum_to_str)
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, **kw)
    elif is_sequence(cfg):
        return type(cfg)(omegaconf_to_dict(c, **kw) for c in cfg)
    elif is_mapping(cfg):
        return {k: omegaconf_to_dict(c, **kw) for k, c in cfg.items()}
    else:
        return cfg
