from omegaconf import DictConfig, OmegaConf


def _is_cuda_solver(x, y):
    if isinstance(y, int):
        return y >= 0
    if isinstance(y, str):
        if "cuda" in y.lower():
            return True
        else:
            return x.lower() in y.lower()


OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", _is_cuda_solver)
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg == "" else arg
)
OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
OmegaConf.register_new_resolver("floor_divide", lambda x, y: x // y)
OmegaConf.register_new_resolver(
    "find_rl_train_config",
    lambda x: x + "PPO" if x[-3:] != "PCD" else x[:-3] + "PPO",
)
