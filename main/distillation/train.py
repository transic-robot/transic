import isaacgym
import hydra
import transic
from transic.utils.utils import set_seed
from transic.learn.lightning import LightingTrainer
from transic.utils.config_utils import omegaconf_to_dict


@hydra.main(config_name="distillation_config", config_path="../cfg", version_base="1.1")
def main(cfg):
    cfg.seed = set_seed(cfg.seed)
    trainer_ = LightingTrainer(cfg)
    trainer_.trainer.loggers[-1].log_hyperparams(omegaconf_to_dict(cfg))
    trainer_.fit()


if __name__ == "__main__":
    main()
