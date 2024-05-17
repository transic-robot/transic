import isaacgym
import hydra
import transic
from transic.utils.utils import set_seed
from transic.distillation.data.data_module import DummyDataset
from transic.utils.config_utils import omegaconf_to_dict

import cv2
import numpy as np

cv2.imshow("dummy", np.zeros((1, 1, 3), dtype=np.uint8))
cv2.waitKey(1)


@hydra.main(config_name="distillation_config", config_path="../cfg", version_base="1.1")
def main(cfg):
    cfg.seed = set_seed(cfg.seed)

    if cfg.test.ckpt_path is None:
        print(
            "[WARNING] No ckpt_path is provided, will test with random weights. Press enter to continue."
        )
        input()

    from transic.learn.lightning import LightingTrainer

    trainer_ = LightingTrainer(cfg)
    trainer_.trainer.loggers[-1].log_hyperparams(omegaconf_to_dict(cfg))
    trainer_.trainer.test(
        model=trainer_.module,
        dataloaders=DummyDataset(batch_size=1, epoch_len=1).get_dataloader(),
        ckpt_path=cfg.test.ckpt_path,
    )


if __name__ == "__main__":
    main()
