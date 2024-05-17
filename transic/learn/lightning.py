from typing import List
import os
import time
from copy import deepcopy
import pprint

import sys
from omegaconf import DictConfig, OmegaConf, ListConfig
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    ProgressBar,
    TQDMProgressBar,
)
from pytorch_lightning.utilities import rank_zero_only
from hydra.utils import instantiate


class LightingTrainer:
    def __init__(self, cfg: DictConfig):
        cfg = deepcopy(cfg)
        OmegaConf.set_struct(cfg, False)
        self.cfg = cfg
        self.run_command_args = sys.argv[1:]
        run_name = self.generate_run_name(cfg)
        self.run_dir = os.path.join(cfg.exp_root_dir, run_name)
        rank_zero_print("Run name:", run_name, "\nExp dir:", self.run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "tb"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "ckpt"), exist_ok=True)
        OmegaConf.save(cfg, os.path.join(self.run_dir, "conf.yaml"))
        self.cfg = cfg
        self.run_name = run_name
        self.ckpt_cfg = cfg.trainer.pop("checkpoint")
        self.data_module = self.create_data_module(cfg)
        self._monkey_patch_add_info(self.data_module)
        self.trainer = self.create_trainer(cfg)
        self.module = self.create_module(cfg)
        self.module.data_module = self.data_module
        self._monkey_patch_add_info(self.module)

    def create_module(self, cfg):
        return instantiate(cfg.module, _recursive_=False)

    def create_data_module(self, cfg):
        return instantiate(cfg.data_module)

    def generate_run_name(self, cfg):
        return cfg.run_name + "_" + time.strftime("%Y%m%d-%H%M%S")

    def _monkey_patch_add_info(self, obj):
        """
        Add useful info to module and data_module so they can access directly
        """
        # our own info
        obj.run_config = self.cfg
        obj.run_name = self.run_name
        obj.run_command_args = self.run_command_args
        # add properties from trainer
        for attr in [
            "global_rank",
            "local_rank",
            "world_size",
            "num_nodes",
            "num_processes",
            "node_rank",
            "num_gpus",
            "data_parallel_device_ids",
        ]:
            if hasattr(obj, attr):
                continue
            setattr(
                obj.__class__,
                attr,
                # force capture 'attr'
                property(lambda self, attr=attr: getattr(self.trainer, attr)),
            )

    def create_loggers(self, cfg) -> List[pl.loggers.Logger]:
        loggers = [
            pl_loggers.TensorBoardLogger(self.run_dir, name="tb", version=""),
            pl_loggers.CSVLogger(self.run_dir, name="logs", version=""),
        ]
        if cfg.use_wandb:
            loggers.append(
                pl_loggers.WandbLogger(
                    name=cfg.wandb_run_name, project=cfg.wandb_project, id=self.run_name
                )
            )
        return loggers

    def create_callbacks(self) -> List[Callback]:
        ModelCheckpoint.FILE_EXTENSION = ".pth"
        callbacks = []
        if isinstance(self.ckpt_cfg, DictConfig):
            ckpt = ModelCheckpoint(
                dirpath=os.path.join(self.run_dir, "ckpt"), **self.ckpt_cfg
            )
            callbacks.append(ckpt)
        else:
            assert isinstance(self.ckpt_cfg, ListConfig)
            for _cfg in self.ckpt_cfg:
                ckpt = ModelCheckpoint(
                    dirpath=os.path.join(self.run_dir, "ckpt"), **_cfg
                )
                callbacks.append(ckpt)

        if not any(isinstance(c, ProgressBar) for c in callbacks):
            callbacks.append(TQDMProgressBar())
        rank_zero_print(
            "Lightning callbacks:", [c.__class__.__name__ for c in callbacks]
        )
        return callbacks

    def create_trainer(self, cfg) -> pl.Trainer:
        assert "trainer" in cfg
        C = cfg.trainer
        return instantiate(
            C, logger=self.create_loggers(cfg), callbacks=self.create_callbacks()
        )

    @property
    def tb_logger(self):
        return self.logger[0].experiment

    def fit(self):
        return self.trainer.fit(
            self.module,
            datamodule=self.data_module,
            ckpt_path=None,
        )


def pprint_(*objs, **kwargs):
    """
    Use pprint to format the objects
    """
    print(
        *[
            pprint.pformat(obj, indent=2) if not isinstance(obj, str) else obj
            for obj in objs
        ],
        **kwargs,
    )


@rank_zero_only
def rank_zero_print(*msg, **kwargs):
    pprint_(*msg, **kwargs)
