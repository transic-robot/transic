from typing import Optional, Tuple
from functools import partial

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from transic.distillation.data.dummy import DummyDataset
from transic.distillation.data.collate import collate_fn as _collate_fn
from transic.distillation.data.dataset import (
    DistillationDataset,
    DistillationSeqDataset,
)


class DistillationDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        data_path: str,
        matched_scene_data_path: Optional[str] = None,
        ctx_len: int = -1,  # -1 means not using the SeqDataset at all
        skip_first_n_steps: int,
        sampled_pcd_points: int,
        refresh_pcd_sampling_idxs_interval: float,
        real_pcd_x_limits: Tuple[float, float],
        real_pcd_y_limits: Tuple[float, float],
        real_pcd_z_min: float,
        batch_size: int,
        dataloader_num_workers: int,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self._data_path = data_path
        self._matched_scene_data_path = matched_scene_data_path
        self._skip_first_n_steps = skip_first_n_steps
        self._sampled_pcd_points = sampled_pcd_points
        self._refresh_pcd_sampling_idxs_interval = refresh_pcd_sampling_idxs_interval
        self._real_pcd_x_limits = real_pcd_x_limits
        self._real_pcd_y_limits = real_pcd_y_limits
        self._real_pcd_z_min = real_pcd_z_min

        self._batch_size = batch_size
        self._dataloader_num_workers = dataloader_num_workers
        self._seed = seed

        self._ds_cls = DistillationSeqDataset if ctx_len != -1 else DistillationDataset
        self._collate_fn = (
            partial(
                _collate_fn,
                with_matched_scene=matched_scene_data_path is not None,
                ctx_len=ctx_len,
            )
            if ctx_len != -1
            else None
        )
        self._train_dataset = None

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self._train_dataset = self._ds_cls(
                fpath=self._data_path,
                matched_scene_fpath=self._matched_scene_data_path,
                sampled_pcd_points=self._sampled_pcd_points,
                skip_first_n_steps=self._skip_first_n_steps,
                refresh_pcd_sampling_idxs_interval=self._refresh_pcd_sampling_idxs_interval,
                real_pcd_x_limits=self._real_pcd_x_limits,
                real_pcd_y_limits=self._real_pcd_y_limits,
                real_pcd_z_min=self._real_pcd_z_min,
                seed=self._seed,
            )

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=min(self._batch_size, self._dataloader_num_workers),
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DummyDataset(batch_size=1).get_dataloader()
