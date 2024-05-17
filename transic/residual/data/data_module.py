from typing import Optional, Literal, List
from functools import partial

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from transic.residual.data.dataset import ResidualDataset, ResidualSeqDataset
from transic.residual.data.collate import collate_fn as _collate_fn


class ResidualDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        data_dir: str,
        variable_len_pcd_handle_strategy: Literal["pad", "truncate"],
        include_grasp_action: bool,
        gripper_close_width: float,
        gripper_open_width: float = 0.08,
        ctx_len: int = -1,  # -1 means not using the SeqDataset at all
        seed: Optional[int] = None,
        batch_size: int,
        val_batch_size: Optional[int],
        train_portion: float = 0.9,
        dataloader_num_workers: int,
    ):
        super().__init__()

        self._data_dir = data_dir
        self._variable_len_pcd_handle_strategy = variable_len_pcd_handle_strategy
        self._include_grasp_action = include_grasp_action
        self._gripper_close_width = gripper_close_width
        self._gripper_open_width = gripper_open_width
        self._seed = seed
        self._bs = batch_size
        self._vbs = val_batch_size or batch_size
        self._train_portion = train_portion
        self._dataloader_num_workers = dataloader_num_workers

        self._ds_cls = ResidualSeqDataset if ctx_len != -1 else ResidualDataset
        self._collate_fn = (
            partial(_collate_fn, ctx_len=ctx_len) if ctx_len != -1 else None
        )

        self._train_dataset, self._val_dataset = None, None
        self._P_intervention = None

    @property
    def P_intervention(self):
        assert self._P_intervention is not None, "Call setup() first"
        return self._P_intervention

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            ds = self._ds_cls(
                data_dir=self._data_dir,
                variable_len_pcd_handle_strategy=self._variable_len_pcd_handle_strategy,
                include_grasp_action=self._include_grasp_action,
                gripper_close_width=self._gripper_close_width,
                gripper_open_width=self._gripper_open_width,
                seed=self._seed,
            )
            self._P_intervention = ds.P_intervention
            self._train_dataset, self._val_dataset = _sequential_split_dataset(
                ds, split_portions=[self._train_portion, 1 - self._train_portion]
            )

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self._bs,
            num_workers=min(self._bs, self._dataloader_num_workers),
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self._vbs,
            num_workers=min(self._vbs, self._dataloader_num_workers),
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self._collate_fn,
        )


def _accumulate(iterable, fn=lambda x, y: x + y):
    """
    Return running totals
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    """
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


def _sequential_split_dataset(
    dataset: torch.utils.data.Dataset, split_portions: List[float]
):
    """
    Split a dataset into multiple datasets, each with a different portion of the
    original dataset. Uses torch.utils.data.Subset.
    """
    assert len(split_portions) > 0, "split_portions must be a non-empty list"
    assert all(0.0 <= p <= 1.0 for p in split_portions), f"{split_portions=}"
    assert abs(sum(split_portions) - 1.0) < 1e-6, f"{sum(split_portions)=} != 1.0"
    L = len(dataset)
    assert L > 0, "dataset must be non-empty"
    # split the list with proportions
    lengths = [int(p * L) for p in split_portions]
    # make sure the last split fills the full dataset
    lengths[-1] += L - sum(lengths)
    indices = list(range(L))

    return [
        torch.utils.data.Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]
