import numpy as np
from torch.utils.data import Dataset, DataLoader


class DummyDataset(Dataset):
    """
    For test_step(), simply returns None N times.
    test_step() can have arbitrary logic
    """

    def __init__(self, batch_size, epoch_len=1):
        """
        Still set batch_size because pytorch_lightning tracks it
        """
        self.n = epoch_len
        self._batch_size = batch_size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return np.zeros((self._batch_size,), dtype=bool)

    def get_dataloader(self) -> DataLoader:
        """
        Our dataset directly returns batched tensors instead of single samples,
        so for DataLoader we don't need a real collate_fn and set batch_size=1
        """
        return DataLoader(
            self,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            collate_fn=_singleton_collate_fn,
        )


def _singleton_collate_fn(tensor_list):
    """
    Our dataset directly returns batched tensors instead of single samples,
    so for DataLoader we don't need a real collate_fn.
    """
    assert len(tensor_list) == 1, "INTERNAL: collate_fn only allows a single item"
    return tensor_list[0]
