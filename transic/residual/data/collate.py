from math import ceil
from copy import deepcopy

import numpy as np

from transic.utils.array import (
    get_batch_size,
    any_slice,
    any_stack,
    any_concat,
    any_ones_like,
    any_to_torch_tensor,
    nested_np_split,
)


def collate_fn(
    sample_list,
    ctx_len: int = 5,
):
    """
    sample_list: List of Dict[str, np.ndarray]
    """
    L_max = max(get_batch_size(sample) for sample in sample_list)
    N_chunks = ceil(L_max / ctx_len)
    L_pad_max = N_chunks * ctx_len

    sample_structure = deepcopy(any_slice(sample_list[0], np.s_[0:1]))
    # pad to max length in this batch
    processed_main_data = any_stack(
        [
            any_concat(
                [
                    sample,
                ]
                + [any_ones_like(sample_structure)]
                * (L_pad_max - get_batch_size(sample)),
                dim=0,
            )
            for sample in sample_list
        ],
        dim=0,
    )  # dict of (B, L_pad_max, ...)
    # construct mask
    mask = any_stack(
        [
            any_concat(
                [
                    np.ones((get_batch_size(sample),), dtype=bool),
                    np.zeros((L_pad_max - get_batch_size(sample),), dtype=bool),
                ]
            )
            for sample in sample_list
        ],
        dim=0,
    )  # (B, L_pad_max)

    # split into chunks
    processed_main_data = {
        k: any_stack(v, dim=0)
        for k, v in nested_np_split(processed_main_data, N_chunks, axis=1).items()
    }  # dict of (N_chunks, B, ctx_len, ...)
    mask = any_stack(np.split(mask, N_chunks, axis=1), dim=0)  # (N_chunks, B, ctx_len)
    processed_main_data["pad_mask"] = mask

    # convert to tensor
    processed_main_data = {
        k: any_to_torch_tensor(v) for k, v in processed_main_data.items()
    }
    return processed_main_data
