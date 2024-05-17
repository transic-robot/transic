from math import ceil
from copy import deepcopy

import numpy as np

from transic.utils.array import (
    any_slice,
    nested_np_split,
    get_batch_size,
    any_concat,
    any_stack,
    any_ones_like,
    any_to_torch_tensor,
)


def collate_fn(
    sample_list,
    with_matched_scene: bool,
    ctx_len: int = 5,
):
    """
    sample_list: List of
        Tuple[main_data: Dict, Tuple[Tuple[real_pcds, real_pcd_ee_masks], Tuple[sim_pcds, sim_pcd_ee_masks]]] if with_matched_scene is True
        main_data: Dict, if with_matched_scene is False
    """
    if with_matched_scene:
        main_data = [sample[0] for sample in sample_list]  # List[Dict]
        matched_scenes = [
            sample[1] for sample in sample_list
        ]  # List[Tuple[Tuple[real_pcds, real_pcd_ee_masks], Tuple[sim_pcds, sim_pcd_ee_masks]]]
    else:
        main_data = sample_list
        matched_scenes = None

    L_max = max(get_batch_size(sample) for sample in main_data)
    N_chunks = ceil(L_max / ctx_len)
    L_pad_max = N_chunks * ctx_len

    sample_structure = deepcopy(any_slice(main_data[0], np.s_[0:1]))
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
            for sample in main_data
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
            for sample in main_data
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
    if matched_scenes is not None:
        matched_scenes = any_stack(matched_scenes, dim=0)
        matched_scenes_tensor = (
            (
                any_to_torch_tensor(matched_scenes[0][0]),
                any_to_torch_tensor(matched_scenes[0][1]),
            ),
            (
                any_to_torch_tensor(matched_scenes[1][0]),
                any_to_torch_tensor(matched_scenes[1][1]),
            ),
        )
        return processed_main_data, matched_scenes_tensor
    return processed_main_data
