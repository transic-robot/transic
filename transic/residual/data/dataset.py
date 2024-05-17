from typing import Optional, Literal

import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from transic.utils.array import any_slice, any_concat


class ResidualDataset(Dataset):
    def __init__(
        self,
        *,
        data_dir: str,
        variable_len_pcd_handle_strategy: Literal["pad", "truncate"],
        include_grasp_action: bool,
        gripper_close_width: float,
        gripper_open_width: float = 0.08,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self._random_state = np.random.RandomState(seed)

        self._include_grasp_action = include_grasp_action
        self._gripper_close_width = gripper_close_width
        self._gripper_open_width = gripper_open_width

        assert os.path.exists(data_dir)
        # dataset files are .hdf5 files
        ds_fs = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".hdf5")
        ]
        ds_fs = self._random_state.permutation(ds_fs)

        self._variable_len_pcd_handle_strategy = variable_len_pcd_handle_strategy
        self._pcd_N_min = None
        self._pcd_N_max = None
        # find the minimum/maximum length of pointcloud
        for fpath in ds_fs:
            f = h5py.File(fpath, "r", swmr=True, libver="latest")
            policy_obs_pcd_padding_mask = f["policy_obs/pcd/padding_mask"][()].astype(
                bool
            )
            pcd_N_min = min(np.sum(policy_obs_pcd_padding_mask, axis=1))
            pcd_N_max = max(np.sum(policy_obs_pcd_padding_mask, axis=1))
            if self._pcd_N_min is None:
                self._pcd_N_min = pcd_N_min
            else:
                self._pcd_N_min = min(self._pcd_N_min, pcd_N_min)
            if self._pcd_N_max is None:
                self._pcd_N_max = pcd_N_max
            else:
                self._pcd_N_max = max(self._pcd_N_max, pcd_N_max)

        self._total_steps = 0
        self._total_intervention_steps = 0
        self._P_intervention = None
        all_data = [self._load_single_file(f) for f in ds_fs]
        self.all_data = [x for x in all_data if x is not None]
        self._len = sum([len(d["policy_obs_proprioception"]) for d in self.all_data])

        self.demo_pointer = 0
        self.step_pointer = 0

    @property
    def P_intervention(self):
        if self._P_intervention is None:
            self._P_intervention = self._total_intervention_steps / self._total_steps
        return self._P_intervention

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        data = any_slice(self.all_data[self.demo_pointer], self.step_pointer)
        self.step_pointer += 1
        if (
            self.step_pointer
            >= self.all_data[self.demo_pointer]["policy_obs_proprioception"].shape[0]
        ):
            self.demo_pointer += 1
            self.demo_pointer %= len(self.all_data)
            self.step_pointer = 0
        return data

    def _load_single_file(self, fpath: str):
        f = h5py.File(fpath, "r", swmr=True, libver="latest")
        policy_obs_proprioception = f["policy_obs/proprioception"][()].astype("float32")
        policy_obs_pcd_coordinate = f["policy_obs/pcd/coordinate"][()].astype("float32")
        policy_obs_pcd_ee_mask = f["policy_obs/pcd/ee_mask"][()].astype(bool)
        policy_obs_pcd_padding_mask = f["policy_obs/pcd/padding_mask"][()].astype(bool)
        policy_action = f["policy_action"][()].astype("float32")
        is_human_intervention = f["is_human_intervention"][()].astype(bool)
        pre_intervention_q = f["pre_intervention_q"][()].astype("float32")
        post_intervention_q = f["post_intervention_q"][()].astype("float32")
        if self._include_grasp_action:
            pre_intervention_gripper_q = f["pre_intervention_gripper_q"][()].astype(
                "float32"
            )
            post_intervention_gripper_q = f["post_intervention_gripper_q"][()].astype(
                "float32"
            )

        # only consider steps with human intervention
        policy_obs_proprioception = policy_obs_proprioception[is_human_intervention]
        policy_obs_pcd_coordinate = policy_obs_pcd_coordinate[is_human_intervention]
        policy_obs_pcd_ee_mask = policy_obs_pcd_ee_mask[is_human_intervention]
        policy_obs_pcd_padding_mask = policy_obs_pcd_padding_mask[is_human_intervention]
        policy_action = policy_action[is_human_intervention]
        pre_intervention_q = pre_intervention_q[is_human_intervention]
        post_intervention_q = post_intervention_q[is_human_intervention]
        if self._include_grasp_action:
            pre_intervention_gripper_q = pre_intervention_gripper_q[
                is_human_intervention
            ]
            post_intervention_gripper_q = post_intervention_gripper_q[
                is_human_intervention
            ]

        if pre_intervention_q.shape[0] == 0:
            return None

        # handle pointcloud with variable length
        if self._variable_len_pcd_handle_strategy == "truncate":
            policy_obs_pcd_coordinate_processed = []
            policy_obs_pcd_ee_mask_processed = []
            for coordinate, ee_mask, padding_mask in zip(
                policy_obs_pcd_coordinate,
                policy_obs_pcd_ee_mask,
                policy_obs_pcd_padding_mask,
            ):
                # because pcd is already randomly sampled, so just taking first pcd_N_min points shouldn't incur bias
                policy_obs_pcd_coordinate_processed.append(
                    coordinate[padding_mask][: self._pcd_N_min]
                )
                policy_obs_pcd_ee_mask_processed.append(
                    ee_mask[padding_mask][: self._pcd_N_min]
                )
            policy_obs_pcd_coordinate_processed = np.stack(
                policy_obs_pcd_coordinate_processed
            )
            policy_obs_pcd_ee_mask_processed = np.stack(
                policy_obs_pcd_ee_mask_processed
            )
        elif self._variable_len_pcd_handle_strategy == "pad":
            # already padded when saving
            policy_obs_pcd_coordinate_processed = policy_obs_pcd_coordinate
            policy_obs_pcd_ee_mask_processed = policy_obs_pcd_ee_mask
        else:
            raise ValueError
        residual_q = post_intervention_q - pre_intervention_q
        # construct change gripper action if needed
        if self._include_grasp_action:
            pre_intervention_gripper_q = rectify(
                pre_intervention_gripper_q,
                self._gripper_open_width,
                self._gripper_close_width,
            )
            post_intervention_gripper_q = rectify(
                post_intervention_gripper_q,
                self._gripper_open_width,
                self._gripper_close_width,
            )
            # gripper change action is 1 if post-pre changes, otherwise 0
            gripper_change_action = (
                post_intervention_gripper_q - pre_intervention_gripper_q
            ) != 0

        rtn = {
            "policy_obs_proprioception": policy_obs_proprioception,
            "policy_obs_pcd_coordinate": policy_obs_pcd_coordinate_processed,
            "policy_obs_pcd_ee_mask": policy_obs_pcd_ee_mask_processed,
            "policy_action": policy_action,
            "residual_q": residual_q,
        }
        if self._include_grasp_action:
            rtn["gripper_change_action"] = gripper_change_action[..., None]  # (N, 1)
        return rtn


class ResidualSeqDataset(ResidualDataset):
    def __getitem__(self, index):
        data = self.all_data[self.demo_pointer]
        self.demo_pointer += 1
        self.demo_pointer %= len(self.all_data)
        return data

    def __len__(self):
        return len(self.all_data)

    def _load_single_file(self, fpath: str):
        f = h5py.File(fpath, "r", swmr=True, libver="latest")
        policy_obs_proprioception = f["policy_obs/proprioception"][()].astype("float32")
        policy_obs_pcd_coordinate = f["policy_obs/pcd/coordinate"][()].astype("float32")
        policy_obs_pcd_ee_mask = f["policy_obs/pcd/ee_mask"][()].astype(bool)
        policy_obs_pcd_padding_mask = f["policy_obs/pcd/padding_mask"][()].astype(bool)
        policy_action = f["policy_action"][()].astype("float32")
        is_human_intervention = f["is_human_intervention"][()].astype(bool)
        pre_intervention_q = f["pre_intervention_q"][()].astype("float32")
        post_intervention_q = f["post_intervention_q"][()].astype("float32")
        if self._include_grasp_action:
            pre_intervention_gripper_q = f["pre_intervention_gripper_q"][()].astype(
                "float32"
            )
            post_intervention_gripper_q = f["post_intervention_gripper_q"][()].astype(
                "float32"
            )

        # skip traj without human intervention
        if not np.any(is_human_intervention):
            return None

        self._total_steps += is_human_intervention.shape[0]
        self._total_intervention_steps += np.sum(is_human_intervention)

        # handle pointcloud with variable length
        if self._variable_len_pcd_handle_strategy == "truncate":
            policy_obs_pcd_coordinate_processed = []
            policy_obs_pcd_ee_mask_processed = []
            policy_obs_pcd_pad_mask_processed = []
            for coordinate, ee_mask, padding_mask in zip(
                policy_obs_pcd_coordinate,
                policy_obs_pcd_ee_mask,
                policy_obs_pcd_padding_mask,
            ):
                # because pcd is already randomly sampled, so just taking first pcd_N_min points shouldn't incur bias
                policy_obs_pcd_coordinate_processed.append(
                    coordinate[padding_mask][: self._pcd_N_min]
                )
                policy_obs_pcd_ee_mask_processed.append(
                    ee_mask[padding_mask][: self._pcd_N_min]
                )
                policy_obs_pcd_pad_mask_processed.append(
                    padding_mask[: self._pcd_N_min]
                )
            policy_obs_pcd_coordinate_processed = np.stack(
                policy_obs_pcd_coordinate_processed
            )
            policy_obs_pcd_ee_mask_processed = np.stack(
                policy_obs_pcd_ee_mask_processed
            )
            policy_obs_pcd_pad_mask_processed = np.stack(
                policy_obs_pcd_pad_mask_processed
            )
        elif self._variable_len_pcd_handle_strategy == "pad":
            # pad to self._pcd_N_max
            T, N_points = policy_obs_pcd_coordinate.shape[:2]
            N_padding = self._pcd_N_max - N_points
            policy_obs_pcd_coordinate_processed = any_concat(
                [
                    policy_obs_pcd_coordinate,
                    np.zeros((T, N_padding, 3), dtype=np.float32),
                ],
                dim=1,
            )
            policy_obs_pcd_ee_mask_processed = any_concat(
                [
                    policy_obs_pcd_ee_mask,
                    np.zeros((T, N_padding), dtype=policy_obs_pcd_ee_mask.dtype),
                ],
                dim=1,
            )
            policy_obs_pcd_pad_mask_processed = any_concat(
                [
                    policy_obs_pcd_padding_mask,
                    np.zeros((T, N_padding), dtype=policy_obs_pcd_padding_mask.dtype),
                ],
                dim=1,
            )
        else:
            raise ValueError
        residual_q = post_intervention_q - pre_intervention_q
        # construct change gripper action if needed
        if self._include_grasp_action:
            pre_intervention_gripper_q = rectify(
                pre_intervention_gripper_q,
                self._gripper_open_width,
                self._gripper_close_width,
            )
            post_intervention_gripper_q = rectify(
                post_intervention_gripper_q,
                self._gripper_open_width,
                self._gripper_close_width,
            )
            # gripper change action is 1 if post-pre changes, otherwise 0
            gripper_change_action = (
                post_intervention_gripper_q - pre_intervention_gripper_q
            ) != 0

        rtn = {
            "policy_obs_proprioception": policy_obs_proprioception,
            "policy_obs_pcd_coordinate": policy_obs_pcd_coordinate_processed,
            "policy_obs_pcd_ee_mask": policy_obs_pcd_ee_mask_processed,
            "policy_obs_pcd_pad_mask": policy_obs_pcd_pad_mask_processed,
            "policy_action": policy_action,
            "residual_q": residual_q,
            "is_human_intervention": is_human_intervention,
        }
        if self._include_grasp_action:
            rtn["gripper_change_action"] = gripper_change_action[..., None]  # (N, 1)
        return rtn


def rectify(x, a, b):
    # Calculate the absolute difference between each element in x and a, and x and b
    diff_a = np.abs(x - a)
    diff_b = np.abs(x - b)

    # Replace elements in x with a or b depending on which is closer
    x[diff_a <= diff_b] = a
    x[diff_a > diff_b] = b

    return x
