from typing import Optional, Tuple

import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from transic.utils.array import any_slice


class DistillationDataset(Dataset):
    def __init__(
        self,
        *,
        fpath: str,
        matched_scene_fpath: Optional[str] = None,
        skip_first_n_steps: int = 0,
        sampled_pcd_points: int,
        refresh_pcd_sampling_idxs_interval: float = 0.25,
        real_pcd_x_limits: Tuple[float, float],
        real_pcd_y_limits: Tuple[float, float],
        real_pcd_z_min: float,
        seed: Optional[int] = None,
    ):
        super().__init__()

        assert os.path.exists(fpath)

        self._fpath = fpath
        self._hdf5_file = None
        self._skip_first_n_steps = skip_first_n_steps
        if matched_scene_fpath is not None:
            assert os.path.exists(matched_scene_fpath)
            self._matched_scene_fpath = matched_scene_fpath
            self.sampled_pcd_points = sampled_pcd_points
            self.real_pcd_x_limits = real_pcd_x_limits
            self.real_pcd_y_limits = real_pcd_y_limits
            self.real_pcd_z_min = real_pcd_z_min
        else:
            self._matched_scene_fpath = None
            self.sampled_pcd_points = None
            self.real_pcd_x_limits = None
            self.real_pcd_y_limits = None
            self.real_pcd_z_min = None
        self._matched_scene_hdf5_file = None

        self._random_state = np.random.RandomState(seed)
        # construct demo ids
        demo_ids = [
            f"rollouts/successful/{k}"
            for k in list(self.hdf5_file["rollouts"]["successful"].keys())
        ]
        # random permutate demo order
        self._demo_ids = self._random_state.permutation(demo_ids)
        self._demos_len = {
            k: len(self.hdf5_file[k]["q"]) - 1 - skip_first_n_steps
            for k in self._demo_ids
        }
        self._len = None

        if matched_scene_fpath is not None:
            self._refresh_pcd_sampling_idxs_interval = int(
                refresh_pcd_sampling_idxs_interval * len(self)
            )
            self._refresh_pcd_sampling_curr_steps = 0
            self._sim_pcd_sampling_idxs, self._real_pcd_sampling_idxs = {}, {}
        else:
            self._refresh_pcd_sampling_idxs_interval = None
            self._refresh_pcd_sampling_curr_steps = None
            self._sim_pcd_sampling_idxs, self._real_pcd_sampling_idxs = None, None

        self.demo_pointer = 0
        self.step_pointer = 0
        self.all_data = [self._load_single_demo(demo_id) for demo_id in self._demo_ids]
        self.single_demo = self.all_data[self.demo_pointer]

        if matched_scene_fpath is not None:
            self.matched_scene_data = {}
            for subdataset_name in self.matched_scene_hdf5_file.keys():
                for data_sample_name in self.matched_scene_hdf5_file[
                    subdataset_name
                ].keys():
                    k = f"{subdataset_name}/{data_sample_name}"
                    self.matched_scene_data[k] = {"sim": {}}
                    self.matched_scene_data[k]["sim"][
                        "pointcloud"
                    ] = self.matched_scene_hdf5_file[k]["sim"]["pointcloud"][()].astype(
                        "float32"
                    )
                    self.matched_scene_data[k]["sim"][
                        "ee_mask"
                    ] = self.matched_scene_hdf5_file[k]["sim"]["ee_mask"][()].astype(
                        "bool"
                    )
                    real_pcds = []
                    real_pcd_ee_masks = []
                    for example in self.matched_scene_hdf5_file[k]["real"].keys():
                        real_pcd_measured = self.matched_scene_hdf5_file[k]["real"][
                            example
                        ]["measured_pointcloud"][()].astype("float32")
                        # filter measured real pcd based on limits
                        x_mask = np.logical_and(
                            real_pcd_measured[:, 0] >= self.real_pcd_x_limits[0],
                            real_pcd_measured[:, 0] <= self.real_pcd_x_limits[1],
                        )
                        y_mask = np.logical_and(
                            real_pcd_measured[:, 1] >= self.real_pcd_y_limits[0],
                            real_pcd_measured[:, 1] <= self.real_pcd_y_limits[1],
                        )
                        z_mask = real_pcd_measured[:, 2] >= self.real_pcd_z_min
                        real_pcd_measured = real_pcd_measured[
                            np.logical_and(np.logical_and(x_mask, y_mask), z_mask)
                        ]
                        real_pcd_measured_ee_mask = np.zeros(
                            (real_pcd_measured.shape[0],), dtype=bool
                        )
                        real_pcd_finger = self.matched_scene_hdf5_file[k]["real"][
                            example
                        ]["fk_finger_pointcloud"][()].astype("float32")
                        real_pcd_finger_ee_mask = np.ones(
                            (real_pcd_finger.shape[0],), dtype=bool
                        )
                        real_pcd = np.concatenate(
                            [real_pcd_measured, real_pcd_finger], axis=0
                        )
                        real_pcd_ee_mask = np.concatenate(
                            [real_pcd_measured_ee_mask, real_pcd_finger_ee_mask], axis=0
                        )
                        real_pcds.append(real_pcd)
                        real_pcd_ee_masks.append(real_pcd_ee_mask)
                    self.matched_scene_data[k]["real"] = {
                        "pointcloud": real_pcds,
                        "ee_mask": real_pcd_ee_masks,
                    }
            self.matched_scene_data_key_list = list(self.matched_scene_data.keys())
            # shuffle to avoid bias in scene orders
            self._random_state.shuffle(self.matched_scene_data_key_list)
        else:
            self.matched_scene_data = None

    def __getitem__(self, idx):
        main_data = self._get_main_data()
        if self._matched_scene_fpath is not None:
            if (
                len(self._sim_pcd_sampling_idxs) == 0
                or self._refresh_pcd_sampling_curr_steps
                >= self._refresh_pcd_sampling_idxs_interval
            ):
                # refresh the sampling idxs
                self._refresh_pcd_sampling_curr_steps = 0
                for k in self.matched_scene_data.keys():
                    sim_pcd = self.matched_scene_data[k]["sim"]["pointcloud"]
                    sampling_idxs = self._random_state.permutation(sim_pcd.shape[0])[
                        : self.sampled_pcd_points
                    ]
                    self._sim_pcd_sampling_idxs[k] = sampling_idxs
                    self._real_pcd_sampling_idxs[k] = []
                    for real_pcd_sample_idx in range(
                        len(self.matched_scene_data[k]["real"])
                    ):
                        real_pcd = self.matched_scene_data[k]["real"]["pointcloud"][
                            real_pcd_sample_idx
                        ]
                        sampling_idxs = self._random_state.permutation(
                            real_pcd.shape[0]
                        )[: self.sampled_pcd_points]
                        self._real_pcd_sampling_idxs[k].append(sampling_idxs)
            self._refresh_pcd_sampling_curr_steps += 1

            real_pcds, real_pcd_ee_masks = [], []
            sim_pcds, sim_pcd_ee_masks = [], []
            k = self.matched_scene_data_key_list[idx % len(self.matched_scene_data)]
            sim_pcd = self.matched_scene_data[k]["sim"]["pointcloud"]
            sim_pcd_ee_mask = self.matched_scene_data[k]["sim"]["ee_mask"]
            sampling_idxs = self._sim_pcd_sampling_idxs[k]
            sim_pcds.append(sim_pcd[sampling_idxs])
            sim_pcd_ee_masks.append(sim_pcd_ee_mask[sampling_idxs])

            real_pcd_sample_idx = self._random_state.randint(
                0, len(self.matched_scene_data[k]["real"])
            )
            real_pcd = self.matched_scene_data[k]["real"]["pointcloud"][
                real_pcd_sample_idx
            ]
            real_pcd_ee_mask = self.matched_scene_data[k]["real"]["ee_mask"][
                real_pcd_sample_idx
            ]
            sampling_idxs = self._real_pcd_sampling_idxs[k][real_pcd_sample_idx]
            real_pcds.append(real_pcd[sampling_idxs])
            real_pcd_ee_masks.append(real_pcd_ee_mask[sampling_idxs])
            real_pcds = np.stack(real_pcds, axis=0)
            real_pcd_ee_masks = np.stack(real_pcd_ee_masks, axis=0)
            sim_pcds = np.stack(sim_pcds, axis=0)
            sim_pcd_ee_masks = np.stack(sim_pcd_ee_masks, axis=0)
            return main_data, (
                (real_pcds, real_pcd_ee_masks),
                (sim_pcds, sim_pcd_ee_masks),
            )
        return main_data

    def _get_main_data(self):
        main_data = any_slice(self.single_demo, self.step_pointer)
        self.step_pointer += 1
        # update to the next episode if we have reached the end of the current episode
        if self.step_pointer >= self._demos_len[self._demo_ids[self.demo_pointer]]:
            self.demo_pointer += 1
            # reset to the first episode if we have reached the end of the last episode
            self.demo_pointer %= len(self._demo_ids)
            self.step_pointer = 0
            self.single_demo = self.all_data[self.demo_pointer]
        return main_data

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self._fpath, "r", swmr=True, libver="latest")
        return self._hdf5_file

    @property
    def matched_scene_hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._matched_scene_hdf5_file is None:
            self._matched_scene_hdf5_file = h5py.File(
                self._matched_scene_fpath, "r", swmr=True, libver="latest"
            )
        return self._matched_scene_hdf5_file

    def _load_single_demo(self, demo_id):
        raw_data = {
            k: self.hdf5_file[demo_id][k][()].astype("float32")
            for k in self.hdf5_file[demo_id].keys()
        }
        # skip first n steps
        raw_data = {k: v[self._skip_first_n_steps :] for k, v in raw_data.items()}
        raw_actions = raw_data.pop("actions")
        gripper_actions = raw_actions[:, -1:]
        q_actions = raw_data["q"][
            1:
        ]  # q_{t + 1} is actually the action for time step t
        actions = np.concatenate([q_actions, gripper_actions], axis=1)
        raw_data = {k: v[:-1] for k, v in raw_data.items()}
        assert actions.shape[0] == raw_data["q"].shape[0]
        raw_data["actions"] = actions
        return raw_data

    def _close_and_delete_hdf5_handle(self):
        """
        Maybe close the file handle.
        """
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None
        if self._matched_scene_hdf5_file is not None:
            self._matched_scene_hdf5_file.close()
        self._matched_scene_hdf5_file = None

    def __len__(self):
        if self._len is None:
            self._len = sum(self._demos_len.values())
        return self._len

    def __del__(self):
        self._close_and_delete_hdf5_handle()


class DistillationSeqDataset(DistillationDataset):
    def _get_main_data(self):
        main_data = self.all_data[self.demo_pointer]
        self.demo_pointer += 1
        self.demo_pointer %= len(self._demo_ids)
        return main_data

    def __len__(self):
        if self._len is None:
            self._len = len(self._demo_ids)
        return self._len
