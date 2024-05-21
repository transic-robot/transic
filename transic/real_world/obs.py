from typing import List, Tuple
import os
import numpy as np
import torch

try:
    import open3d as o3d
    from deoxys.franka_interface import FrankaInterface
except ImportError as e:
    print("Please install open3d and deoxys!")
    raise e

from transic.utils.array import any_concat, any_stack
from transic_envs.utils.fb_control_utils import batched_pose2mat
from transic_envs.utils.torch_jit_utils import (
    matrix_to_quaternion,
    axisangle2quat,
    quat_mul,
)


class PointCloudAndPropObservable:
    """
    Example code for real-world observation pipeline.

    Args:
        camera_ids: List of camera ids.
        camera_transformations: List of camera transformation in 4x4 homogeneous matrix.
        is_wrist: List of boolean values indicating whether the camera is a wrist camera.
        pointcloud_sampled_points: Point cloud downsampled points.
        pointcloud_filter_x: Tuple of x-axis range to filter the point cloud.
        pointcloud_filter_y: Tuple of y-axis range to filter the point cloud.
        pointcloud_filter_z_min: Minimum z-axis value to filter the point cloud.
        gripper_pcd_from_fk: Whether to add gripper point cloud from forward kinematics.
        robot_interface: FrankaInterface object.
    """

    def __init__(
        self,
        *,
        camera_ids: List[int],
        camera_transformations: List[np.ndarray],
        is_wrist: List[bool],
        pointcloud_sampled_points: int,
        pointcloud_filter_x: Tuple[float, float],
        pointcloud_filter_y: Tuple[float, float],
        pointcloud_filter_z_min: float,
        gripper_pcd_from_fk: bool = False,
        robot_interface: FrankaInterface,
    ):
        assert len(camera_ids) == len(camera_transformations) == len(is_wrist)
        assert sum(is_wrist) <= 1, "support at most one wrist camera"

        self.camera_interfaces = [
            # TODO
            """
            Create your camera interface here
            """
            for camera_id in camera_ids
        ]
        self.camera_transformations = np.stack(
            [t for t, wrist in zip(camera_transformations, is_wrist) if not wrist],
            axis=0,
        )  # (n_cameras, 4, 4)
        self.wrist_transformation = None
        self.wrist_camera_interface = None
        for t, wrist, camera_interface in zip(
            camera_transformations, is_wrist, self.camera_interfaces
        ):
            if wrist:
                self.wrist_transformation = t
                self.wrist_camera_interface = camera_interface
        self.is_wrist = is_wrist
        self.pointcloud_sampled_points = pointcloud_sampled_points
        assert (
            isinstance(pointcloud_filter_x, tuple)
            and len(pointcloud_filter_x) == 2
            and pointcloud_filter_x[0] < pointcloud_filter_x[1]
        )
        assert (
            isinstance(pointcloud_filter_y, tuple)
            and len(pointcloud_filter_y) == 2
            and pointcloud_filter_y[0] < pointcloud_filter_y[1]
        )
        self.pointcloud_filter_x = pointcloud_filter_x
        self.pointcloud_filter_y = pointcloud_filter_y
        self.pointcloud_filter_z_min = pointcloud_filter_z_min
        self._gripper_site_z = None
        self.robot_interface = robot_interface

        self.gripper_pcd_from_fk = gripper_pcd_from_fk
        if gripper_pcd_from_fk:
            from transic_envs.asset_root import ASSET_ROOT
            from transic_envs.utils.urdf2casadi import URDFparser

            franka_parser = URDFparser()
            franka_parser.from_file(
                os.path.join(
                    ASSET_ROOT,
                    "franka_description/robots/franka_panda_finray.urdf",
                )
            )
            self._leftfinger_fk_fn = franka_parser.get_forward_kinematics(
                root="panda_link0", tip="panda_leftfinger"
            )["T_fk"]
            self._rightfinger_fk_fn = franka_parser.get_forward_kinematics(
                root="panda_link0", tip="panda_rightfinger"
            )["T_fk"]

            finger_pcd = np.load(
                os.path.join(
                    ASSET_ROOT,
                    "franka_description/meshes/collision/finray_finger.npy",
                )
            )  # (N, 3)
            self._finger_pcd = torch.tensor(
                np.hstack((finger_pcd, np.ones((finger_pcd.shape[0], 1)))),
                dtype=torch.float32,
            )  # (N, 4)
        else:
            self._leftfinger_fk_fn = None
            self._rightfinger_fk_fn = None
            self._finger_pcd = None

    def get_obs(self):
        obs = {}
        prop = self._get_prop()
        obs.update(prop)
        pcd, ee_mask = self._get_pointcloud()
        obs["pointcloud"] = pcd
        obs["ee_mask"] = ee_mask
        return obs

    def _get_pointcloud(self):
        # fixed cameras
        all_points = []
        for interface, wrist in zip(self.camera_interfaces, self.is_wrist):
            if wrist:
                continue
            points = interface.get_img()["points"]
            points = np.hstack((points, np.ones((points.shape[0], 1))))
            all_points.append(points)
        max_points = max([points.shape[0] for points in all_points])
        padding_masks = np.zeros(
            (len(all_points), max_points), dtype=bool
        )  # (n_cameras, max_points)
        for i in range(len(all_points)):
            padding_masks[i, : all_points[i].shape[0]] = True
            all_points[i] = any_concat(
                [all_points[i], np.zeros((max_points - all_points[i].shape[0], 4))],
                dim=0,
            )  # (max_points, 4)
        all_points = any_stack(all_points, dim=0)  # (n_cameras, max_points, 4)

        all_points = all_points.transpose((0, 2, 1))  # (3, 4, N)
        all_points = self.camera_transformations @ all_points  # (3, 4, N)
        all_points = all_points.transpose((0, 2, 1))[:, :, :3]  # (3, N, 3)
        all_points = all_points.reshape((-1, 3))  # (n_cameras * max_points, 3)
        pcd_data = all_points[padding_masks.reshape(-1)]  # (n_cameras * max_points, 3)

        # wrist camera
        if len(self.robot_interface._state_buffer) > 0:
            gripper2base = self.robot_interface.last_eef_pose
            points_wrist_camera = self.wrist_camera_interface.get_img()["points"]
            points_wrist_camera = np.hstack(
                (points_wrist_camera, np.ones((points_wrist_camera.shape[0], 1)))
            )
            points_wrist_camera = points_wrist_camera.transpose()
            points_in_gripper = self.wrist_transformation @ points_wrist_camera
            points_in_base = gripper2base @ points_in_gripper
            points_in_base = points_in_base.transpose()[:, :3]
            pcd_data = np.vstack((pcd_data, points_in_base))
        else:
            print("[Warning] No robot connection, skip wrist camera...")

        # filter out based on locations
        x_mask = np.logical_and(
            pcd_data[:, 0] > self.pointcloud_filter_x[0],
            pcd_data[:, 0] < self.pointcloud_filter_x[1],
        )
        y_mask = np.logical_and(
            pcd_data[:, 1] > self.pointcloud_filter_y[0],
            pcd_data[:, 1] < self.pointcloud_filter_y[1],
        )
        z_mask = np.logical_and(
            pcd_data[:, 2] > self.pointcloud_filter_z_min,
            pcd_data[:, 2] < (self._gripper_site_z or 999.9),
        )
        mask = np.logical_and(np.logical_and(x_mask, y_mask), z_mask)
        pcd_data = pcd_data[mask]

        # add finger pcd from forward kinematics
        finger_pcd_data = None
        if self.gripper_pcd_from_fk:
            q = self.robot_interface.last_q  # (7,)
            q_gripper = self.robot_interface.last_gripper_q / 2  # (1,)
            q = np.concatenate((q, [q_gripper]), axis=0)  # (8,)
            q = list(q)
            leftfinger_pose = (
                np.array(self._leftfinger_fk_fn(q)).astype(np.float32).reshape((4, 4))
            )  # (4, 4)
            rightfinger_pose = (
                np.array(self._rightfinger_fk_fn(q)).astype(np.float32).reshape((4, 4))
            )

            leftfinger_pos = leftfinger_pose[:3, 3].reshape((1, 3))  # (1, 3)
            rightfinger_pos = rightfinger_pose[:3, 3].reshape((1, 3))
            leftfinger_pos = torch.tensor(leftfinger_pos)
            rightfinger_pos = torch.tensor(rightfinger_pos)

            leftfinger_rot = leftfinger_pose[:3, :3]  # (3, 3)
            leftfinger_quat = matrix_to_quaternion(torch.tensor(leftfinger_rot))
            rightfinger_rot = rightfinger_pose[:3, :3]
            rightfinger_quat = matrix_to_quaternion(torch.tensor(rightfinger_rot))

            # convert quat from wxyz order to xyzw order
            leftfinger_quat = torch.cat(
                (leftfinger_quat[1:], leftfinger_quat[:1]), dim=0
            ).unsqueeze(
                0
            )  # (1, 4)
            rightfinger_quat = torch.cat(
                (rightfinger_quat[1:], rightfinger_quat[:1]), dim=0
            ).unsqueeze(
                0
            )  # (1, 4)

            rot_bias = axisangle2quat(torch.tensor([0, 0, np.pi])).unsqueeze(
                0
            )  # (1, 4)

            leftfinger_quat = quat_mul(leftfinger_quat, rot_bias)
            leftfinger_quat = leftfinger_quat[..., [3, 0, 1, 2]]  # (1, 4)
            leftfinger_tf = batched_pose2mat(
                leftfinger_pos, leftfinger_quat, device=leftfinger_quat.device
            )
            leftfinger_pcd_transformed = leftfinger_tf @ self._finger_pcd.T  # (1, 4, N)
            leftfinger_pcd_transformed = leftfinger_pcd_transformed.squeeze(0).T[
                :, :3
            ]  # (N, 3)
            leftfinger_pcd_transformed = leftfinger_pcd_transformed.numpy()

            # right finger needs to be flipped
            flip_rot = axisangle2quat(
                torch.tensor([0, 0, np.pi], dtype=torch.float32)
            ).unsqueeze(
                0
            )  # (1, 4)
            rightfinger_quat = quat_mul(rightfinger_quat, rot_bias)
            rightfinger_quat = quat_mul(rightfinger_quat, flip_rot)
            rightfinger_quat = rightfinger_quat[..., [3, 0, 1, 2]]
            rightfinger_tf = batched_pose2mat(
                rightfinger_pos, rightfinger_quat, device=rightfinger_quat.device
            )
            rightfinger_pcd_transformed = rightfinger_tf @ self._finger_pcd.T
            rightfinger_pcd_transformed = rightfinger_pcd_transformed.squeeze(0).T[
                :, :3
            ]
            rightfinger_pcd_transformed = rightfinger_pcd_transformed.numpy()
            finger_pcd_data = np.concatenate(
                (leftfinger_pcd_transformed, rightfinger_pcd_transformed),
                axis=0,
            )  # (N, 3)

            # remove finger pcd captured by cameras
            sample_ratio = self.pointcloud_sampled_points / (
                pcd_data.shape[0] + finger_pcd_data.shape[0]
            )
            pcd_data_initial_sample_idxs = np.random.permutation(pcd_data.shape[0])[
                : self.pointcloud_sampled_points
            ]
            pcd_data = pcd_data[pcd_data_initial_sample_idxs]

            # build a KD tree for pcd_data
            pcd_data_kd_tree = o3d.geometry.KDTreeFlann(
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_data))
            )
            nn_idxs = []
            for finger_pcd_point in finger_pcd_data:
                [_, idx, _] = pcd_data_kd_tree.search_knn_vector_3d(
                    finger_pcd_point, 10
                )
                nn_idxs += list(idx)
            # deduplicate nn_idxs
            nn_idxs = list(set(nn_idxs))
            # remove the nearest neighbor points from pcd_data
            pcd_data = np.delete(pcd_data, nn_idxs, axis=0)
            finger_pcd_data_sample_amount = int(sample_ratio * finger_pcd_data.shape[0])
            finger_pcd_sample_idxs = np.random.permutation(finger_pcd_data.shape[0])[
                :finger_pcd_data_sample_amount
            ]
            finger_pcd_data = finger_pcd_data[finger_pcd_sample_idxs]

        # remove statistical outlier
        pcd_data_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_data))
        pcd_data_o3d, _ = pcd_data_o3d.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=1.3
        )
        # remove radius outlier
        pcd_data_o3d, _ = pcd_data_o3d.remove_radius_outlier(nb_points=20, radius=0.03)
        pcd_data = np.asarray(pcd_data_o3d.points)

        scene_pcd_ee_mask = np.zeros((pcd_data.shape[0],), dtype=bool)
        ee_mask = [scene_pcd_ee_mask]
        pcd_data_rtn = [pcd_data]
        if finger_pcd_data is not None:
            finger_pcd_ee_mask = np.ones((finger_pcd_data.shape[0],), dtype=bool)
            ee_mask.append(finger_pcd_ee_mask)
            pcd_data_rtn.append(finger_pcd_data)
        ee_mask = np.concatenate(ee_mask, axis=0)
        pcd_data = np.concatenate(pcd_data_rtn, axis=0)

        # sample points
        if pcd_data.shape[0] > self.pointcloud_sampled_points:
            sampling_idx = np.random.permutation(pcd_data.shape[0])[
                : self.pointcloud_sampled_points
            ]
            pcd_data = pcd_data[sampling_idx]
            ee_mask = ee_mask[sampling_idx]
        return pcd_data.astype(np.float32), ee_mask.astype(bool)

    def _get_prop(self):
        prop = {}
        if len(self.robot_interface._state_buffer) == 0:
            prop["q"] = None
            prop["cos_q"] = None
            prop["sin_q"] = None
            prop["dq"] = None
            prop["eef_pos"] = None
            prop["eef_quat"] = None
        else:
            robot_state = self.robot_interface._state_buffer[-1]
            q = np.array(robot_state.q)
            prop["q"] = q
            prop["cos_q"] = np.cos(q)
            prop["sin_q"] = np.sin(q)
            prop["dq"] = np.array(robot_state.dq)
            eef_quat, eef_pos = self.robot_interface.last_eef_quat_and_pos
            prop["eef_pos"] = eef_pos[:, 0]
            self._gripper_site_z = eef_pos[2]
            prop["eef_quat"] = eef_quat
        if len(self.robot_interface._gripper_state_buffer) == 0:
            prop["gripper_width"] = None
        else:
            gripper_state = self.robot_interface._gripper_state_buffer[-1]
            prop["gripper_width"] = np.array([gripper_state.width])
        return prop
