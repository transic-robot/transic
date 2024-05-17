import os
import argparse
import time

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys import config_root
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
import numpy as np
import torch
import h5py

from transic.distillation.policy import PointNetPolicy
from transic.utils.torch_utils import load_state_dict
from transic.utils.tree_utils import stack_sequence_fields
from transic.utils.array import any_to_torch_tensor, any_concat, any_to_numpy


logger = get_deoxys_example_logger()


dof_upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
dof_lower_limits = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-policy-ckpt-path", type=str, required=True)
    parser.add_argument("--data-save-path", type=str, required=True)
    parser.add_argument("--interface-cfg", type=str, default="config/charmander.yml")
    parser.add_argument("--controller-type", type=str, default="JOINT_POSITION")

    args = parser.parse_args()

    # mkdir if not exist
    os.makedirs(args.data_save_path, exist_ok=True)

    # file name is yyyy-mm-dd-hh-mm-ss
    strftime = time.strftime("%Y-%m-%d-%H-%M-%S")
    data_save_fname = os.path.join(args.data_save_path, f"{strftime}.hdf5")
    data_save_file = h5py.File(data_save_fname, "w")
    data_save_file.attrs["Collected Time"] = strftime
    data_save_file.attrs["Base Policy Ckpt"] = args.base_policy_ckpt_path

    ckpt = torch.load(args.base_policy_ckpt_path, map_location="cpu")

    policy = PointNetPolicy(
        point_channels=3,
        subtract_point_mean=False,
        add_ee_embd=True,
        ee_embd_dim=128,
        pointnet_output_dim=256,
        pointnet_hidden_dim=256,
        pointnet_hidden_depth=2,
        pointnet_activation="gelu",
        prop_input_dim=29,
        feature_fusion_hidden_depth=1,
        feature_fusion_hidden_dim=512,
        feature_fusion_output_dim=512,
        feature_fusion_activation="relu",
        feature_fusion_add_input_activation=False,
        feature_fusion_add_output_activation=False,
        action_dim=8,
        action_net_gmm_n_modes=5,
        action_net_hidden_dim=128,
        action_net_hidden_depth=3,
        action_net_activation="relu",
        deterministic_inference=True,
        gmm_low_noise_eval=True,
    )

    load_state_dict(
        policy, ckpt["state_dict"], strip_prefix="student_policy.", strict=True
    )

    policy.cuda()
    policy.eval()

    robot_interface = FrankaInterface(
        config_root + "/charmander.yml",
        use_visualizer=False,
        control_freq=20,
    )
    # TODO: change to your real-world observable interface
    observable = ObservableRedisSubInterface()

    controller_type = args.controller_type
    controller_cfg = get_default_controller_config(controller_type=controller_type)

    # TODO: change to your real-robot controller
    pose_follower = JointPositionControlFollower(robot_interface, controller_cfg)

    # space mouse for human in the loop
    space_mouse = SpaceMouse(
        vendor_id=9583, product_id=50734, pos_sensitivity=0.25, rot_sensitivity=0.1
    )
    space_mouse_controller_cfg = get_default_controller_config(
        controller_type="OSC_POSE"
    )
    space_mouse.start_control()

    observable.start()

    time.sleep(3)

    robot_interface.gripper_control(-1.0)

    input("Press Enter to continue...")

    # data save fields
    policy_obs_to_save = []
    policy_action_to_save = []
    is_human_intervention_to_save = []
    pre_intervention_eef_pose_to_save = []
    post_intervention_eef_pose_to_save = []
    pre_intervention_q_to_save = []
    post_intervention_q_to_save = []
    pre_intervention_gripper_q_to_save = []
    post_intervention_gripper_q_to_save = []

    obs = observable.get_obs()
    robot_state = robot_interface._state_buffer[-1]
    gripper_state = robot_interface._gripper_state_buffer[-1]
    q = np.array(robot_state.q)
    eef_quat, eef_pos = robot_interface.last_eef_quat_and_pos
    prop_obs = {
        "q": q,
        "cos_q": np.cos(q),
        "sin_q": np.sin(q),
        "dq": np.array(robot_state.dq),
        "eef_pos": eef_pos[:, 0],
        "eef_quat": eef_quat,
        "gripper_width": np.array([gripper_state.width]),
    }

    prop_keys = ["q", "cos_q", "sin_q", "eef_pos", "eef_quat", "gripper_width"]

    try:
        while True:
            pc_obs = {
                "coordinate": any_to_torch_tensor(
                    obs["pointcloud"], device="cuda", dtype=torch.float32
                ).unsqueeze(0),
                "ee_mask": any_to_torch_tensor(
                    obs["ee_mask"], device="cuda", dtype=torch.bool
                ).unsqueeze(0),
            }
            prop_obs = any_to_torch_tensor(
                any_concat([prop_obs[k] for k in prop_keys], dim=0)[None, ...],
                device="cuda",
                dtype=torch.float32,
            )

            policy_obs_to_save.append(
                {
                    "pcd": {k: any_to_numpy(v)[0] for k, v in pc_obs.items()},
                    "proprioception": any_to_numpy(prop_obs)[0],
                }
            )

            obs_dict = {
                "pcd": pc_obs,
                "proprioception": prop_obs,
            }

            with torch.no_grad():
                policy_action = policy.act(obs_dict)
            policy_action_to_save.append(any_to_numpy(policy_action)[0])
            action = policy_action.cpu().numpy()[0]

            # clip to +- 1
            action = np.clip(action, -1.0, 1.0)
            arm_action, gripper_action = action[:-1], action[-1]

            # flip the gripper action
            gripper_action = -1 * gripper_action
            if gripper_action < 0:
                gripper_action = -1  # binary close and open

            # scale arm actions
            arm_action = (arm_action + 1) / 2 * (
                dof_upper_limits - dof_lower_limits
            ) + dof_lower_limits

            pose_follower.set_new_goal(arm_action, gripper_action)

            while True:
                pose_follower.control()
                if pose_follower.is_goal_arrived:
                    break

            # decide if we need human intervention here
            need_human_intervention = input("Need human intervention? (y/n)") == "y"
            is_human_intervention_to_save.append(int(need_human_intervention))
            pre_intervention_eef_pose_to_save.append(robot_interface.last_eef_pose)
            pre_intervention_q_to_save.append(robot_interface.last_q)
            pre_intervention_gripper_q_to_save.append(robot_interface.last_gripper_q)
            if need_human_intervention:
                while True:
                    human_action, _, exit_human_intervention = input2action(
                        device=space_mouse,
                        controller_type="OSC_POSE",
                        lock_rotation=False,
                    )
                    robot_interface.control(
                        controller_type="OSC_POSE",
                        action=human_action,
                        controller_cfg=space_mouse_controller_cfg,
                    )
                    if exit_human_intervention:
                        print("Exiting human intervention...")
                        break
            post_intervention_eef_pose_to_save.append(robot_interface.last_eef_pose)
            post_intervention_q_to_save.append(robot_interface.last_q)
            post_intervention_gripper_q_to_save.append(robot_interface.last_gripper_q)

            obs = observable.get_obs()
            robot_state = robot_interface._state_buffer[-1]
            gripper_state = robot_interface._gripper_state_buffer[-1]
            q = np.array(robot_state.q)
            eef_quat, eef_pos = robot_interface.last_eef_quat_and_pos
            prop_obs = {
                "q": q,
                "cos_q": np.cos(q),
                "sin_q": np.sin(q),
                "dq": np.array(robot_state.dq),
                "eef_pos": eef_pos[:, 0],
                "eef_quat": eef_quat,
                "gripper_width": np.array([gripper_state.width]),
            }
    except KeyboardInterrupt:
        pre_intervention_q_to_save.append(robot_interface.last_q)
        post_intervention_q_to_save.append(robot_interface.last_q)
        pre_intervention_gripper_q_to_save.append(robot_interface.last_gripper_q)
        post_intervention_gripper_q_to_save.append(robot_interface.last_gripper_q)
        is_human_intervention_to_save.append(0)
        pre_intervention_eef_pose_to_save.append(robot_interface.last_eef_pose)
        post_intervention_eef_pose_to_save.append(robot_interface.last_eef_pose)

        is_task_successful = input("Is task successful? (y/n)") == "y"

        # because measured pointcloud is variable length
        # we find the max number of points and pad the pointcloud
        max_pcd_n = max([len(d["pcd"]["coordinate"]) for d in policy_obs_to_save])
        for each_obs in policy_obs_to_save:
            pcd, ee_mask = each_obs["pcd"]["coordinate"], each_obs["pcd"]["ee_mask"]
            padded_pcd = any_concat(
                [pcd, np.zeros((max_pcd_n - len(pcd), 3), dtype=np.float32)], dim=0
            )
            padded_ee_mask = any_concat(
                [ee_mask, np.zeros((max_pcd_n - len(ee_mask),), dtype=bool)], dim=0
            )
            padding_mask = any_concat(
                [
                    np.ones((len(pcd),), dtype=bool),
                    np.zeros((max_pcd_n - len(pcd),), dtype=bool),
                ],
            )
            each_obs["pcd"]["coordinate"] = padded_pcd
            each_obs["pcd"]["ee_mask"] = padded_ee_mask
            each_obs["pcd"]["padding_mask"] = padding_mask

        policy_obs_to_save = stack_sequence_fields(policy_obs_to_save)
        policy_action_to_save = np.stack(policy_action_to_save)
        is_human_intervention_to_save = np.stack(is_human_intervention_to_save)
        pre_intervention_eef_pose_to_save = np.stack(pre_intervention_eef_pose_to_save)
        post_intervention_eef_pose_to_save = np.stack(
            post_intervention_eef_pose_to_save
        )
        pre_intervention_q_to_save = np.stack(pre_intervention_q_to_save)
        post_intervention_q_to_save = np.stack(post_intervention_q_to_save)
        pre_intervention_gripper_q_to_save = np.stack(
            pre_intervention_gripper_q_to_save
        )
        post_intervention_gripper_q_to_save = np.stack(
            post_intervention_gripper_q_to_save
        )

        policy_obs_grp = data_save_file.create_group("policy_obs")
        policy_obs_grp.create_dataset(
            "proprioception", data=policy_obs_to_save["proprioception"]
        )
        pcd_grp = policy_obs_grp.create_group("pcd")
        for k, v in policy_obs_to_save["pcd"].items():
            pcd_grp.create_dataset(k, data=v)
        data_save_file.create_dataset("policy_action", data=policy_action_to_save)
        data_save_file.create_dataset(
            "is_human_intervention", data=is_human_intervention_to_save
        )
        data_save_file.create_dataset(
            "pre_intervention_eef_pose", data=pre_intervention_eef_pose_to_save
        )
        data_save_file.create_dataset(
            "post_intervention_eef_pose", data=post_intervention_eef_pose_to_save
        )
        data_save_file.create_dataset(
            "pre_intervention_q", data=pre_intervention_q_to_save
        )
        data_save_file.create_dataset(
            "post_intervention_q", data=post_intervention_q_to_save
        )
        data_save_file.create_dataset(
            "pre_intervention_gripper_q", data=pre_intervention_gripper_q_to_save
        )
        data_save_file.create_dataset(
            "post_intervention_gripper_q", data=post_intervention_gripper_q_to_save
        )
        data_save_file.attrs["is_task_successful"] = is_task_successful

        data_save_file.close()
        robot_interface.close()


if __name__ == "__main__":
    main()
