import argparse
import time

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys import config_root
import torch
import numpy as np

from transic.distillation.policy import PointNetPolicy
from transic.residual.policy import PerceiverResidualPolicy
from transic.utils.torch_utils import load_state_dict
from transic.utils.array import any_to_torch_tensor, any_concat


logger = get_deoxys_example_logger()


dof_upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
dof_lower_limits = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
)
delta_q_upper_limits = dof_upper_limits - dof_lower_limits
delta_q_lower_limits = dof_lower_limits - dof_upper_limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-policy-ckpt-path", type=str, required=True)
    parser.add_argument("--residual-policy-ckpt-path", type=str, required=True)
    parser.add_argument("--interface-cfg", type=str, default="config/charmander.yml")
    parser.add_argument("--controller-type", type=str, default="JOINT_POSITION")

    args = parser.parse_args()

    base_policy_ckpt = torch.load(args.base_policy_ckpt_path, map_location="cpu")

    base_policy = PointNetPolicy(
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
        gmm_low_noise_eval=False,
    )

    load_state_dict(
        base_policy,
        base_policy_ckpt["state_dict"],
        strip_prefix="student_policy.",
        strict=True,
    )

    base_policy.cuda()
    base_policy.eval()

    residual_policy_ckpt = torch.load(
        args.residual_policy_ckpt_path, map_location="cpu"
    )
    residual_policy = PerceiverResidualPolicy(
        point_channels=3,
        subtract_point_mean=False,
        add_ee_embd=True,
        ee_embd_dim=128,
        set_xf_hidden_dim=256,
        set_xf_num_heads=8,
        set_xf_num_queries=8,
        set_xf_pool_type="concat",
        set_xf_layer_norm=False,
        prop_input_dim=29,
        robot_policy_output_dim=7,
        include_robot_policy_gripper_action_input=True,
        robot_policy_gripper_action_embd_dim=64,
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
        intervention_head_hidden_dim=128,
        intervention_head_hidden_depth=3,
        intervention_head_activation="relu",
        deterministic_inference=True,
        gmm_low_noise_eval=False,
    )
    load_state_dict(
        residual_policy,
        residual_policy_ckpt["state_dict"],
        strip_prefix="residual_policy.",
        strict=True,
    )
    residual_policy.cuda()
    residual_policy.eval()

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

    observable.start()

    time.sleep(3)

    robot_interface.gripper_control(-1.0)

    input("Press Enter to continue...")

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

            obs_dict = {
                "pcd": pc_obs,
                "proprioception": prop_obs,
            }

            with torch.no_grad():
                policy_action = base_policy.act(obs_dict)
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

            robot_policy_q_action, robot_policy_gripper_action = (
                policy_action[..., :-1],
                policy_action[..., -1],
            )
            # robot_policy_gripper_action is within [-1, 1], we rectify to {0, 1}
            robot_policy_gripper_action = torch.where(
                robot_policy_gripper_action >= 0, 1, 0
            )
            residual_policy_obs_dict = {
                "robot_policy_action": robot_policy_q_action,
                "robot_policy_gripper_action": robot_policy_gripper_action.long(),
            }
            residual_policy_obs_dict.update(obs_dict)
            with torch.no_grad():
                residual_action, engage_intervention = residual_policy.act(
                    residual_policy_obs_dict
                )

            # decide if we need to engage residual policy
            need_residual_policy = engage_intervention

            if need_residual_policy:
                print("Engaging residual policy")
                residual_action = residual_action.cpu().numpy()[0]  # (8,)
                residual_q_action, residual_gripper_action = (
                    residual_action[:-1],
                    residual_action[-1],
                )
                # residual_q_action is within [-1, 1], de-normalize it
                residual_q_action = (
                    0.5
                    * (residual_q_action + 1)
                    * (delta_q_upper_limits - delta_q_lower_limits)
                    + delta_q_lower_limits
                )
                # add to main policy action
                arm_action = arm_action + residual_q_action
                # clamp to valid range
                arm_action = np.clip(arm_action, dof_lower_limits, dof_upper_limits)
                # decide if we need to negate robot policy's gripper action
                # residual_gripper_action is within [-1, 1], first rectify to {0, 1}
                residual_gripper_action = 1 if residual_gripper_action >= 0 else 0
                if residual_gripper_action == 1:
                    print("Negating robot policy's gripper action")
                    gripper_action = -1 * gripper_action
            else:
                print("NOT engaging residual policy")
            pose_follower.set_new_goal(arm_action, gripper_action)

            while True:
                pose_follower.control()
                if pose_follower.is_goal_arrived:
                    break

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
        robot_interface.close()


if __name__ == "__main__":
    main()
