from functools import partial
import os
import shutil
import threading
import time
from datetime import datetime
import gym
import numpy as np
import torch
from os.path import basename
from typing import Optional
from rl_games.common import vecenv
from rl_games.common import env_configurations
from rl_games.algos_torch import model_builder
from rl_games.common.tr_helpers import unsqueeze_obs
from rl_games.algos_torch.torch_ext import safe_filesystem_op
import h5py

from transic.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields


def safe_load(filename):
    return safe_filesystem_op(partial(torch.load, map_location="cpu"), filename)


def load_checkpoint(filename):
    print("=> loading checkpoint '{}'".format(filename))
    state = safe_load(filename)
    return state


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class MyBasePlayer(object):
    def __init__(
        self,
        params,
        *,
        # rollout dumping for distillation
        save_rollouts: bool = False,
        rollout_saving_fpath: Optional[str] = None,
        save_successful_rollouts_only: bool = True,
        num_rollouts_to_save: int = 10000,
        # weird bug in states if episodes are too short
        min_episode_length: int = 25,
    ):
        self.config = config = params["config"]
        self.load_networks(params)
        self.env_name = self.config["env_name"]
        self.player_config = self.config.get("player", {})
        self.env_config = self.config.get("env_config", {})
        self.env_config = self.player_config.get("env_config", self.env_config)
        self.env_info = self.config.get("env_info")
        self.clip_actions = config.get("clip_actions", True)
        self.seed = self.env_config.pop("seed", None)

        if self.env_info is None:
            use_vecenv = self.player_config.get("use_vecenv", False)
            if use_vecenv:
                print("[BasePlayer] Creating vecenv: ", self.env_name)
                self.env = vecenv.create_vec_env(
                    self.env_name, self.config["num_actors"], **self.env_config
                )
                self.env_info = self.env.get_env_info()
            else:
                print("[BasePlayer] Creating regular env: ", self.env_name)
                self.env = self.create_env()
                self.env_info = env_configurations.get_env_info(self.env)
        else:
            self.env = config.get("vec_env")

        self.num_agents = self.env_info.get("agents", 1)
        self.value_size = self.env_info.get("value_size", 1)
        self.action_space = self.env_info["action_space"]

        self.observation_space = self.env_info["observation_space"]
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = self.observation_space
        else:
            self.obs_shape = self.observation_space.shape
        self.is_tensor_obses = False

        self.states = None
        self.player_config = self.config.get("player", {})
        self.use_cuda = True
        self.batch_size = 1
        self.has_batch_dimension = False
        self.has_central_value = self.config.get("central_value_config") is not None
        self.device_name = self.config.get("device_name", "cuda")
        self.render_env = self.player_config.get("render", False)

        if "deterministic" in self.player_config:
            self.is_deterministic = self.player_config["deterministic"]
        else:
            self.is_deterministic = self.player_config.get("deterministic", True)

        self.print_stats = self.player_config.get("print_stats", True)
        self.render_sleep = self.player_config.get("render_sleep", 0.002)
        self.max_steps = 108000 // 4
        self.device = torch.device(self.device_name)

        self.evaluation = self.player_config.get("evaluation", False)
        self.update_checkpoint_freq = self.player_config.get(
            "update_checkpoint_freq", 100
        )
        # if we run player as evaluation worker this will take care of loading new checkpoints
        self.dir_to_monitor = self.player_config.get("dir_to_monitor")
        # path to the newest checkpoint
        self.checkpoint_to_load: Optional[str] = None

        if self.evaluation and self.dir_to_monitor is not None:
            self.checkpoint_mutex = threading.Lock()
            self.eval_checkpoint_dir = os.path.join(
                self.dir_to_monitor, "eval_checkpoints"
            )
            os.makedirs(self.eval_checkpoint_dir, exist_ok=True)

            patterns = ["*.pth"]
            from watchdog.observers import Observer
            from watchdog.events import PatternMatchingEventHandler

            self.file_events = PatternMatchingEventHandler(patterns)
            self.file_events.on_created = self.on_file_created
            self.file_events.on_modified = self.on_file_modified

            self.file_observer = Observer()
            self.file_observer.schedule(
                self.file_events, self.dir_to_monitor, recursive=False
            )
            self.file_observer.start()

        # initialize wandb if needed
        if "features" in config:
            self.wandb = config["features"]["observer"].wandb_observer
            if self.wandb is not None:
                self.wandb.before_init(
                    "test",
                    config,
                    config["name"] + datetime.now().strftime("_%m-%d-%H-%M-%S"),
                )

        # rollout dumping
        self.save_rollouts = save_rollouts
        self.save_successful_rollouts_only = save_successful_rollouts_only
        self.num_rollouts_to_save = num_rollouts_to_save
        self.min_episode_length = min_episode_length

        if save_rollouts:
            os.makedirs(os.path.dirname(rollout_saving_fpath), exist_ok=True)
            self.h5py_file = h5py.File(rollout_saving_fpath, "w")
            grp = self.h5py_file.create_group("rollouts")
            self.s_grp = grp.create_group("successful")
            self.f_grp = grp.create_group("failed")
            self.saved_successful_rollouts, self.saved_failed_rollouts = 0, 0
        else:
            self.h5py_file, self.s_grp, self.f_grp = None, None, None
            self.saved_successful_rollouts, self.saved_failed_rollouts = None, None

    def wait_for_checkpoint(self):
        if self.dir_to_monitor is None:
            return

        attempt = 0
        while True:
            attempt += 1
            with self.checkpoint_mutex:
                if self.checkpoint_to_load is not None:
                    if attempt % 10 == 0:
                        print(
                            f"Evaluation: waiting for new checkpoint in {self.dir_to_monitor}..."
                        )
                    break
            time.sleep(1.0)

        print(f"Checkpoint {self.checkpoint_to_load} is available!")

    def maybe_load_new_checkpoint(self):
        # lock mutex while loading new checkpoint
        with self.checkpoint_mutex:
            if self.checkpoint_to_load is not None:
                print(
                    f"Evaluation: loading new checkpoint {self.checkpoint_to_load}..."
                )
                # try if we can load anything from the pth file, this will quickly fail if the file is corrupted
                # without triggering the retry loop in "safe_filesystem_op()"
                load_error = False
                try:
                    torch.load(self.checkpoint_to_load)
                except Exception as e:
                    print(
                        f"Evaluation: checkpoint file is likely corrupted {self.checkpoint_to_load}: {e}"
                    )
                    load_error = True

                if not load_error:
                    try:
                        self.restore(self.checkpoint_to_load)
                    except Exception as e:
                        print(
                            f"Evaluation: failed to load new checkpoint {self.checkpoint_to_load}: {e}"
                        )

                # whether we succeeded or not, forget about this checkpoint
                self.checkpoint_to_load = None

    def process_new_eval_checkpoint(self, path):
        with self.checkpoint_mutex:
            # print(f"New checkpoint {path} available for evaluation")
            # copy file to eval_checkpoints dir using shutil
            # since we're running the evaluation worker in a separate process,
            # there is a chance that the file is changed/corrupted while we're copying it
            # not sure what we can do about this. In practice it never happened so far though
            try:
                eval_checkpoint_path = os.path.join(
                    self.eval_checkpoint_dir, basename(path)
                )
                shutil.copyfile(path, eval_checkpoint_path)
            except Exception as e:
                print(f"Failed to copy {path} to {eval_checkpoint_path}: {e}")
                return

            self.checkpoint_to_load = eval_checkpoint_path

    def on_file_created(self, event):
        self.process_new_eval_checkpoint(event.src_path)

    def on_file_modified(self, event):
        self.process_new_eval_checkpoint(event.src_path)

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config["network"] = builder.load(params)

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)
        if hasattr(obs, "dtype") and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return (
                self.obs_to_torch(obs),
                torch.from_numpy(rewards),
                torch.from_numpy(dones),
                infos,
            )

    def obs_to_torch(self, obs):
        if isinstance(obs, dict):
            if "obs" in obs:
                obs = obs["obs"]
            if isinstance(obs, dict):
                upd_obs = {}
                for key, value in obs.items():
                    upd_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                upd_obs = self.cast_obs(obs)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def _obs_to_tensors_internal(self, obs, cast_to_dict=True):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value, False)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert obs.dtype != np.int8
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        elif np.isscalar(obs):
            obs = torch.FloatTensor([obs]).to(self.device)
        return obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_reset(self, env):
        obs = env.reset()
        return self.obs_to_torch(obs)

    def restore(self, fn):
        raise NotImplementedError("restore")

    def get_weights(self):
        weights = {}
        weights["model"] = self.model.state_dict()
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights["model"])
        if self.normalize_input and "running_mean_std" in weights:
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])

    def create_env(self):
        return env_configurations.configurations[self.env_name]["env_creator"](
            **self.env_config
        )

    def get_action(self, obs, is_deterministic=False):
        raise NotImplementedError("step")

    def get_masked_action(self, obs, mask, is_deterministic=False):
        raise NotImplementedError("step")

    def reset(self):
        raise NotImplementedError("raise")

    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [
                torch.zeros(
                    (s.size()[0], self.batch_size, s.size()[2]), dtype=torch.float32
                ).to(self.device)
                for s in rnn_states
            ]

    def run(self):
        render = self.render_env
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        self.wait_for_checkpoint()

        if self.save_rollouts:
            rollouts = [[] for _ in range(self.env.num_envs)]
            rollouts_actions = [[] for _ in range(self.env.num_envs)]
            prev_done = None
            prev_success_buf = None

        need_init_rnn = self.is_rnn

        obses = self.env_reset(self.env)
        batch_size = 1
        batch_size = self.get_batch_size(obses, batch_size)

        if need_init_rnn:
            self.init_rnn()
            need_init_rnn = False

        cr = torch.zeros(batch_size, dtype=torch.float32)
        steps = torch.zeros(batch_size, dtype=torch.float32)

        while True:
            if has_masks:
                masks = self.env.get_action_mask()
                actions = self.get_masked_action(obses, masks, is_deterministic)
            else:
                actions = self.get_action(obses, is_deterministic)

            if self.save_rollouts:
                states_to_save = {
                    k: v.cpu().numpy() for k, v in self.env.dump_fileds.items()
                }
                states_to_save = unstack_sequence_fields(
                    states_to_save, batch_size=self.env.num_envs
                )  # list of dict
                for rollout, state_to_save in zip(rollouts, states_to_save):
                    rollout.append(state_to_save)

                for rollout_actions, action in zip(rollouts_actions, actions):
                    rollout_actions.append(action.cpu().numpy())

                if prev_done is not None:
                    prev_done_indices = prev_done.nonzero(as_tuple=False)
                    prev_done_count = len(done_indices)
                    if prev_done_count > 0:
                        for prev_done_idx in prev_done_indices:
                            # there are some weird bugs in furniture's states when episodes are too short
                            if len(rollouts[prev_done_idx]) < self.min_episode_length:
                                rollouts[prev_done_idx] = []
                                rollouts_actions[prev_done_idx] = []
                            else:
                                successful = prev_success_buf[prev_done_idx]
                                rollout_to_save = rollouts[prev_done_idx]
                                actions_to_save = np.stack(
                                    rollouts_actions[prev_done_idx][:-1], axis=0
                                )  # :-1 is because the last action is corresponding to the terminal state
                                rollout_to_save = stack_sequence_fields(
                                    rollout_to_save
                                )  # dict of (T + 1, ...)
                                rollout_to_save["actions"] = actions_to_save
                                # discard first step since the data are wrong due to IG's bug
                                rollout_to_save = {
                                    k: v[1:] for k, v in rollout_to_save.items()
                                }
                                if self.save_successful_rollouts_only and successful:
                                    rollout_grp = self.s_grp.create_group(
                                        f"rollout_{self.saved_successful_rollouts}"
                                    )
                                    for k, v in rollout_to_save.items():
                                        rollout_grp.create_dataset(k, data=v)
                                    self.saved_successful_rollouts += 1
                                elif not self.save_successful_rollouts_only:
                                    if successful:
                                        rollout_grp = self.s_grp.create_group(
                                            f"rollout_{self.saved_successful_rollouts}"
                                        )
                                    else:
                                        rollout_grp = self.f_grp.create_group(
                                            f"rollout_{self.saved_failed_rollouts}"
                                        )
                                    for k, v in rollout_to_save.items():
                                        rollout_grp.create_dataset(k, data=v)
                                    if successful:
                                        self.saved_successful_rollouts += 1
                                    else:
                                        self.saved_failed_rollouts += 1
                                rollouts[prev_done_idx] = []
                                rollouts_actions[prev_done_idx] = []
                                if (
                                    self.saved_successful_rollouts
                                    + self.saved_failed_rollouts
                                ) >= self.num_rollouts_to_save:
                                    self.h5py_file.close()
                                    exit()

            obses, r, done, info = self.env_step(self.env, actions)
            prev_done = done.clone()
            prev_success_buf = self.env.success_buf.clone()
            cr += r
            steps += 1

            if render:
                self.env.render(mode="human")
                time.sleep(self.render_sleep)

            all_done_indices = done_indices = done.nonzero(as_tuple=False)
            done_count = len(done_indices)

            if done_count > 0:
                if self.is_rnn:
                    for s in self.states:
                        s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                cur_rewards = cr[done_indices].sum().item()
                cur_steps = steps[done_indices].sum().item()

                cr = cr * (1.0 - done.float())
                steps = steps * (1.0 - done.float())
                sum_rewards += cur_rewards
                sum_steps += cur_steps

                if self.print_stats:
                    cur_rewards_done = cur_rewards / done_count
                    cur_steps_done = cur_steps / done_count
                    print(f"reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f}")

    def get_batch_size(self, obses, batch_size):
        obs_shape = self.obs_shape
        if isinstance(self.obs_shape, gym.spaces.Dict):
            if "obs" in obses:
                obses = obses["obs"]
            keys_view = self.obs_shape.keys()
            keys_iterator = iter(keys_view)
            if "observation" in obses:
                first_key = "observation"
            else:
                first_key = next(keys_iterator)
            obs_shape = self.obs_shape[first_key].shape
            obses = obses[first_key]

        if len(obses.size()) > len(obs_shape):
            batch_size = obses.size()[0]
            self.has_batch_dimension = True

        self.batch_size = batch_size

        return batch_size


class MyPPOPlayerContinuous(MyBasePlayer):
    def __init__(self, *args, **kwargs):
        MyBasePlayer.__init__(self, *args, **kwargs)
        self.network = self.config["network"]
        self.actions_num = self.action_space.shape[0]
        self.actions_low = (
            torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        )
        self.actions_high = (
            torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        )
        self.mask = [False]

        self.normalize_input = self.config["normalize_input"]
        self.normalize_input_excluded_keys = self.config.get(
            "normalize_input_excluded_keys", None
        )
        self.normalize_value = self.config.get("normalize_value", False)

        obs_shape = self.obs_shape
        config = {
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            "num_seqs": self.num_agents,
            "value_size": self.env_info.get("value_size", 1),
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
            "normalize_input_excluded_keys": self.normalize_input_excluded_keys,
        }
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_action(self, obs, is_deterministic=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict["mus"]
        action = res_dict["actions"]
        self.states = res_dict["rnn_states"]
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(
                self.actions_low,
                self.actions_high,
                torch.clamp(current_action, -1.0, 1.0),
            )
        else:
            return current_action

    def restore(self, fn):
        checkpoint = load_checkpoint(fn)
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

        env_state = checkpoint.get("env_state", None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def reset(self):
        self.init_rnn()
