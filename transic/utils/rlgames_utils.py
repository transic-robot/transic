# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from collections import deque
from typing import Callable, Dict, Tuple, Any

import os
import gym
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from transic_envs import TASK_MAP

from transic.utils.utils import set_seed, flatten_dict
from transic.utils.wandb_utils import WandbAlgoObserver


def multi_gpu_get_rank(multi_gpu):
    if multi_gpu:
        rank = int(os.getenv("LOCAL_RANK", "0"))
        print("GPU rank: ", rank)
        return rank

    return 0


def get_rlgames_env_creator(
    # used to create the vec task
    task_config: dict,
    task_name: str,
    sim_device: str,
    rl_device: str,
    graphics_device_id: int,
    display: bool = False,
    record: bool = False,
    has_headless_arg: bool = False,
    headless: bool = True,
    # used to handle multi-gpu case
    multi_gpu: bool = False,
    post_create_hook: Callable = None,
):
    """Parses the configuration parameters for the environment task and creates a VecTask

    Args:
        task_config: environment configuration.
        task_name: Name of the task, used to evaluate based on the imported name (eg 'Trifinger')
        sim_device: The type of env device, eg 'cuda:0'
        rl_device: Device that RL will be done on, eg 'cuda:0'
        graphics_device_id: Graphics device ID.
        multi_gpu: Whether to use multi gpu
        post_create_hook: Hooks to be called after environment creation.
            [Needed to setup WandB only for one of the RL Games instances when doing multiple GPUs]
    Returns:
        A VecTaskPython object.
    """

    def create_rlgpu_env():
        """
        Creates the task from configurations and wraps it using RL-games wrappers if required.
        """
        if multi_gpu:
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            global_rank = int(os.getenv("RANK", "0"))

            # local rank of the GPU in a node
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            world_size = int(os.getenv("WORLD_SIZE", "1"))

            print(
                f"global_rank = {global_rank} local_rank = {local_rank} world_size = {world_size}"
            )

            _sim_device = f"cuda:{local_rank}"
            _rl_device = f"cuda:{local_rank}"

            task_config["rank"] = local_rank
            task_config["rl_device"] = _rl_device
        else:
            _sim_device = sim_device
            _rl_device = rl_device

        # create native task and pass custom config
        kwargs = dict(
            cfg=task_config,
            rl_device=_rl_device,
            sim_device=_sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
        )
        if has_headless_arg:
            kwargs["headless"] = headless
        env = TASK_MAP[task_name](**kwargs)

        if post_create_hook is not None:
            post_create_hook()

        return env

    return create_rlgpu_env


class RLGPUAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats."""

    def __init__(self):
        super().__init__()
        self.algo = None
        self.writer = None

        self.ep_infos = []
        self.direct_info = {}

        self.episode_cumulative = dict()
        self.episode_cumulative_avg = dict()
        self.new_finished_episodes = False

    def after_init(self, algo):
        self.algo = algo
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        assert isinstance(infos, dict), "RLGPUAlgoObserver expects dict info"
        if not isinstance(infos, dict):
            return

        if "episode" in infos:
            self.ep_infos.append(infos["episode"])

        if "episode_cumulative" in infos:
            for key, value in infos["episode_cumulative"].items():
                if key not in self.episode_cumulative:
                    self.episode_cumulative[key] = torch.zeros_like(value)
                self.episode_cumulative[key] += value

            for done_idx in done_indices:
                self.new_finished_episodes = True
                done_idx = done_idx.item()

                for key, value in infos["episode_cumulative"].items():
                    if key not in self.episode_cumulative_avg:
                        self.episode_cumulative_avg[key] = deque(
                            [], maxlen=self.algo.games_to_track
                        )

                    self.episode_cumulative_avg[key].append(
                        self.episode_cumulative[key][done_idx].item()
                    )
                    self.episode_cumulative[key][done_idx] = 0

        # turn nested infos into summary keys (i.e. infos['scalars']['lr'] -> infos['scalars/lr']
        if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
            infos_flat = flatten_dict(infos, prefix="", separator="/")
            self.direct_info = {}
            for k, v in infos_flat.items():
                # only log scalars
                if (
                    isinstance(v, float)
                    or isinstance(v, int)
                    or (isinstance(v, torch.Tensor) and len(v.shape) == 0)
                ):
                    self.direct_info[k] = v

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.algo.device))
                    )
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, epoch_num)
            self.ep_infos.clear()

        # log these if and only if we have new finished episodes
        if self.new_finished_episodes:
            for key in self.episode_cumulative_avg:
                self.writer.add_scalar(
                    f"episode_cumulative/{key}",
                    np.mean(self.episode_cumulative_avg[key]),
                    frame,
                )
                self.writer.add_scalar(
                    f"episode_cumulative_min/{key}_min",
                    np.min(self.episode_cumulative_avg[key]),
                    frame,
                )
                self.writer.add_scalar(
                    f"episode_cumulative_max/{key}_max",
                    np.max(self.episode_cumulative_avg[key]),
                    frame,
                )
            self.new_finished_episodes = False

        for k, v in self.direct_info.items():
            self.writer.add_scalar(f"{k}/frame", v, frame)
            self.writer.add_scalar(f"{k}/iter", v, epoch_num)
            self.writer.add_scalar(f"{k}/time", v, total_time)


class MultiObserver(AlgoObserver):
    """Meta-observer that allows the user to add several observers."""

    def __init__(self, observers_):
        super().__init__()
        self.observers = observers_

    @property
    def wandb_observer(self):
        for observer in self.observers:
            if isinstance(observer, WandbAlgoObserver):
                return observer
        return None

    def _call_multi(self, method, *args_, **kwargs_):
        for o in self.observers:
            getattr(o, method)(*args_, **kwargs_)

    def before_init(self, base_name, config, experiment_name):
        self._call_multi("before_init", base_name, config, experiment_name)

    def after_init(self, algo):
        self._call_multi("after_init", algo)

    def process_infos(self, infos, done_indices):
        self._call_multi("process_infos", infos, done_indices)

    def after_steps(self):
        self._call_multi("after_steps")

    def after_clear_stats(self):
        self._call_multi("after_clear_stats")

    def after_print_stats(self, frame, epoch_num, total_time):
        self._call_multi("after_print_stats", frame, epoch_num, total_time)


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors):
        self.env = env_configurations.configurations[config_name]["env_creator"]()

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = self.env.action_space
        info["observation_space"] = self.env.observation_space

        if hasattr(self.env, "amp_observation_space"):
            info["amp_observation_space"] = self.env.amp_observation_space

        if self.env.num_states > 0:
            info["state_space"] = self.env.state_space
            print(info["action_space"], info["observation_space"], info["state_space"])
        else:
            print(info["action_space"], info["observation_space"])

        return info

    def set_train_info(self, env_frames, *args_, **kwargs_):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        if hasattr(self.env, "set_train_info"):
            self.env.set_train_info(env_frames, *args_, **kwargs_)

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        if hasattr(self.env, "get_env_state"):
            return self.env.get_env_state()
        else:
            return None

    def set_env_state(self, env_state):
        if hasattr(self.env, "set_env_state"):
            self.env.set_env_state(env_state)


class ComplexObsRLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name):
        """RLGPU wrapper for Isaac Gym tasks.

        Args:
            config_name: Name of rl games env_configurations configuration to use.
            obs_spec: Dictinoary listing out specification for observations to use.
                eg.
                {
                 'obs': {'names': ['obs_1', 'obs_2'], 'concat': True, space_name: 'observation_space'},},
                 'states': {'names': ['state_1', 'state_2'], 'concat': False, space_name: 'state_space'},}
                }
                Within each, if 'concat' is set, concatenates all the given observaitons into a single tensor of dim (num_envs, sum(num_obs)).
                    Assumes that each indivdual observation is single dimensional (ie (num_envs, k), so image observation isn't supported).
                    Currently applies to student and teacher both.
                "space_name" is given into the env info which RL Games reads to find the space shape
        """
        self.env = env_configurations.configurations[config_name]["env_creator"]()

    def step(
        self, action: torch.Tensor
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, Dict[str, Any]
    ]:
        """Step the Isaac Gym task.

        Args:
            action: Enivronment action.
        Returns:
            observations, rewards, dones, infos
            Returned obeservations are a dict which contains key 'obs' corresponding to a dictionary of observations,
            and possible 'states' key corresponding to dictionary of privileged observations.
        """
        env_obs, rewards, dones, infos = self.env.step(action)
        return env_obs, rewards, dones, infos

    def reset(self) -> Dict[str, Dict[str, torch.Tensor]]:
        env_obs = self.env.reset()
        return env_obs

    def get_number_of_agents(self) -> int:
        return self.env.get_number_of_agents()

    def get_env_info(self) -> Dict[str, gym.spaces.Space]:
        """Gets information on the environment's observation, action, and privileged observation (states) spaces."""
        info = {}
        info["action_space"] = self.env.action_space
        info["observation_space"] = self.env.observation_space
        return info

    def set_train_info(self, env_frames, *args_, **kwargs_):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        if hasattr(self.env, "set_train_info"):
            self.env.set_train_info(env_frames, *args_, **kwargs_)

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        if hasattr(self.env, "get_env_state"):
            return self.env.get_env_state()
        else:
            return None

    def set_env_state(self, env_state):
        if hasattr(self.env, "set_env_state"):
            self.env.set_env_state(env_state)

    @property
    def success_buf(self):
        return self.env.success_buf

    @property
    def failure_buf(self):
        return self.env.failure_buf
