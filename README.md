# TRANSIC: Sim-to-Real Policy Transfer by Learning from Online Correction
<div align="center">

[Yunfan Jiang](https://yunfanj.com/),
[Chen Wang](https://www.chenwangjeremy.net/),
[Ruohan Zhang](https://ai.stanford.edu/~zharu/),
[Jiajun Wu](https://jiajunwu.com/),
[Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li)

<img src="media/SUSig-red.png" width=200>

**Conference on Robot Learning (CoRL) 2024**

[[Website]](https://transic-robot.github.io/)
[[arXiv]](https://arxiv.org/abs/2405.10315)
[[PDF]](https://transic-robot.github.io/assets/pdf/transic_paper.pdf)
[[TRANSIC-Envs]](https://github.com/transic-robot/transic-envs)
[[Model Weights]](https://huggingface.co/transic-robot/models)
[[Training Data]](https://huggingface.co/datasets/transic-robot/data)
[[Model Card]](https://huggingface.co/transic-robot/models/blob/main/README.md)
[[Data Card]](https://huggingface.co/datasets/transic-robot/data/blob/main/README.md)

[![Python Version](https://img.shields.io/badge/Python-3.8-blue.svg)](https://github.com/transic-robot/transic)
[<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg"/>](https://pytorch.org/)
[![GitHub license](https://img.shields.io/github/license/transic-robot/transic)](https://github.com/transic-robot/transic/blob/main/LICENSE)

![](media/method_overview.gif)
______________________________________________________________________
</div>

Learning in simulation and transferring the learned policy to the real world has the potential to enable generalist robots. The key challenge of this approach is to address simulation-to-reality (sim-to-real) gaps. Previous methods often require domain-specific knowledge *a priori*. We argue that a straightforward way to obtain such knowledge is by asking humans to observe and assist robot policy execution in the real world. The robots can then learn from humans to close various sim-to-real gaps. We propose **TRANSIC**, a data-driven approach to enable successful sim-to-real transfer based on a human-in-the-loop framework. **TRANSIC** allows humans to augment simulation policies to overcome various unmodeled sim-to-real gaps holistically through intervention and online correction. Residual policies can be learned from human corrections and integrated with simulation policies for autonomous execution. We show that our approach can achieve successful sim-to-real transfer in complex and contact-rich manipulation tasks such as furniture assembly. Through synergistic integration of policies learned in simulation and from humans, **TRANSIC** is effective as a holistic approach to addressing various, often coexisting sim-to-real gaps. It displays attractive properties such as scaling with human effort.

## Table of Contents
1. [Installation](#Installation)
2. [Usage](#usage)
3. [Acknowledgement](#acknowledgement)
4. [Check out Our Paper](#check-out-our-paper)
5. [License](#license)

## Installation
First follow the [instruction](https://github.com/transic-robot/transic-envs/#Installation) to create a virtual environment, install IsaacGym, and install our simulation codebase [TRANSIC-Envs](https://github.com/transic-robot/transic-envs).

Now clone this repo and install it.
```bash
git clone https://github.com/transic-robot/transic
cd transic
pip3 install -e .
```

Optionally, if you would like to use our [model checkpoints](https://huggingface.co/transic-robot/models) and [training data](https://huggingface.co/datasets/transic-robot/data), download them from 🤗Hugging Face.
```bash
git clone https://huggingface.co/transic-robot/models transic-models
git clone https://huggingface.co/datasets/transic-robot/data transic-data
```

## Usage
### Training Teacher Policies
The basic syntax to launch teacher policy RL training is
```bash
python3 main/rl/train.py task=<task_name> num_envs=<num_of_parallel_envs> \
  sim_device=cuda:<gpu_id> rl_device=cuda:<gpu_id> graphics_device_id=<gpu_id>
```
You need to replace anything within `<>` with suitable values. For example, you can select `task_name` with one from [here](https://github.com/transic-robot/transic-envs/#overview).

> [!TIP]
> You may need to tune the number of parallel envs `num_envs=<num_of_parallel_envs>` depending on your GPU memory to achieve the maximum throughput.

> [!TIP]
> You may use [wandb](https://wandb.ai/site) to log experiments. To do this, add `wandb_activate=true` to the command and specify your wandb username and project name through `wandb_entity=<your_wandb_user_name> wandb_project=<your_wandb_project_name>`.

The training command will create a folder called `runs/{experiment_name}` under the current directory, where you can find the training config and saved checkpoints.

To test a checkpoint, run the following command.
```bash
python3 main/rl/train.py task=<task_name> num_envs=<num_of_parallel_envs> \
  test=true checkpoint=<path_to_your_checkpoint>
```

> [!TIP]
> To visualize a trained policy, use either `display=true` or `headless=false`. The first option will pop up an OpenCV window showing the env-level workspace from a frontal view. This doesn't require a physical monitor attached. The second option will open the IsaacGym GUI and you will see all parallel environments. This REQUIRES a physical monitor connected to your workstation.

> [!TIP]
> You can also log policy rollouts as mp4 videos to your wandb. Simply add `capture_video=true` to the test command.

### Training Student Policies
#### Prepare the Training Data
We use trained teacher policies to generate data for student policies. To do so, simply run the following command.
```bash
python3 main/rl/train.py task=<task_name> num_envs=<num_of_parallel_envs> \
  test=true checkpoint=<path_to_your_checkpoint> \
  save_rollouts=true
```
Rollouts will be saved in `runs/{experiment_name}/rollouts.hdf5`.

> [!TIP]
> By default, this will generate 10K successful trajectories. Each trajectory will have a minium length of 20 steps. You can change these behaviors by setting `save_successful_rollouts_only`, `num_rollouts_to_save`, and `min_episode_length`.

We provide weights for trained RL teachers. To use them, replace `checkpoint` with the suitable path. For example,
```bash
python3 main/rl/train.py task=Stabilize \
  test=true checkpoint=<path_to_transic-models/rl/stabilize.pth> \
  save_rollouts=true
```

We also provide pre-generated data for student distillation. They can be found in the `distillation` folder from our [🤗Hugging Face data repository](https://huggingface.co/datasets/transic-robot/data).

#### Start Training
The basic syntax to launch student policy distillation is
```bash
python3 main/distillation/train.py task=<task_name> distillation_student_arch=<arch> \
  bs=<batch_size> num_envs=<num_of_parallel_envs> exp_root_dir=<where_to_log_experiment> \
  data_path=<path_to_hdf5_file> matched_scene_data_path=<path_to_matched_scene_data> \
  sim_device=cuda:<gpu_id> rl_device=cuda:<gpu_id> graphics_device_id=<gpu_id> gpus=\[<gpus>\] \
  wandb_project=<your_wandb_project_name>
```
Similarly, you need to replace anything within `<>` with suitable values. For example, you can select `task_name` with one from [here](https://github.com/transic-robot/transic-envs/#overview). But make sure they have the `PCD` suffix since you are training student policies with visual observations. You can select either `pointnet` or `rnn_pointnet` for policy architecture. You may need to tune the batch size `bs` and number of parallel environments `num_envs` to fit into your GPU. The `exp_root_dir` specifies where you would like to log the experiment. The `data_path` is where your generated rollouts are saved. The `matched_scene_data_path` is a static and fixed dataset we used to regularize the point cloud encoder. It can be found as `distillation/matched_point_cloud_scesim_device=cuda:nes.h5` from our [🤗Hugging Face data repository](https://huggingface.co/datasets/transic-robot/data).

> [!WARNING]
> By default we add data randomization during the distillation. You may opt to set `module.enable_pcd_augmentation=false` to turn off point cloud augmentation and `module.enable_prop_augmentation=false` to turn off proprioception augmentation. But this will lead to suboptimal student policies that are not robust enough for sim-to-real transfer.

> [!TIP]
> The argument `gpus` specifies the devices to use for distillation and follows the same [syntax](https://lightning.ai/docs/pytorch/stable/common/trainer.html#devices) as in PyTorch Lightning. Other device-related arguments such as `sim_device`, `rl_device`, and `graphics_device` control which GPU should IsaacGym use. GPUs for distillation and simulation do not need to be the same. Actually, we also support multi-GPU distillation with IsaacGym running on another GPU for evaluation.

The experiment will be logged at `exp_root_dir`, where you can find the saved config, logs, tensorboard, and checkpoints. Since we periodically switch between training and simulation evaluation. Policies are saved based on their success rates. You can find weights of our student policies in the folder `student` from our [🤗Hugging Face model repository](https://huggingface.co/transic-robot/models).

To test and visualize trained student policies, run the following command.
```bash
python3 main/distillation/test.py task=<task_name> distillation_student_arch=<arch> \
  bs=null num_envs=<num_of_parallel_envs> exp_root_dir=<where_to_log_experiment> \
  data_path=null matched_scene_data_path=null \
  test.ckpt_path=<path_to_student_policy> display=true
```

### Correction Data Collection
Once we have the simulation base policy, we deploy it on a real robot while a human operator monitors its execution. The human operator intervenes the policy execution when necessary and provides correction through teleoperation. To collect such correction data, checkout the script
```bash
python3 main/correction_data_collection.py \
  --base-policy-ckpt-path <path_to_simulation_base_policy_ckpt> \
  --data-save-path <where_to_save_correction_data>
```
We notice that the real-world observation pipeline and real robot controller may differ across different groups. Therefore, you have to fill in the instantiation of these two components in the script. In our case, we use [`deoxys`](https://github.com/UT-Austin-RPL/deoxys_control) as our robot controller. We provide an example of observation pipeline [here](transic/real_world/obs.py).

We provide correction data we collected during the project in the `correction_data` folder from our [🤗Hugging Face data repository](https://huggingface.co/datasets/transic-robot/data).

### Training Residual Policies
Once we have enough correction data, we can train residual policies with two steps. First, we only learn the residual action head.
```bash
python3 main/residual/train.py residual_policy_arch=<arch> \
  data_dir=<correction_data_path> exp_root_dir=<where_to_log_experiment> \
  residual_policy_task=<task> \
  gpus=<gpus> bs=<batch_size> \
  module.intervention_pred_loss_weight=0.0 \
  wandb_project=<your_wandb_project_name>
```
For `residual_policy_task`, use `insert` for the task Insert and `default` for others.

We then freeze everything and only learn the head to predict intervention or not.
```bash
python3 main/residual/train.py residual_policy_arch=<arch> \
  data_dir=<correction_data_path> exp_root_dir=<where_to_log_experiment> \
  residual_policy_task=<task> \
  gpus=<gpus> bs=<batch_size> \
  module.residual_policy.update_intervention_head_only=True \
  module.residual_policy.ckpt_path_if_update_intervention_head_only=<path_to_ckpt_from_the_first_step>
  wandb_project=<your_wandb_project_name>
```

> [!NOTE]
> Residual policies also can be trained in a single step where both the action and intervention prediction heads are jointly learned. We found that the two-step method leads to overall better residual policies.

Our trained residual policies can be found in the folder `residual` from our [🤗Hugging Face model repository](https://huggingface.co/transic-robot/models).

### Integrated Deployment
Once we have both the simulation base policy and the residual policy, we can integrate them together for successful sim-to-real transfer. Checkout the script
```bash
python3 main/integrated_deployment.py \
  --base-policy-ckpt-path <path_to_simulation_base_policy_ckpt> \
  --residual-policy-ckpt-path <path_to_residual_policy_ckpt>
```
Similarly, you need to fill in the instantiation for real-world observation pipeline and the real-robot controller.

## Acknowledgement
We would like to acknowledge the following open-source project that greatly inspired our development.
- [deoxys](https://github.com/UT-Austin-RPL/deoxys_control)

## Check out Our Paper
Our paper is posted on [arXiv](https://arxiv.org/abs/2405.10315). If you find our work useful, please consider citing us! 

```bibtex
@inproceedings{jiang2024transic,
  title     = {TRANSIC: Sim-to-Real Policy Transfer by Learning from Online Correction},
  author    = {Yunfan Jiang and Chen Wang and Ruohan Zhang and Jiajun Wu and Li Fei-Fei},
  booktitle = {Conference on Robot Learning},
  year      = {2024}
}
```

## License
This codebase is released under the [MIT License](LICENSE).
