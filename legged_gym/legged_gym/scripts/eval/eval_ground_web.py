"""
Evaluation with Web Visualizer for headless servers.

USAGE:
1. First start meshcat server in a separate terminal:
   meshcat-server

2. Then run this script:
   python eval_web.py --task x02_ground --load_run <run_name>

3. Open browser at: http://<server-ip>:7000/static/
"""
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

# Web Visualizer Setup - MUST be before isaacgym import
from sim_web_visualizer.isaac_visualizer_client import (
    create_isaac_visualizer,
    bind_visualizer_to_gym,
    set_gpu_pipeline
)

# Connect to already running meshcat-server
# ZMQ port is 6000, Web port is 7000
WEB_VIZ_PORT = 6000  # ZMQ port, not web port!
WEB_VIZ_HOST = "127.0.0.1"
MAX_ENV_DISPLAY = 1  # Nur 1 Env für bessere Performance

create_isaac_visualizer(
    port=WEB_VIZ_PORT,
    host=WEB_VIZ_HOST,
    keep_default_viewer=False,  # False for headless servers - use MimicViewer instead
    max_env=MAX_ENV_DISPLAY
)

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.envs.base.host_ground import LeggedRobot as LeggedRobotGround
from legged_gym.envs.base.host_ground_prone import LeggedRobot as LeggedRobotProne
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Process, Value
import torch
from tqdm import tqdm

# Patch create_sim to bind web visualizer BETWEEN sim creation and env creation
# The key insight: we need to bind AFTER gym.create_sim() but BEFORE _create_envs()

def _patched_create_sim(self):
    """Patched create_sim that binds web visualizer at the right moment."""
    # Step 1: Create sim (same as original)
    self.up_axis_idx = 2
    self.sim = self.gym.create_sim(
        self.sim_device_id, self.graphics_device_id,
        self.physics_engine, self.sim_params
    )

    # Step 2: BIND VISUALIZER HERE - after sim exists, before envs are created
    self.gym = bind_visualizer_to_gym(self.gym, self.sim)
    set_gpu_pipeline(self.sim_params.use_gpu_pipeline)
    print("Web visualizer bound to gym!")

    # Step 3: Create terrain (same as original)
    mesh_type = self.cfg.terrain.mesh_type
    if mesh_type in ['heightfield', 'trimesh']:
        from legged_gym.utils.terrain import Terrain
        self.terrain = Terrain(self.cfg.terrain, self.num_envs)
    if mesh_type == 'plane':
        self._create_ground_plane()
    elif mesh_type == 'heightfield':
        self._create_heightfield()
    elif mesh_type == 'trimesh':
        self._create_trimesh()
    elif mesh_type is not None:
        raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

    # Step 4: Create environments - now with bound visualizer
    self._create_envs()

# Patch both environment classes for web visualizer
LeggedRobotGround.create_sim = _patched_create_sim
LeggedRobotProne.create_sim = _patched_create_sim

# Also patch the MimicViewer to handle get_viewer_camera_transform
from sim_web_visualizer.isaac_visualizer_client import MimicViewer, _REGISTERED_VISUALIZER

def _add_camera_transform_override():
    """Add override for get_viewer_camera_transform to handle MimicViewer."""
    if len(_REGISTERED_VISUALIZER) == 0:
        print("Warning: No visualizer registered yet")
        return
    viz = _REGISTERED_VISUALIZER[0]

    def get_viewer_camera_transform(viewer, env):
        if isinstance(viewer, MimicViewer):
            # Return a dummy transform for MimicViewer
            transform = gymapi.Transform()
            transform.p = gymapi.Vec3(2.0, 2.0, 2.0)
            transform.r = gymapi.Quat(0, 0, 0, 1)
            return transform
        else:
            return viz.original_gym.get_viewer_camera_transform(viewer, env)

    viz.new_gym.add_method("get_viewer_camera_transform", get_viewer_camera_transform)

# This will be called after visualizer is bound
_camera_override_added = False

def _ensure_camera_override(env):
    global _camera_override_added
    if not _camera_override_added:
        _add_camera_transform_override()
        _camera_override_added = True


class EvalLogger:
    def __init__(self, env, num_episodes, device='cuda:0'):
        self.env = env
        self.num_episodes = num_episodes
        self.device = device
        self.curr_episode = 0

        self.reset_buffer()

        self.success_buffer = torch.zeros((env.num_envs, num_episodes), dtype=torch.bool)
        self.feet_movement = torch.zeros((env.num_envs, num_episodes), dtype=torch.float, device=self.device)
        self.smoothness = torch.zeros((env.num_envs, num_episodes), dtype=torch.float, device=self.device)
        self.motion_tracking_error = torch.zeros((env.num_envs, num_episodes), dtype=torch.float, device=self.device)
        self.power_all = torch.zeros((env.num_envs, num_episodes), dtype=torch.float, device=self.device)
        self.smoothness_before_standingup = torch.zeros((env.num_envs, num_episodes), dtype=torch.float, device=self.device)

    def log(self):
        new_base_xy = self.env.root_states[:, :2].clone()
        base_xy_movement = torch.sum(torch.square(new_base_xy - self.base_xy), dim=-1)
        self.base_xy[:] = new_base_xy.clone()
        self.base_xy_movement[:] += base_xy_movement.clone() * self.ever_standingup

        new_left_feet_xyz = self.env.rigid_body_states[:, self.env.left_foot_indices, :3].clone().squeeze(1) * 100
        new_right_feet_xyz = self.env.rigid_body_states[:, self.env.right_foot_indices, :3].clone().squeeze(1) * 100
        feet_xyz_movement = torch.sum(torch.square(new_left_feet_xyz - self.left_feet_xyz), dim=-1) + torch.sum(torch.square(new_right_feet_xyz - self.right_feet_xyz), dim=-1)
        self.left_feet_xyz[:] = new_left_feet_xyz.clone()
        self.right_feet_xyz[:] = new_right_feet_xyz.clone()
        self.feet_xyz_movement[:] += feet_xyz_movement.clone() * self.ever_standingup

        self.action_smooth[:] = torch.sum(torch.square(self.env.dof_pos - self.env.last_dof_pos - self.env.last_dof_pos + self.env.last_last_dof_pos), dim=1)
        self.smoothness[:, self.curr_episode] += self.action_smooth.clone()

        pose_mse = torch.sum(torch.square(self.env.dof_pos[:, self.env.upper_body_joint_indices] - self.env.target_dof_pos[:, self.env.upper_body_joint_indices]), dim=-1)
        self.motion_tracking_error[:, self.curr_episode] += pose_mse.clone() * self.ever_standingup

        self.power[:] += torch.sum(torch.abs(self.env.dof_vel) * torch.abs(self.env.torques), dim=-1) * 0.02 * ~self.ever_standingup
        self.smoothness_before_standingup[:, self.curr_episode] += self.action_smooth.clone() * ~self.ever_standingup

        self.base_height_buffer[:] = self.env.root_states[:, 2].clone()
        self.ever_standingup[:] = self.ever_standingup | (self.base_height_buffer > 0.7)
        self.falldown_after_standup[:] = self.falldown_after_standup | (self.ever_standingup & (self.base_height_buffer < 0.5))
        self.standingup_times += self.ever_standingup
        self.action_times += self.env.real_episode_length_buf > self.env.unactuated_time
        self.before_standingup_times[:] += ~self.ever_standingup.clone()

    def compute_metric(self):
        self.success_buffer[:, self.curr_episode] = self.ever_standingup.clone() & ~self.falldown_after_standup.clone()
        self.feet_movement[:, self.curr_episode] = self.feet_xyz_movement.clone() / (self.standingup_times + 1)
        self.smoothness[:, self.curr_episode] /= self.action_times
        self.motion_tracking_error[:, self.curr_episode] /= self.standingup_times + 1
        self.power_all[:, self.curr_episode] = self.power.clone()
        self.smoothness_before_standingup[:, self.curr_episode] /= self.before_standingup_times

    def reset_buffer(self):
        self.base_height_buffer = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.ever_standingup = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.falldown_after_standup = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.standingup_times = torch.zeros(self.env.num_envs, dtype=torch.int, device=self.device)
        self.action_times = torch.zeros(self.env.num_envs, dtype=torch.int, device=self.device)
        self.before_standingup_times = torch.zeros(self.env.num_envs, dtype=torch.int, device=self.device)

        self.base_xy = torch.zeros((self.env.num_envs, 2), dtype=torch.float, device=self.device)
        self.base_xy_movement = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.left_feet_xyz = torch.zeros((self.env.num_envs, 3), dtype=torch.float, device=self.device)
        self.right_feet_xyz = torch.zeros((self.env.num_envs, 3), dtype=torch.float, device=self.device)
        self.feet_xyz_movement = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.action_smooth = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.power = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    def reset(self):
        print('before_standingup_times', self.before_standingup_times.float().mean())
        self.compute_metric()
        self.reset_buffer()
        self.curr_episode += 1

    def plot_terminal_metrics(self):
        print('Success rate: ', torch.mean(self.success_buffer.float() * 100).item(), torch.std(torch.mean(self.success_buffer.float() * 100, dim=0)).item())
        print('Average feet movement: ', torch.mean(self.feet_movement * (self.feet_movement > 0)).item(), torch.std(torch.mean(self.feet_movement * (self.feet_movement > 0), dim=0)).item())
        print('Average smoothness: ', torch.mean(self.smoothness * 180 / np.pi).item(), torch.std(torch.mean(self.smoothness * 180 / np.pi, dim=0)).item())
        print('Average motion tracking error: ', torch.mean(self.motion_tracking_error).item(), torch.std(torch.mean(self.motion_tracking_error, dim=0)).item())
        print('Average times before standing up: ', torch.mean(self.before_standingup_times.float()).item(), torch.std(self.before_standingup_times.float()).item())
        print('Average energy: ', torch.mean(self.power_all).item(), torch.std(torch.mean(self.power_all, dim=0)).item())
        print('Average smoothness before standing up: ', torch.mean(self.smoothness_before_standingup * 180 / np.pi).item(), torch.std(torch.mean(self.smoothness_before_standingup * 180 / np.pi, dim=0)).item())


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Reduce envs for visualization
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)  # Nur 1 Env für Visualisierung
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.env.episode_length_s = 5
    env_cfg.control.action_scale = 0.25
    env_cfg.curriculum.pull_force = False
    env_cfg.env.test = True

    # IMPORTANT: headless must be False for web visualizer
    args.headless = False

    # Create environment (web visualizer is bound automatically via patched create_sim)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Move env to center of terrain
    env.env_origins[0, :2] = 10.0

    # Add camera transform override for MimicViewer
    _ensure_camera_override(env)

    print(f"\n{'='*50}")
    print(f"Web Visualizer: http://<server-ip>:7000/static/")
    print(f"{'='*50}\n")

    obs = env.get_observations()

    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, env_cfg=env_cfg, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    num_episodes = 5
    evalogger = EvalLogger(env, num_episodes)
    evalogger.log()

    print("Starting evaluation. Press Ctrl+C to stop early.")

    try:
        for i in tqdm(range(num_episodes)):
            for j in range(int(env.max_episode_length + 1)):
                actions = policy(obs.detach())
                obs, _, rews, dones, infos = env.step(actions.detach())
                if j != int(env.max_episode_length):
                    evalogger.log()
            evalogger.reset()
        evalogger.plot_terminal_metrics()
    except KeyboardInterrupt:
        print("\nEvaluation stopped early.")
        evalogger.plot_terminal_metrics()


if __name__ == '__main__':
    args = get_args()
    play(args)