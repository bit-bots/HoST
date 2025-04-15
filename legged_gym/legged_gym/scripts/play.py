import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True


    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = 0.1
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, env_cfg=env_cfg, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)


    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 6 # which joint is used for logging

    stop_state_log = 200 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards

    obs_list = []
    act_list = []

    root_state = []
    dof_pos = []


    for i in range(10*int(env.max_episode_length)):

        result = env.gym.fetch_results(env.sim, True)
        env.commands[:, 0] = 0.1
        # if env.real_episode_length_buf[1] == env.unactuated_time:
        #     obs[1, 375+3] = 0
        # obs[:, 375+3] = 0
        actions = policy(obs.detach())
        # print(actions.max())
        obs, _, rews, dones, infos = env.step(actions.detach())
        # print(len(obs_list))

        if env.real_episode_length_buf[0] >= env.unactuated_time:
            obs_list.append(obs[0].detach().cpu().numpy())
            act_list.append(actions[0].detach().cpu().numpy())
            root_state.append(env.rigid_body_states[0].detach().cpu().numpy())
            dof_pos.append(env.dof_pos[:].detach().cpu().numpy())

        # if i < stop_state_log:
        #     # import ipdb; ipdb.set_trace()
        #     # print(env.contact_forces.shape)
        #     # print(env.contact_forces[robot_index, env.feet_indices, 2].cpu().max().numpy(), env.contact_forces[robot_index, env.hip_joint_indices, 2].cpu().numpy())
        #     logger.log_states(
        #         {
        #             'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.dof_pos[0, joint_index].item(),
        #             'dof_pos': env.dof_pos[robot_index, joint_index].item(),
        #             'dof_vel': env.dof_vel[robot_index, joint_index].item(),
        #             'dof_torque': env.torques[robot_index, env.knee_joint_indices[0]].item(),
        #             'dof_torque_2': env.torques[robot_index, env.knee_joint_indices[1]].item(),
        #             'command_x': env.commands[robot_index, 0].item(),
        #             'command_y': env.commands[robot_index, 0].item(),
        #             'command_yaw': env.commands[robot_index, 0].item(),
        #             'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
        #             'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
        #             'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
        #             'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
        #             'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
        #             # 'power': env.power[robot_index, env.knee_joint_indices[0]].cpu().numpy(),
        #             #  'power_right': env.power[robot_index, env.knee_joint_indices[1]].cpu().numpy()
        #         }
        #     )

        # elif i==stop_state_log:
        #     logger.plot_states()

        # if len(obs_list) == 250:
        #     np.save(f'/home/PJLAB/huangtao/Documents/project/humanoid/g1-unitree/legged_gym/data/{train_cfg.runner.experiment_name}_obs.npy', np.array(obs_list), allow_pickle=True)
        #     np.save(f'/home/PJLAB/huangtao/Documents/project/humanoid/g1-unitree/legged_gym/data/{train_cfg.runner.experiment_name}_act.npy', np.array(act_list), allow_pickle=True)
        #     # np.save(f'/home/PJLAB/huangtao/Documents/project/humanoid/g1-unitree/legged_gym/data/{train_cfg.runner.experiment_name}_root_state.npy', np.array(root_state), allow_pickle=True)
        #     # np.save(f'/home/PJLAB/huangtao/Documents/project/humanoid/g1-unitree/legged_gym/data/{train_cfg.runner.experiment_name}_dof_pos.npy', np.array(dof_pos), allow_pickle=True)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
