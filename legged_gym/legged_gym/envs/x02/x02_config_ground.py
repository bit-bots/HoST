# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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

# Copyright (c) 2024,Shanghai Droid Robot CO.,LTD. All rights reserved.


from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class x02Cfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_one_step_observations= 64
        num_actions = 19
        num_dofs = 19
        num_actor_history = 6
        num_observations = num_actor_history * num_one_step_observations
        episode_length_s = 10 # episode length in seconds
        unactuated_timesteps = 30

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x02_2/x2.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/X02Lite/X02Lite.xml' das ist der alte pfad
        name = "x02"
        
        ################################################ ab hier kopiert aus g1
        left_foot_name = 'left_ankle'
        right_foot_name = 'right_ankle'
        left_knee_name = 'left_knee'
        right_knee_name = 'right_knee'
        left_thigh_name = 'left_hip_pitch'
        right_thigh_name = 'right_hip_pitch'
        foot_name = "ankle"
        penalize_contacts_on = ["elbow", 'shoulder', 'torso', 'knee', 'hip']
        terminate_after_contacts_on = []    #'torse'
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

        left_shoulder_name = 'left_shoulder'
        right_shoulder_name = 'right_shoulder'

        ####### -- bereits geschafft bis hier 
        ###### Die Joints in der Urdf Joints nennen weil Substrings genommen werden
        left_leg_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_pitch_joint', 'left_ankle_pitch_joint']
        right_leg_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_pitch_joint', 'right_ankle_pitch_joint']
        left_hip_joints = ['left_hip_yaw_joint']
        right_hip_joints = ['right_hip_yaw_joint']

        left_hip_roll_joints = ['left_hip_roll_joint']
        right_hip_roll_joints = ['right_hip_roll_joint']    

        left_hip_pitch_joints = ['left_hip_pitch_joint']
        right_hip_pitch_joints = ['right_hip_pitch_joint']    

        left_shoulder_roll_joints = ['left_shoulder_roll_joint']
        right_shoulder_roll_joints = ['right_shoulder_roll_joint']    


        left_knee_joints = ['left_knee_pitch_joint']
        right_knee_joints = ['right_knee_pitch_joint']    

        left_arm_joints = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint']
        right_arm_joints = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']
        waist_joints = ["torso_joint"]
        knee_joints = ['left_knee_pitch_joint', 'right_knee_pitch_joint']
        ankle_joints = ['left_ankle_pitch_joint', 'right_ankle_pitch_joint']


        trunk_names = ["pelvis", "torso"]
        base_name = 'torso_link' 

        keyframe_name = "keyframe"
        head_name = 'keyframe_head'
        armature = 0


        left_upper_body_names = ['left_shoulder_pitch', 'left_elbow']
        right_upper_body_names = ['right_shoulder_pitch', 'right_elbow']
        left_lower_body_names = ['left_hip_pitch', 'left_ankle', 'left_knee']
        right_lower_body_names = ['right_hip_pitch', 'right_ankle', 'right_knee']

        left_ankle_names = ['left_ankle']
        right_ankle_names = ['right_ankle']

        density = 0.001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01
        thickness = 0.01
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 0.8
        dynamic_friction = 0.7
        restitution = 0.3
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 1   # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [1, 0., 0, 0, 0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    #class noise:
        # add_noise = True
        # noise_level = 1.    # scales other values

        # class noise_scales:
        #     dof_pos = 0.05
        #     dof_vel = 1.5
        #     ang_vel = 0.3
        #     lin_vel = 0.05
        #     quat = 0.05
        #     gravity = 0.05
        #     height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        rot = [0.0, -1, 0, 1.0]

        #Hier nochmal die Target-Pose anschauen
        target_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint': 0.,
            'left_hip_roll_joint': 0.08,
            'left_hip_pitch_joint': 0.1, #-0.1
            'left_knee_pitch_joint': -0.3, #0.3
            'left_ankle_pitch_joint': 0.2, #-0.2
            'left_shoulder_yaw_joint':0.0,
            'left_shoulder_roll_joint':0.3, #-0.3
            'left_shoulder_pitch_joint':0.0,
            'left_elbow_joint':-0.5, #0.7

            'right_hip_yaw_joint': 0.,
            'right_hip_roll_joint': 0.08, #0.0
            'right_hip_pitch_joint': 0.1,
            'right_knee_pitch_joint': -0.3,
            'right_ankle_pitch_joint': 0.2, #-0.2
            'right_shoulder_yaw_joint':0.0,
            'right_shoulder_roll_joint': 0.3,#-0.3
            'right_shoulder_pitch_joint':0.0,
            'right_elbow_joint':-0.5, #0.7

            'torso_joint':0.0,
        }

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint': 0.,
            'left_hip_roll_joint': 0.,
            'left_hip_pitch_joint': 0.5,
            'left_knee_pitch_joint': -1.0,
            'left_ankle_pitch_joint': 0.5,
            'left_shoulder_yaw_joint':0.0,
            'left_shoulder_roll_joint':0.0,
            'left_shoulder_pitch_joint':0.0,
            'left_elbow_joint':0.0,

            'right_hip_yaw_joint': 0.,
            'right_hip_roll_joint': 0.,
            'right_hip_pitch_joint': 0.5,
            'right_knee_pitch_joint': -1.0,
            'right_ankle_pitch_joint': 0.5,
            'right_shoulder_yaw_joint':0.0,
            'right_shoulder_roll_joint':0.0,
            'right_shoulder_pitch_joint':0.0,
            'right_elbow_joint':0.0,

            'torso_joint':0.0
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'   # das kann man noch auf PD ändern     
        
        
        stiffness = {'hip_yaw': 160.0, 
                     'hip_roll': 200.0, 
                     'hip_pitch': 200.0,
                     'knee': 200.0, 
                     'ankle': 30, 
                     'shoulder': 100,
                     'elbow': 100,
                     'torso': 100,}

        damping = { 'hip_yaw': 4, 
                    'hip_roll': 5, 
                    'hip_pitch': 5, 
                   'knee': 5, 
                   'ankle': 1,
                   'shoulder': 4,
                   'elbow': 4,
                   'torso': 4}
        
        # action scale: target angle = actionRescale * action + cur_dof_pos
        action_scale = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4    



    class constraints( LeggedRobotCfg.rewards ):
        is_gaussian = True
        target_head_height = 1.22
        target_head_margin = 1
        orientation_height_threshold = 0.9
        target_base_height = 0.45   # was ist das? # soll warscheinlich so sein also 50 % von finaler höhe

        left_foot_displacement_sigma = -2
        right_foot_displacement_sigma = -2
        hip_yaw_var_sigma = -2
        target_dof_pos_sigma = -0.1
        post_task = False
        
        class scales:
            # regularization reward
            regu_dof_acc = -2.5e-7
            regu_action_rate = -0.01
            regu_smoothness = -0.01 
            regu_torques = -2.5e-6
            regu_joint_power = -2.5e-5
            regu_dof_vel = -1e-3
            regu_joint_tracking_error = -0.00025
            regu_dof_pos_limits = -100.0
            regu_dof_vel_limits = -1 

            # style reward
            style_waist_deviation = -10
            style_hip_yaw_deviation = -10
            style_hip_roll_deviation = -10
            style_shoulder_roll_deviation = -2.5
            style_left_foot_displacement = 2.5
            style_right_foot_displacement = 2.5
            style_knee_deviation = -0.25
            style_shank_orientation = 10
            style_ground_parallel = 20
            style_feet_distance = -10
            style_style_ang_vel_xy = 1

            # post-task reward
            target_ang_vel_xy = 10
            target_lin_vel_xy = 10
            target_feet_height_var = 2.5
            target_target_upper_dof_pos = 10
            target_target_lower_dof_pos = 15 #von uns hinzugefügt
            target_lower_body_var = 10 #hip_yaw ist nicht mit drinnen
            target_lower_body_deviation = 10
            target_target_orientation = 10
            target_target_base_height = 10



    class sim:
        dt = 0.005  
        substeps = 1  # 2
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        use_random = True

        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]

        randomize_motor_strength = use_random
        motor_strength_range = [0.9, 1.1]

        randomize_payload_mass = use_random
        payload_mass_range = [-2, 5]

        randomize_com_displacement = use_random
        com_displacement_range = [-0.03, 0.03]

        randomize_link_mass = use_random
        link_mass_range = [0.8, 1.2]
        
        randomize_friction = use_random
        friction_range = [0.1, 1]
        
        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]
        
        randomize_kp = use_random
        kp_range = [0.85, 1.15]
        
        randomize_kd = use_random
        kd_range = [0.85, 1.15]
        
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.9, 1.1]
        initial_joint_pos_offset = [-0.1, 0.1]
        
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.5

        delay = use_random
        max_delay_timesteps = 5

    

    class rewards( LeggedRobotCfg.rewards ):
        #Die anderen Hyperparameter leigen in der legged_robot_config (mehr oder weniger fest)
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        base_height_target = 0.89
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        orientation_sigma = 1
        is_gaussian = True
        target_head_height = 1.27 # angespasst sind 75% der Höhe vom x02
        target_head_margin = 1
        target_base_height_phase1 = 0.45
        target_base_height_phase2 = 0.45
        target_base_height_phase3 = 0.65
        orientation_threshold = 0.99
        left_foot_displacement_sigma = -2
        right_foot_displacement_sigma = -2
        target_dof_pos_sigma = -0.1
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)

        reward_groups = ['task', 'regu', 'style', 'target']
        num_reward_groups = len(reward_groups)
        reward_group_weights = [2.5, 0.1, 1, 1]

        class scales:
            task_orientation = 1
            task_head_height = 1
    
    
    class curriculum:
        pull_force = True
        force = 100 # 100*2=200 is the actuatl force because of a extra keyframe torso link # haben wir jetzt so gelassen aber kp so wirklich
        dof_vel_limit = 300
        base_vel_limit = 20
        threshold_height = 1.17    # ist aus rewarsd targets_head_height minus 0.1
        no_orientation = True 

class x02CfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256]
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        # smoothness
        value_smoothness_coef = 0.1
        smoothness_upper_bound = 1.0
        smoothness_lower_bound = 0.1
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'x02_ground'
        algorithm_class_name = 'PPO'
        init_at_random_ep_len = True
        max_iterations = 12000 # number of policy updates