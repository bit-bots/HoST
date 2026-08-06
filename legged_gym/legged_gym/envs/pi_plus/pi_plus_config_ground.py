from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Pi_PlusCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.351] # x,y,z [m], updated to match Piwaist
        rot = [0.0, -1, 0, 1.0] # x,y,z,w [quat]
        target_joint_angles = { # = target angles [rad] when action = 0.0
            # left leg (6 dof)
            "l_hip_pitch_joint": -0.2,
            "l_hip_roll_joint": 0.0,
            "l_thigh_joint": 0.0,
            "l_calf_joint": -0.5,
            "l_ankle_pitch_joint": -0.25,
            "l_ankle_roll_joint": 0,
            #left arm (4 dof)
            "l_shoulder_pitch_joint": -1.57,
            "l_shoulder_roll_joint": -1.3,
            "l_upper_arm_joint": 0.0,
            "l_elbow_joint": -1.57,
            # right leg (6 dof)
            "r_hip_pitch_joint": 0.2,
            "r_hip_roll_joint": 0.0,
            "r_thigh_joint": 0.0,
            "r_calf_joint": 0.5,
            "r_ankle_pitch_joint": 0.25,
            "r_ankle_roll_joint": 0,
            #right arm (4 dof)
            "r_shoulder_pitch_joint": -1.57,
            "r_shoulder_roll_joint": -1.3,
            "r_upper_arm_joint": 0.0,
            "r_elbow_joint": -1.57,
            
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
        }

        default_joint_angles = {
           # left leg (6 dof)
            "l_hip_pitch_joint": -0.0,
            "l_hip_roll_joint": 0.0,
            "l_thigh_joint": 0.0,
            "l_calf_joint": 0.0,
            "l_ankle_pitch_joint": -0.0,
            "l_ankle_roll_joint": 0,
            #left arm (4 dof)
            "l_shoulder_pitch_joint": 0.0,  # arm rotation auskommentiert (war -1.57)
            "l_shoulder_roll_joint": 0.0,   # arm rotation auskommentiert (war -0.5)
            "l_upper_arm_joint": 0.0,
            "l_elbow_joint": 0.0,
            # right leg (6 dof)
            "r_hip_pitch_joint": -0.0,
            "r_hip_roll_joint": 0.0,
            "r_thigh_joint": 0.0,
            "r_calf_joint": 0.0,
            "r_ankle_pitch_joint": -0.0,
            "r_ankle_roll_joint": 0,
            #right arm (4 dof)
            "r_shoulder_pitch_joint": 0.0,  # arm rotation auskommentiert (war -1.57)
            "r_shoulder_roll_joint": 0.0,   # arm rotation auskommentiert (war -0.5)
            "r_upper_arm_joint": 0.0,
            "r_elbow_joint": 0.0,
            #head (2 dof)
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
        }

    class env(LeggedRobotCfg.env):
        num_one_step_observations= 73
        num_actions = 22
        num_dofs = 22
        num_actor_history = 6
        num_observations = num_actor_history * num_one_step_observations
        episode_length_s = 10 # episode length in seconds
        unactuated_timesteps = 30

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters, taken 1:1 from the BeyondMimic motion-tracking setup that
        # transfers to the real Pi+ (mocap/Mini-Pi-Plus_BeyondMimic, source/whole_body_tracking/
        # whole_body_tracking/robots/pi_plus.py):
        #   STIFFNESS_5047 = 80, DAMPING_5047 = 1.1  -> all leg + foot joints
        #   STIFFNESS_4438 = 30, DAMPING_4438 = 0.6  -> all arm joints
        # The same gains are hardcoded in that repo's MuJoCo sim2sim validation
        # (scripts/sim2sim.py: PI_PLUS_LEG_KP/KD, PI_PLUS_ARM_KP/KD), i.e. they are the
        # gains the working sim2real policy was trained and validated with.
        # The head is fixed in BeyondMimic, so its gains stay at the bitbots_main MuJoCo
        # values (bitbots_mujoco_sim/xml/pi_plus.xml, class "pi_actuator_head": kp=6, kv=0.6).
        control_type = 'P'
        stiffness = {
            "hip_pitch": 80,
            "hip_roll": 80,
            "thigh": 80,
            "calf": 80,
            "ankle_pitch": 80,
            "ankle_roll": 80,
            "shoulder_pitch": 30,
            "shoulder_roll": 30,
            "upper_arm": 30,
            "elbow": 30,
            "head_yaw": 6,
            "head_pitch": 6,
        }  # [N*m/rad]
        damping = {
            "hip_pitch": 1.1,
            "hip_roll": 1.1,
            "thigh": 1.1,
            "calf": 1.1,
            "ankle_pitch": 1.1,
            "ankle_roll": 1.1,
            "shoulder_pitch": 0.6,
            "shoulder_roll": 0.6,
            "upper_arm": 0.6,
            "elbow": 0.6,
            "head_yaw": 0.6,
            "head_pitch": 0.6,
        }  # [N*m*s/rad]
        # action scale: target angle = actionRescale * action + cur_dof_pos
        action_scale = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 0.8
        dynamic_friction = 0.7
        restitution = 0.3
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 1 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        terrain_proportions = [1, 0., 0, 0, 0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class asset( LeggedRobotCfg.asset ):
        #file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_plus_torso/pi_plus.urdf"
        #file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_plus_torso/pi_plus_limited_knee.urdf"
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_plus_torso/pi_plus_correct_limits.urdf"
        name = "pi_plus"
        left_foot_name = "l_ankle_pitch"
        right_foot_name = "r_ankle_pitch"
        left_knee_name = 'l_calf'
        right_knee_name = 'r_calf'
        left_thigh_name = 'l_hip_pitch'
        right_thigh_name = 'r_hip_pitch'
        foot_name = "ankle_roll"
        penalize_contacts_on = ['calf', 'hip', 'head', 'wrist', 'elbow', 'upper_arm', 'shoulder']
        terminate_after_contacts_on = []    #'torse'

        left_shoulder_name = "l_shoulder"
        right_shoulder_name = "r_shoulder"

        left_leg_joints = [ 'l_hip_pitch_joint', 'l_hip_roll_joint','l_thigh_joint', 'l_calf_joint', 'l_ankle_pitch_joint', 'l_ankle_roll_joint']
        right_leg_joints = [  'r_hip_pitch_joint','r_hip_roll_joint', 'r_thigh_joint','r_calf_joint', 'r_ankle_pitch_joint', 'r_ankle_roll_joint']
       
        left_hip_joints = ['l_thigh_joint']
        right_hip_joints = ['r_thigh_joint']
        left_hip_roll_joints = ['l_hip_roll_joint']
        right_hip_roll_joints = ['r_hip_roll_joint']    
        left_hip_pitch_joints = ['l_hip_pitch_joint']
        right_hip_pitch_joints = ['r_hip_pitch_joint']    

        left_shoulder_roll_joints = ['l_shoulder_roll_joint']
        right_shoulder_roll_joints = ['r_shoulder_roll_joint']

        waist_joints = ['torso_joint']  # Placeholder

        left_knee_joints = ['l_calf_joint']
        right_knee_joints = ['r_calf_joint']    

        left_arm_joints = ['l_shoulder_pitch_joint', 'l_shoulder_roll_joint', 'l_upper_arm_joint', 'l_elbow_joint']
        right_arm_joints = ['r_shoulder_pitch_joint', 'r_shoulder_roll_joint', 'r_upper_arm_joint', 'r_elbow_joint']
        
        knee_joints = ['l_calf_joint', 'r_calf_joint']
        ankle_joints = [ 'l_ankle_pitch_joint', 'l_ankle_roll_joint', 'r_ankle_pitch_joint', 'r_ankle_roll_joint']

        keyframe_name = "keyframe"
        head_name = 'keyframe_head'

        trunk_names = ["base_link"]
        torso_name = 'base_link'
        base_name = 'base_link'
        tracking_body_names =  ['base_link']

        left_upper_body_names = ['l_shoulder_pitch_joint', 'l_shoulder_roll_joint', 'l_upper_arm_joint', 'l_elbow_joint']
        right_upper_body_names = ['r_shoulder_pitch_joint', 'r_shoulder_roll_joint', 'r_upper_arm_joint', 'r_elbow_joint']
        left_lower_body_names = ['l_hip_pitch', 'l_ankle_roll', 'l_calf']
        right_lower_body_names = ['r_hip_pitch', 'r_ankle_roll', 'r_calf']

        left_ankle_names = ['l_ankle_roll']
        right_ankle_names = ['r_ankle_roll']

        density = 0.001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01316  # uniform fallback; per-joint values applied via per_joint_armature
        # Per-joint armature override (substring match against dof_names) to
        # mirror the real-robot / bitbots_main MuJoCo XML, which sets armature
        # per joint group rather than uniformly. Applied in
        # `host_ground._process_dof_props`. The substrings here use the same
        # pattern as `control.stiffness`/`control.damping` for consistency.
        per_joint_armature = {
            "hip":        0.01316,
            "thigh":      0.01316,
            "calf":       0.01316,
            "ankle":      0.01316,
            "shoulder":   0.01317,
            "upper_arm":  0.01317,
            "elbow":      0.01317,
            "head":       0.00249,
        }
        # Passive DOF drive damping / Coulomb friction.
        # pi_plus_correct_limits.urdf carries <dynamics damping="1.5"/> (legs),
        # damping="0.66" friction="0.2" (arms) and damping="0.48" friction="0.1" (head),
        # copied from the bitbots_main MuJoCo model. Isaac Gym loads those into
        # dof_props["damping"]/["friction"], i.e. into the PhysX drive, where they can act
        # *on top of* the explicit PD torques from `_compute_torques` - that would give the
        # leg joints an effective kd of 1.1 + 1.5 = 2.6 instead of 1.1.
        # The BeyondMimic setup this robot actually transfers with has no passive term at
        # all: Isaac Lab ignores the URDF <dynamics> tag and imports the joint drive with
        # stiffness=0/damping=0 (robots/pi_plus.py, UrdfConverterCfg.JointDriveCfg), so
        # `control.stiffness`/`control.damping` are the complete actuator model there.
        # Zeroing them here reproduces that; also what the Isaac Gym docs prescribe for
        # DOF_MODE_EFFORT. Applied in `host_ground._process_dof_props`.
        # Set to None to keep the URDF values instead.
        dof_damping = 0.0
        dof_friction = 0.0
        thickness = 0.01
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        base_height_sigma = 0.3 #für ground_prone hinuzgefügt
        tracking_dof_sigma = 0.3 #für ground_prone hinuzgefügt
        base_height_target = 0.34  # updated to match Piwaist
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        orientation_sigma = 1
        is_gaussian = True
        target_head_height = 0.45  # updated to match Piwaist head_height_target (base_height + 0.08)
        target_head_margin = 0.30
        target_base_height_phase1 = 0.23  # updated to match Piwaist
        target_base_height_phase2 = 0.23 #0.05 updated to 0.05 to get better standing style
        target_base_height_phase3 = 0.35  # updated to match Piwaist
        orientation_threshold = 0.99
        left_foot_displacement_sigma = -20#-200 updated to get better standing style
        right_foot_displacement_sigma = -20#-200 updated to get better standing style
        target_dof_pos_sigma = -0.1
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)

        reward_groups = ['task', 'regu', 'style', 'target']
        num_reward_groups = len(reward_groups)
        reward_group_weights = [2.5, 0.1, 1, 1]

        class scales:
            task_orientation = 1
            task_head_height = 1

    class constraints( LeggedRobotCfg.rewards ):
        is_gaussian = True
        target_head_height = 0.37
        target_head_margin = 0.37
        orientation_height_threshold = 0.9
        target_base_height = 0.34  # updated to match Piwaist

        left_foot_displacement_sigma = -20#-200 updated to get better standing style
        right_foot_displacement_sigma = -20#-200 updated to get better standing style
        hip_yaw_var_sigma = -2
        target_dof_pos_sigma = -0.1
        # allowed knee overshoot past straight (rad) before style_knee_hyperextension bites
        knee_hyperextension_margin = 0.1
        # Coupled hip_pitch/calf limit, see _reward_hip_knee_coupling. The two points are
        # (r_hip_pitch, r_calf) poses read off in the URDF viewer at which the heel reaches
        # the bottom / the top of the torso; the line through them is the boundary. Forbidden
        # is hip_pitch further negative and calf further positive. The left leg is the exact
        # mirror of this (l_hip_pitch 1.28 / l_calf -1.76 and 2.03 / -1.00).
        hip_knee_coupling_p1 = (-1.28, 1.76)   # heel at the bottom of the torso
        hip_knee_coupling_p2 = (-2.03, 1.00)   # heel at the top of the torso
        # The line IS the self-collision boundary, so keep clear of it: the effective boundary
        # sits offset rad inside it (0.15 rad perpendicular ~ 12 deg of calf or hip alone), and
        # the warning ramp starts another margin rad before that (~37 deg of calf in total).
        hip_knee_coupling_offset = 0.15
        hip_knee_coupling_margin = 0.3         # rad of warning zone before the boundary
        hip_knee_coupling_soft = 0.4           # weight of that zone vs. the actual overshoot
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
            #style_waist_deviation = -10 #pi+ has no waist joint
            style_hip_yaw_deviation = -10
            style_hip_roll_deviation = -10
            style_hip_pitch_deviation = -10
            style_shoulder_roll_deviation = -10
            style_left_foot_displacement = 7.5 #7.5 updated to get better standing style
            style_right_foot_displacement = 7.5 #7.5  updated to get better standing style
            # style_knee_deviation = -0.25 
            # style_knee_hyperextension below.
            style_knee_hyperextension = -10
            style_hip_knee_coupling = -10
            #style_knee_deviation_pi_plus = -20
            style_shank_orientation = 10
            style_ground_parallel = 25
            style_feet_distance = 10
            style_style_ang_vel_xy = 1
            style_feet_parallel = -10
            #style_soft_symmetry_action=-10  #  updated to get better standing style
            #style_soft_symmetry_body=2.5 # updated to get better standing style

            # post-task reward
            target_ang_vel_xy = 5
            target_lin_vel_xy = 5
            target_feet_height_var = 2.5
            target_lower_body_deviation = 20
            target_upper_body_var = 20
            target_target_lower_dof_pos = 30  #  updated to get better standing style
            target_target_upper_dof_pos = 30
            target_target_orientation = 20
            target_target_base_height = 10
            #target_target_knee_angle = 10 #  updated to get better standing style

    class domain_rand:
        use_random = True

        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]

        randomize_motor_strength = use_random
        motor_strength_range = [0.9, 1.1]

        randomize_payload_mass = use_random
        payload_mass_range = [-0.5, 1]

        randomize_com_displacement = use_random
        com_displacement_range = [-0.03, 0.03]

        randomize_link_mass = use_random
        link_mass_range = [0.8, 1.2]
        
        randomize_friction = use_random
        friction_range = [0.1, 1]
        
        randomize_restitution = use_random
        restitution_range = [0.0, 0.7]
        
        randomize_kp = use_random
        kp_range = [0.85, 1.15]
        
        randomize_kd = use_random
        kd_range = [0.85, 1.15]
        
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.5, 1.2]
        initial_joint_pos_offset = [-0.1, 0.1]

        randomize_arm_joint_pos = False
        arm_joint_pos_scale = [0.3, 1.5]
        arm_joint_pos_offset = [-0.4, 0.4]

        push_robots = True 
        push_interval_s = 10
        max_push_vel_xy = 0.5

        delay = use_random
        max_delay_timesteps = 5
    
    class curriculum:
        pull_force = True
        force = 60 # 100*2=200 is the actuatl force because of a extra keyframe torso link
        dof_vel_limit = 300
        base_vel_limit = 20
        threshold_height = 0.37
        no_orientation = False

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class Pi_PlusCfgPPO( LeggedRobotCfgPPO ):
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
        experiment_name = 'Pi_Plus_ground'
        algorithm_class_name = 'PPO'
        init_at_random_ep_len = True
        max_iterations = 12000 # number of policy updates