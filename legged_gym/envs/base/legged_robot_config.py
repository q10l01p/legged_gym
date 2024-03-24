from params_proto import PrefixProto, ParamsProto


class LeggedRobotCfg(PrefixProto, cli=False):
    class env(PrefixProto, cli=False):
        num_envs = 4096       # 环境的数量
        num_observations = 235      # 每个环境的观测值数量
        num_scalar_observations = 42  # 标量观察值的数量
        num_privileged_obs = 18   # 特权观测值的数量，None表示不返回特权观测值
        privileged_future_horizon = 1  # 特权未来视野
        num_actions = 12            # 每个环境的动作数量
        num_observation_history = 15  # 观察历史的数量
        env_spacing = 3.            # 环境间的间距，这个值在使用高度场/三角网格时不使用
        send_timeouts = True        # 是否将超时信息发送给算法
        episode_length_s = 20       # 单个回合的长度（秒）
        observe_vel = True  # 观察速度
        observe_only_ang_vel = False  # 仅观察角速度
        observe_only_lin_vel = False  #
        observe_yaw = False  # 观察偏航
        observe_contact_states = False  # 观察接触状态
        observe_command = True  # 观察命令
        observe_height_command = False  # 观察高度命令
        observe_gait_commands = False  # 观察步态命令
        observe_timing_parameter = False  # 观察时间参数
        observe_clock_inputs = False  # 观察时钟输入
        observe_two_prev_actions = False  # 观察前两个动作
        observe_imu = False  # 观察惯性测量单元
        record_video = True  # 记录视频
        recording_width_px = 512  # 视频宽度（像素）
        recording_height_px = 512  # 视频高度（像素）
        recording_mode = "COLOR"  # 录制模式
        num_recording_envs = 1  # 录制的环境数量
        debug_viz = False  # 调试可视化
        all_agents_share = False  # 所有代理是否共享配置

        # 以下是特权观察的相关配置，主要涉及到仿真环境的物理特性
        priv_observe_friction = False  # 观察摩擦
        priv_observe_friction_indep = False  # 独立观察摩擦
        priv_observe_ground_friction = False  # 观察地面摩擦
        priv_observe_ground_friction_per_foot = False  # 每只脚的地面摩擦
        priv_observe_restitution = False  # 观察恢复系数
        priv_observe_base_mass = False  # 观察基础质量
        priv_observe_com_displacement = False  # 观察质心偏移
        priv_observe_motor_strength = False  # 观察电机强度
        priv_observe_motor_offset = False  # 观察电机偏移
        priv_observe_joint_friction = False  # 观察关节摩擦
        priv_observe_Kp_factor = False  # 观察比例增益因子
        priv_observe_Kd_factor = False  # 观察微分增益因子
        priv_observe_contact_forces = False  # 观察接触力
        priv_observe_contact_states = False  # 观察接触状态
        priv_observe_body_velocity = False  # 观察身体速度
        priv_observe_foot_height = False  # 观察脚部高度
        priv_observe_body_height = False  # 观察身体高度
        priv_observe_gravity = False  # 观察重力
        priv_observe_heights = False  # 观察地形高度
        priv_observe_terrain_type = False  # 观察地形类型
        priv_observe_clock_inputs = False  # 观察时钟输入
        priv_observe_doubletime_clock_inputs = False  # 观察双倍时间时钟输入
        priv_observe_halftime_clock_inputs = False  # 观察半倍时间时钟输入
        priv_observe_desired_contact_states = False  # 观察期望的接触状态
        priv_observe_dummy_variable = False  # 观察虚拟变量

    class terrain(PrefixProto, cli=False):
        mesh_type = 'trimesh'   # 地形的网格类型
        x_offset = 0            # 水平偏移量
        rows_offset = 0         # 设置行偏移量为0
        horizontal_scale = 0.1  # 水平缩放比例，单位是米
        vertical_scale = 0.005  # 垂直缩放比例，单位是米
        border_size = 25        # 边界大小，单位是米
        curriculum = True       # 是否启用课程学习
        static_friction = 1.0   # 静摩擦系数
        dynamic_friction = 1.0  # 动摩擦系数
        restitution = 0.        # 弹性恢复系数
        terrain_noise_magnitude = 0.1  # 地形噪声幅度

        # 粗糙地形专用:
        terrain_smoothness = 0.005  # 地形平滑度
        measure_heights = True      # 是否测量粗糙地形的高度
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m矩形的X轴测量点（不包括中心线）
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]  # Y轴测量点
        selected = False            # 是否选择了唯一的地形类型并传递所有参数
        terrain_kwargs = None       # 选定地形类型的参数字典
        min_init_terrain_level = 0  # 初始化地形的最低级别
        max_init_terrain_level = 5  # 课程学习开始的地形等级
        terrain_length = 8.         # 地形的长度
        terrain_width = 8.          # 地形的宽度
        num_rows = 2               # 地形行数（等级）
        num_cols = 2               # 地形列数（类型）
        # 地形类型: [平滑斜坡, 粗糙斜坡, 上楼梯, 下楼梯, 离散]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]  # 不同地形类型的比例列表

        # 三角网格专用:
        slope_treshold = 0.75  # 斜率阈值，高于此阈值的斜率将被修正为垂直表面
        difficulty_scale = 1.  # 难度缩放比例
        x_init_range = 1.  # 初始X位置范围
        y_init_range = 1.  # 初始Y位置范围
        yaw_init_range = 0.  # 初始偏航角范围
        x_init_offset = 0.  # 初始X位置偏移
        y_init_offset = 0.  # 初始Y位置偏移
        teleport_robots = True  # 是否传送机器人
        teleport_thresh = 2.0  # 传送阈值
        max_platform_height = 0.2  # 最大平台高度
        center_robots = False  # 是否将机器人置于中心
        center_span = 5  # 中心跨度

    class commands(PrefixProto, cli=False):
        command_curriculum = False  # 是否启用命令课程制
        max_reverse_curriculum = 1.  # 最大反向课程值F
        max_forward_curriculum = 1.  # 最大正向课程值
        yaw_command_curriculum = False  # 是否启用偏航命令课程制
        max_yaw_curriculum = 1.  # 最大偏航课程值
        exclusive_command_sampling = False  # 是否采用独占命令采样
        num_commands = 3  # 命令数量
        resampling_time = 10.  # 命令变更前的时间间隔（秒）
        subsample_gait = False  # 是否进行步态子采样
        gait_interval_s = 10.  # 步态参数重采样时间间隔
        vel_interval_s = 10.  # 速度命令重采样时间间隔
        jump_interval_s = 20.  # 跳跃间隔时间
        jump_duration_s = 0.1  # 跳跃持续时间
        jump_height = 0.3  # 跳跃高度
        heading_command = True  # 是否根据航向误差计算角速度命令
        global_reference = False  # 是否使用全局参考
        observe_accel = False  # 是否观察加速度
        distributional_commands = False  # 是否使用分布式命令
        curriculum_type = "RewardThresholdCurriculum"  # 课程制类型
        lipschitz_threshold = 0.9  # 李普希茨阈值

        num_lin_vel_bins = 20  # 线速度分箱数量
        lin_vel_step = 0.3  # 线速度步长
        num_ang_vel_bins = 20  # 角速度分箱数量
        ang_vel_step = 0.3  # 角速度步长
        distribution_update_extension_distance = 1  # 分布更新扩展距离
        curriculum_seed = 100  # 课程制种子

        lin_vel_x = [-1.0, 1.0]  # 线速度X范围（米/秒）
        lin_vel_y = [-1.0, 1.0]  # 线速度Y范围（米/秒）
        ang_vel_yaw = [-1, 1]  # 角速度偏航范围（弧度/秒）
        body_height_cmd = [-0.05, 0.05]  # 身体高度命令范围
        impulse_height_commands = False  # 是否启用脉冲高度命令

        # 以下为限制参数，定义命令和动作的限制范围
        limit_vel_x = [-10.0, 10.0]  # 线速度X限制
        limit_vel_y = [-0.6, 0.6]  # 线速度Y限制
        limit_vel_yaw = [-10.0, 10.0]  # 角速度偏航限制
        limit_body_height = [-0.05, 0.05]  # 身体高度限制
        limit_gait_phase = [0, 0.01]  # 步态相位限制
        limit_gait_offset = [0, 0.01]  # 步态偏移限制
        limit_gait_bound = [0, 0.01]  # 步态界限限制
        limit_gait_frequency = [2.0, 2.01]  # 步态频率限制
        limit_gait_duration = [0.49, 0.5]  # 步态持续时间限制
        limit_footswing_height = [0.06, 0.061]  # 脚摆动高度限制
        limit_body_pitch = [0.0, 0.01]  # 身体俯仰角限制
        limit_body_roll = [0.0, 0.01]  # 身体翻滚角限制
        limit_aux_reward_coef = [0.0, 0.01]  # 辅助奖励系数限制
        limit_compliance = [0.0, 0.01]  # 合规性限制
        limit_stance_width = [0.0, 0.01]  # 站姿宽度限制
        limit_stance_length = [0.0, 0.01]  # 站姿长度限制

        # 以下为不同参数的分箱数量，用于将连续值离散化
        num_bins_vel_x = 25  # 设置x方向速度的离散分箱数量为25
        num_bins_vel_y = 3  # 设置y方向速度的离散分箱数量为3
        num_bins_vel_yaw = 25  # 设置偏航速度的离散分箱数量为25
        num_bins_body_height = 1  # 设置身体高度的离散分箱数量为1
        num_bins_gait_frequency = 11  # 设置步态频率的离散分箱数量为11
        num_bins_gait_phase = 11  # 设置步态相位的离散分箱数量为11
        num_bins_gait_offset = 2  # 设置步态偏置的离散分箱数量为2
        num_bins_gait_bound = 2  # 设置步态边界的离散分箱数量为2
        num_bins_gait_duration = 3  # 设置步态持续时间的离散分箱数量为3
        num_bins_footswing_height = 1  # 设置步行摆动高度的离散分箱数量为1
        num_bins_body_pitch = 1  # 设置身体倾斜的离散分箱数量为1
        num_bins_body_roll = 1  # 设置身体滚动的离散分箱数量为1
        num_bins_aux_reward_coef = 1  # 设置辅助奖励系数的离散分箱数量为1
        num_bins_compliance = 1  # 设置合规性的离散分箱数量为1
        num_bins_stance_width = 1  # 设置站姿宽度的离散分箱数量为1
        num_bins_stance_length = 1  # 设置站姿长度的离散分箱数量为1

        heading = [-3.14, 3.14]  # 航向范围（弧度）

        # 以下为各种步态命令的范围
        gait_phase_cmd_range = [0.0, 0.01]  # 设置步态相位命令的范围为[0.0, 0.01]
        gait_offset_cmd_range = [0.0, 0.01]  # 设置步态偏置命令的范围为[0.0, 0.01]
        gait_bound_cmd_range = [0.0, 0.01]  # 设置步态边界命令的范围为[0.0, 0.01]
        gait_frequency_cmd_range = [2.0, 2.01]  # 设置步态频率命令的范围为[2.0, 2.01]
        gait_duration_cmd_range = [0.49, 0.5]  # 设置步态持续时间命令的范围为[0.49, 0.5]
        footswing_height_range = [0.06, 0.061]  # 设置步行摆动高度范围为[0.06, 0.061]
        body_pitch_range = [0.0, 0.01]  # 设置身体倾斜范围为[0.0, 0.01]
        body_roll_range = [0.0, 0.01]  # 设置身体滚动范围为[0.0, 0.01]
        aux_reward_coef_range = [0.0, 0.01]  # 设置辅助奖励系数范围为[0.0, 0.01]
        compliance_range = [0.0, 0.01]  # 设置合规性范围为[0.0, 0.01]
        stance_width_range = [0.0, 0.01]  # 设置站姿宽度范围为[0.0, 0.01]
        stance_length_range = [0.0, 0.01]  # 设置站姿长度范围为[0.0, 0.01]

        exclusive_phase_offset = True  # 设置是否独占相位偏移为真
        binary_phases = False  # 设置是否使用二元相位为假
        pacing_offset = False  # 设置是否使用步态偏移为假
        balance_gait_distribution = True  # 设置是否平衡步态分布为真
        gaitwise_curricula = True  # 设置是否使用步态课程制为真

    class curriculum_thresholds(PrefixProto, cli=False):
        tracking_lin_vel = 0.8  # 定义线速度追踪阈值，越接近1精度越高
        tracking_ang_vel = 0.5  # 定义角速度追踪阈值
        tracking_contacts_shaped_force = 0.8  # 定义接触成型力的追踪阈值，越接近1精度越高
        tracking_contacts_shaped_vel = 0.8  # 定义接触成型速度的追踪阈值，越接近1精度越高

    class init_state(PrefixProto, cli=False):
        pos = [0.0, 0.0, 1.]        # 初始位置坐标，单位是米
        rot = [0.0, 0.0, 0.0, 1.0]  # 初始旋转四元数
        lin_vel = [0.0, 0.0, 0.0]   # 初始线速度，单位是米/秒
        ang_vel = [0.0, 0.0, 0.0]   # 初始角速度，单位是弧度/秒
        default_joint_angles = {    # 当动作值为0.0时的目标关节角度
            "joint_a": 0.,
            "joint_b": 0.
        }

    class control(PrefixProto, cli=False):
        control_type = 'actuator_net' # 控制类型：执行器网络，备选：'P' 위치，'V'速度，'T'扭矩
        # PD控制器参数:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # 刚度参数，单位是牛顿米/弧度
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # 阻尼参数，单位是牛顿米*秒/弧度
        # 动作缩放比例:
        action_scale = 0.5      # 目标角度 = 动作缩放比例 * 动作值 + 默认角度
        # 控制动作更新的抽样率:
        decimation = 4          # 控制动作更新的抽样率

        hip_scale_reduction = 1.0  # 臀部缩放减少量

    class asset(PrefixProto, cli=False):
        file = ""                               # 资产文件路径
        name = "legged_robot"                   # 资产的名称
        foot_name = "None"                      # 脚部物体的名称
        penalize_contacts_on = []  # 罚项接触的列表，默认为空。被用于处理接触时的惩罚处理
        terminate_after_contacts_on = []  # 终止接触的列表，默认为空。被用于处理接触后的终止行为
        disable_gravity = False  # 是否禁用重力，默认为`False`
        collapse_fixed_joints = True  # 是否合并由固定关节连接的身体，默认为`True`。特定的固定关节可以通过添加" <... dont_collapse="true">来保持
        fix_base_link = False  # 是否固定机器人的基座，默认为`False`
        default_dof_drive_mode = 3  # 看GymDofDriveMode标志（0是none，1是pos tgt，2是vel tgt，3是effort）
        self_collisions = 0  # 自我碰撞，默认为0（即启用）。1表示禁用。这是一个位掩码（bitwise filter）
        replace_cylinder_with_capsule = True  # 是否将碰撞的圆柱体替换为胶囊。默认为`True`。这种替换可以带来更快/更稳定的模拟
        flip_visual_attachments = True  # 是否反转视觉附件。默认为`True`。有些.obj网格必须从y-up反转为z-up

        # 物理特性参数:
        density = 0.001                 # 密度
        angular_damping = 0.            # 角阻尼
        linear_damping = 0.             # 线阻尼
        max_angular_velocity = 1000.    # 最大角速度
        max_linear_velocity = 1000.     # 最大线速度
        armature = 0.                   # 骨架值
        thickness = 0.01                # 厚度

    class domain_rand(PrefixProto, cli=False):
        rand_interval_s = 10                    # 随机因子间隔时间，单位为秒，默认值为10
        randomize_rigids_after_start = True     # 启动之后是否随机化刚体，默认值为`True`
        randomize_friction = True               # 是否随机化摩擦力，默认值为`True`
        friction_range = [0.5, 1.25]            # 摩擦力范围，范围从0.5到1.25
        randomize_restitution = False           # 是否随机化恢复力，默认值为`False`
        restitution_range = [0, 1.0]            # 恢复力范围，范围从0到1
        randomize_base_mass = False             # 是否随机化基础质量，默认值为`False`
        added_mass_range = [-1., 1.]            # 增加质量的范围，范围从-1到1
        randomize_com_displacement = False      # 是否随机化质心位移，默认值为`False`
        com_displacement_range = [-0.15, 0.15]  # 质心位移范围，范围从-0.15到0.15
        randomize_motor_strength = False        # 是否随机化马达强度，默认值为`False`
        motor_strength_range = [0.9, 1.1]       # 马达强度范围，范围从0.9到1.1
        randomize_motor_offset = False          # 随机化电机偏移
        randomize_Kp_factor = False             # 是否随机化Kp系数，默认值为`False`
        Kp_factor_range = [0.8, 1.3]            # Kp系数范围，范围从0.8到1.3
        randomize_Kd_factor = False             # 是否随机化Kd系数，默认值为`False`
        Kd_factor_range = [0.5, 1.5]            # Kd系数范围，范围从0.5到1.5
        gravity_rand_interval_s = 7             # 重力随机间隔时间，单位为秒，默认值为7
        gravity_impulse_duration = 1.0          # 重力脉冲持续时间，默认为1秒
        randomize_gravity = False               # 是否随机化重力，默认值为`False`
        gravity_range = [-1.0, 1.0]             # 重力范围，范围从-1.0到1.0
        push_robots = True                      # 是否对机器人施加推力，默认为`True`
        push_interval_s = 15                    # 施加推力的间隔时间，单位为秒，默认值为15
        max_push_vel_xy = 1.                    # 最大推力速度（在xy平面），默认值为1
        randomize_lag_timesteps = True          # 是否随机化滞后时间步，默认值为`True`
        lag_timesteps = 6                       # 滞后时间步，默认值为6

    class rewards(PrefixProto, cli=False):
        only_positive_rewards = True                # 如果为真，负的总奖励会在零处被截断（避免早期终止问题）。默认值为`True`
        only_positive_rewards_ji22_style = False    # 只有ji22样式的正奖励，默认值为`False`
        sigma_rew_neg = 5                           # sigma负奖励，默认值为5
        reward_container_name = "CoRLRewards"       # 奖励容器名称，默认为"CoRLRewards"
        tracking_sigma = 0.25                       # 追踪奖励 = exp( - error ^ 2 / sigma)，默认sigma值为0.25
        tracking_sigma_lat = 0.25                   # 纬度追踪奖励 = exp( - error ^ 2 / sigma)，同上
        tracking_sigma_long = 0.25                  # 经度追踪奖励 = exp( - error ^ 2 / sigma)，同上
        tracking_sigma_yaw = 0.25                   # 偏航追踪奖励 = exp( - error ^ 2 / sigma)，同上
        soft_dof_pos_limit = 1.                     # dof（自由度）位置限制，超过这个限制的值将被惩罚，默认值为1
        soft_dof_vel_limit = 1.                     # dof（自由度）速度限制，超过这个限制的值将被惩罚，默认值为1
        soft_torque_limit = 1.                      # 扭矩限制，超过这个限制的值将被惩罚，默认值为1
        base_height_target = 1.                     # 基础高度目标，默认值为1
        max_contact_force = 100.                    # 最大接触力，超过这个值的力将被惩罚，默认值为100
        use_terminal_body_height = False            # 是否使用终端身体高度，默认值为`False`
        terminal_body_height = 0.20                 # 终端身体高度，默认值为0.20
        use_terminal_foot_height = False            # 是否使用终端脚高度，默认值为`False`
        terminal_foot_height = -0.005               # 终端脚高度，默认值为-0.005
        use_terminal_roll_pitch = False             # 是否使用终端滚动和俯仰，默认值为`False`
        terminal_body_ori = 0.5                     # 终端身体方向，默认值为0.5
        kappa_gait_probs = 0.07                     # kappa步态概率，默认值为0.07
        gait_force_sigma = 50.                      # 步态力sigma，默认值为50
        gait_vel_sigma = 0.5                        # 步态速度sigma，默认值为0.5
        footswing_height = 0.09                     # 脚摆动高度，默认值为0.09

    class reward_scales(PrefixProto, cli=False):
        termination = -0.0  # 终止奖励，其值默认为-0.0
        tracking_lin_vel = 1.0  # 追踪线速度奖励，默认值为1.0
        tracking_ang_vel = 0.5  # 追踪角速度奖励，默认值为0.5
        lin_vel_z = -2.0  # z方向的线速度奖励，默认值为-2.0
        ang_vel_xy = -0.05  # xy方向的角速度奖励，默认值为-0.05
        orientation = -0.  # 方向奖励，默认值为 -0
        torques = -0.00001  # 扭矩奖励，默认值为-0.00001
        dof_vel = -0.  # 自由度速度奖励，默认值为-0
        dof_acc = -2.5e-7  # 自由度加速度奖励，默认值为-2.5e-7
        base_height = -0.  # 基础高度奖励，默认值为-0
        feet_air_time = 1.0  # 脚部空中时间奖励，默认值为1.0
        collision = -1.  # 碰撞奖励，默认值为-1
        feet_stumble = -0.0  # 脚部跌倒奖励，默认值为-0.0
        action_rate = -0.01  # 动作速率奖励，默认值为-0.01
        stand_still = -0.  # 静止状态奖励，默认值为-0
        tracking_lin_vel_lat = 0.  # 追踪线速度(纬度方向)，默认值为0
        tracking_lin_vel_long = 0.  # 追踪线速度(经度方向)，默认值为0
        tracking_contacts = 0.  # 追踪接触奖励，默认值为0
        tracking_contacts_shaped = 0.  # 追踪形状化接触奖励，默认值为0
        tracking_contacts_shaped_force = 0.  # 追踪形状化接触力奖励，默认值为0
        tracking_contacts_shaped_vel = 0.  # 追踪形状化接触速度奖励，默认值为0
        jump = 0.0  # 跳绳奖励，默认值为0.0
        energy = 0.0  # 能量奖励，默认值为0.0
        energy_expenditure = 0.0  # 能量消耗奖励，默认值为0.0
        survival = 0.0  # 生存奖励，默认值为0.0
        dof_pos_limits = 0.0  # 自由度位置限制奖励，默认值为0.0
        feet_contact_forces = 0.  # 脚部接触力奖励，默认值为0
        feet_slip = 0.  # 脚部滑动奖励，默认值为0
        feet_clearance_cmd_linear = 0.  # 脚部清除线性命令奖励，默认值为0
        dof_pos = 0.  # 自由度位置奖励，默认值为0
        action_smoothness_1 = 0.  # 动作平滑度1奖励，默认值为0
        action_smoothness_2 = 0.  # 动作平滑度2奖励，默认值为0
        base_motion = 0.  # 基础运动奖励，默认值为0
        feet_impact_vel = 0.0  # 脚部冲击速度奖励，默认值为0.0
        raibert_heuristic = 0.0  # Raibert启发式奖励，默认值为0.0

    class normalization(PrefixProto, cli=False):  # 定义名为`normalization`的类，继承自`PrefixProto`，`cli`是一个参数，默认值为`False`
        clip_observations = 100.  # 观测剪裁值，默认值为100.0
        clip_actions = 100.  # 动作剪裁值，默认值为100.0

        friction_range = [0.05, 4.5]  # 摩擦力范围，默认范围从0.05到4.5
        ground_friction_range = [0.05, 4.5]  # 地面摩擦力范围，默认范围从0.05到4.5
        restitution_range = [0, 1.0]  # 恢复力范围，默认范围从0到1.0
        added_mass_range = [-1., 3.]  # 增加的质量范围，默认范围从-1.0到3.0
        com_displacement_range = [-0.1, 0.1]  # 质心位移范围，默认范围从-0.1到0.1
        motor_strength_range = [0.9, 1.1]  # 马达强度范围，默认范围从0.9到1.1
        motor_offset_range = [-0.05, 0.05]  # 马达偏移范围，默认范围从-0.05到0.05
        Kp_factor_range = [0.8, 1.3]  # Kp系数范围，默认范围从0.8到1.3
        Kd_factor_range = [0.5, 1.5]  # Kd系数范围，默认范围从0.5到1.5
        joint_friction_range = [0.0, 0.7]  # 关节摩擦力范围，默认范围从0.0到0.7
        contact_force_range = [0.0, 50.0]  # 接触力范围，默认范围从0.0到50.0
        contact_state_range = [0.0, 1.0]  # 接触状态范围，默认范围从0.0到1.0
        body_velocity_range = [-6.0, 6.0]  # 机体速度范围，默认范围从-6.0到6.0
        foot_height_range = [0.0, 0.15]  # 脚高度范围，默认范围从0.0到0.15
        body_height_range = [0.0, 0.60]  # 机体高度范围，默认范围从0.0到0.60
        gravity_range = [-1.0, 1.0]  # 重力范围，默认范围从-1.0到1.0
        motion = [-0.01, 0.01]  # 运动范围，默认范围从-0.01到0.01

    class obs_scales(PrefixProto, cli=False):
        lin_vel = 2.0  # 线速度观测缩放，默认值为2.0
        ang_vel = 0.25  # 角速度观测缩放，默认值为0.25
        dof_pos = 1.0  # 自由度位置观测缩放，默认值为1.0
        dof_vel = 0.05  # 自由度速度观测缩放，默认值为0.05
        imu = 0.1  # IMU(惯性测量单元)观测缩放，默认值为0.1
        height_measurements = 5.0  # 高度测量观测缩放，默认值为5.0
        friction_measurements = 1.0  # 摩擦测量观测缩放，默认值为1.0
        body_height_cmd = 2.0  # 机体高度命令观测缩放，默认值为2.0
        gait_phase_cmd = 1.0  # 步态相位命令观测缩放，默认值为1.0
        gait_freq_cmd = 1.0  # 步态频率命令观测缩放，默认值为1.0
        footswing_height_cmd = 0.15  # 脚摆动高度命令观测缩放，默认值为0.15
        body_pitch_cmd = 0.3  # 机体俯仰命令观测缩放，默认值为0.3
        body_roll_cmd = 0.3  # 机体滚动命令观测缩放，默认值为0.3
        aux_reward_cmd = 1.0  # 辅助奖励命令观测缩放，默认值为1.0
        compliance_cmd = 1.0  # 合规命令观测缩放，默认值为1.0
        stance_width_cmd = 1.0  # 站姿宽度命令观测缩放，默认值为1.0
        stance_length_cmd = 1.0  # 站姿长度命令观测缩放，默认值为1.0
        segmentation_image = 1.0  # 分割图像观测缩放，默认值为1.0
        rgb_image = 1.0  # RGB图像观测缩放，默认值为1.0
        depth_image = 1.0  # 深度图像观测缩放，默认值为1.0

    class noise(PrefixProto, cli=False):
        add_noise = True    # 是否添加噪声
        noise_level = 1.0   # 噪声水平，用于缩放其他噪声值

    class noise_scales(PrefixProto, cli=False):
        dof_pos = 0.01  # 自由度位置的噪声缩放值，默认值为0.01
        dof_vel = 1.5  # 自由度速度的噪声缩放值，默认值为1.5
        lin_vel = 0.1  # 线速度的噪声缩放值，默认值为0.1
        ang_vel = 0.2  # 角速度的噪声缩放值，默认值为0.2
        imu = 0.1  # IMU(惯性测量单元)的噪声缩放值，默认值为0.1
        gravity = 0.05  # 重力的噪声缩放值，默认值为0.05
        contact_states = 0.05  # 接触状态的噪声缩放值，默认值为0.05
        height_measurements = 0.1  # 高度测量的噪声缩放值，默认值为0.1
        friction_measurements = 0.0  # 摩擦测量的噪声缩放值，默认值为0.0
        segmentation_image = 0.0  # 分割图像的噪声缩放值，默认值为0.0
        rgb_image = 0.0  # RGB图像的噪声缩放值，默认值为0.0
        depth_image = 0.0  # 深度图像的噪声缩放值，默认值为0.0

    class viewer(PrefixProto, cli=False):
        ref_env = 0             # 参考环境索引
        pos = [10, 0, 6]        # 相机位置坐标，单位：米
        lookat = [11., 5, 3.]   # 相机注视点坐标，单位：米

    class sim(PrefixProto, cli=False):
        dt =  0.005                 # 仿真步长
        substeps = 1                # 每个仿真步长的子步数
        gravity = [0., 0. ,-9.81]   # 重力向量，单位：米/秒^2
        up_axis = 1                 # 上方轴的索引，0代表y轴，1代表z轴
        use_gpu_pipeline = True     # 是否使用GPU管道进行模拟，默认值为True

        class physx(PrefixProto, cli=False):
            num_threads = 10                    # 使用的线程数
            solver_type = 1                     # 求解器类型，0代表PGS，1代表TGS
            num_position_iterations = 4         # 位置迭代次数
            num_velocity_iterations = 0         # 速度迭代次数
            contact_offset = 0.01               # 接触偏移量，单位：米
            rest_offset = 0.0                   # 静止偏移量，单位：米
            bounce_threshold_velocity = 0.5     # 弹跳阈值速度，单位：米/秒
            max_depenetration_velocity = 1.0    # 最大穿透速度
            max_gpu_contact_pairs = 2**23       # GPU接触对的最大数量
            default_buffer_size_multiplier = 5  # 默认缓冲区大小乘数
            contact_collection = 2              # 接触收集方式，0代表从不，1代表最后一个子步骤，2代表所有子步骤
