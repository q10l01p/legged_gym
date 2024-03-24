from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class CassieRoughCfg(LeggedRobotCfg):
    """
    CassieRoughCfg类继承自LeggedRobotCfg，用于配置Cassie机器人在粗糙地形上的仿真环境。

    类属性:
    - env: 设置仿真环境的参数。
    - terrain: 设置地形的参数。
    - init_state: 设置机器人初始状态的参数。
    - control: 设置机器人控制参数。
    - asset: 设置机器人资源文件的参数。
    - rewards: 设置训练过程中的奖励参数。
    """

    class env(LeggedRobotCfg.env):
        """
        env子类，设置仿真环境相关参数。

        类属性:
        - num_envs: 同时运行的环境数量。
        - num_observations: 每个环境中的观测数量。
        - num_actions: 每个环境中的动作数量。
        """
        num_envs = 4096  # 同时运行的环境数量
        num_observations = 169  # 每个环境中的观测数量
        num_actions = 12  # 每个环境中的动作数量

    class terrain(LeggedRobotCfg.terrain):
        """
        terrain子类，设置地形相关参数。

        类属性:
        - measured_points_x: 地形测量点的x坐标列表。
        - measured_points_y: 地形测量点的y坐标列表。
        """
        # 地形测量点的x坐标和y坐标列表，定义了一个1mx1m的矩形区域（不包括中心线）
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    class init_state(LeggedRobotCfg.init_state):
        """
        为LeggedRobotCfg类定义的init_state子类，用于设置机器人的初始状态参数。

        类属性:
        - pos: 机器人在仿真环境中的初始位置坐标，格式为三维列表[x, y, z]，单位为米。
        - default_joint_angles: 当动作指令为0时各关节的目标角度，是一个关节名称到角度值的映射字典，单位为弧度。

        注意:
        - 关节名称遵循特定的命名约定，如'hip_abduction_left'表示左侧髋部的外展/内收动作。
        - 这些角度提供了机器人在仿真开始时的默认姿势，这有助于确保机器人从一个稳定的状态开始仿真。
        """
        pos = [0.0, 0.0, 1.0]  # 设置机器人初始位置坐标，机器人将在仿真环境中这个位置开始，单位为米

        # 设置动作指令为0时各关节的目标角度，这些角度定义了机器人的默认起始姿势
        default_joint_angles = {
            'hip_abduction_left': 0.1,  # 左侧髋关节外展的目标角度
            'hip_rotation_left': 0.0,  # 左侧髋关节旋转的目标角度
            'hip_flexion_left': 1.0,  # 左侧髋关节屈曲的目标角度
            'thigh_joint_left': -1.8,  # 左侧大腿关节的目标角度
            'ankle_joint_left': 1.57,  # 左侧踝关节的目标角度
            'toe_joint_left': -1.57,  # 左侧脚趾关节的目标角度
            'hip_abduction_right': -0.1,  # 右侧髋关节外展的目标角度（负数表示相反方向）
            'hip_rotation_right': 0.0,  # 右侧髋关节旋转的目标角度
            'hip_flexion_right': 1.0,  # 右侧髋关节屈曲的目标角度
            'thigh_joint_right': -1.8,  # 右侧大腿关节的目标角度
            'ankle_joint_right': 1.57,  # 右侧踝关节的目标角度
            'toe_joint_right': -1.57  # 右侧脚趾关节的目标角度
        }

    class control(LeggedRobotCfg.control):
        """
        control子类，设置机器人控制相关参数。

        类属性:
        - stiffness: PD控制器的刚度参数。
        - damping: PD控制器的阻尼参数。
        - action_scale: 动作缩放比例。
        - decimation: 控制动作更新的频率。
        """
        # PD控制器的刚度参数，单位为牛顿米每弧度
        stiffness = {
            'hip_abduction': 100.0, 'hip_rotation': 100.0,
            'hip_flexion': 200., 'thigh_joint': 200., 'ankle_joint': 200.,
            'toe_joint': 40.
        }
        # PD控制器的阻尼参数，单位为牛顿米秒每弧度
        damping = {
            'hip_abduction': 3.0, 'hip_rotation': 3.0,
            'hip_flexion': 6., 'thigh_joint': 6., 'ankle_joint': 6.,
            'toe_joint': 1.
        }
        action_scale = 0.5  # 动作缩放比例
        decimation = 4  # 控制动作更新的频率

    class asset(LeggedRobotCfg.asset):
        """
        asset子类，设置机器人资源文件相关参数。

        类属性:
        - file: 机器人URDF文件的路径。
        - name: 机器人的名称。
        - foot_name: 机器人脚部的名称。
        - terminate_after_contacts_on: 接触终止适用的部件名称列表。
        - flip_visual_attachments: 是否翻转视觉附件。
        - self_collisions: 自碰撞设置，1为禁用，0为启用。
        """
        # 机器人URDF文件的路径，LEGGED_GYM_ROOT_DIR为环境变量
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cassie/urdf/cassie.urdf'
        name = "cassie"  # 机器人的名称
        foot_name = 'toe'  # 机器人脚部的名称
        # 接触终止适用的部件名称列表
        terminate_after_contacts_on = ['pelvis']
        flip_visual_attachments = False  # 是否翻转视觉附件
        self_collisions = 1  # 自碰撞设置，1为禁用，0为启用

    class rewards(LeggedRobotCfg.rewards):
        """
        rewards子类，设置训练过程中的奖励相关参数。

        类属性:
        - soft_dof_pos_limit: 关节位置软限制。
        - soft_dof_vel_limit: 关节速度软限制。
        - soft_torque_limit: 扭矩软限制。
        - max_contact_force: 最大接触力。
        - only_positive_rewards: 是否只有正奖励。
        - scales: 奖励的缩放比例。
        """
        soft_dof_pos_limit = 0.95  # 关节位置软限制
        soft_dof_vel_limit = 0.9  # 关节速度软限制
        soft_torque_limit = 0.9  # 扭矩软限制
        max_contact_force = 300.  # 最大接触力，单位为牛顿
        only_positive_rewards = False  # 是否只有正奖励

        class scales(LeggedRobotCfg.reward_scales):
            """
            scales子类，用于设置奖励的缩放比例。

            类属性:
            - termination: 终止奖励。
            - tracking_ang_vel: 角速度跟踪奖励。
            - torques: 扭矩奖励。
            - dof_acc: 关节加速度奖励。
            - lin_vel_z: z轴线速度奖励。
            - feet_air_time: 脚部空中时间奖励。
            - dof_pos_limits: 关节位置限制奖励。
            - no_fly: 非飞行奖励。
            - dof_vel: 关节速度奖励。
            - ang_vel_xy: xy平面角速度奖励。
            - feet_contact_forces: 脚部接触力奖励。
            """
            termination = -200.  # 终止奖励
            tracking_ang_vel = 1.0  # 角速度跟踪奖励
            torques = -5.e-6  # 扭矩奖励
            dof_acc = -2.e-7  # 关节加速度奖励
            lin_vel_z = -0.5  # z轴线速度奖励
            feet_air_time = 5.  # 脚部空中时间奖励
            dof_pos_limits = -1.  # 关节位置限制奖励
            no_fly = 0.25  # 非飞行奖励
            dof_vel = -0.0  # 关节速度奖励
            ang_vel_xy = -0.0  # xy平面角速度奖励
            feet_contact_forces = -0.  # 脚部接触力奖励
            
class CassieRoughCfgPPO(LeggedRobotCfgPPO):
    """
    CassieRoughCfgPPO类继承自LeggedRobotCfgPPO，用于配置Cassie机器人使用PPO算法在粗糙地形上的训练运行参数。

    类属性:
    - runner: 设置训练运行的参数。
    - algorithm: 设置PPO算法的参数。
    """

    class runner(LeggedRobotCfgPPO.runner):
        """
        runner子类，设置训练运行相关参数。

        类属性:
        - run_name: 训练运行的名称。
        - experiment_name: 实验名称。
        """
        run_name = ''  # 训练运行的名称，通常用于区分不同的训练实验
        experiment_name = 'rough_cassie'  # 实验名称，用于标识当前训练的环境或任务

    class algorithm(LeggedRobotCfgPPO.algorithm):
        """
        algorithm子类，设置PPO算法相关参数。

        类属性:
        - entropy_coef: 熵系数，用于鼓励探索。
        """
        entropy_coef = 0.01  # 熵系数，用于鼓励探索，通常有助于提高训练过程中的策略多样性
