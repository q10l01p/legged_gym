from typing import Union
from params_proto import Meta
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg as Cfg


def config_a1(Cnfg: Union[Cfg, Meta]):

    # 配置初始状态
    _ = Cnfg.init_state
    _.pos = [0.0, 0.0, 0.42]      # x,y,z坐标的起始位置，单位为米（m），与案例代码相比增加了高度
    _.default_joint_angles = {    # 当动作值为0.0时的目标关节角度，默认单位为弧度（rad）
        'FL_hip_joint': 0.1,    # 前左髋关节的默认角度
        'RL_hip_joint': 0.1,    # 后左髋关节的默认角度
        'FR_hip_joint': -0.1,   # 前右髋关节的默认角度
        'RR_hip_joint': -0.1,   # 后右髋关节的默认角度

        'FL_thigh_joint': 0.8,  # 前左大腿关节的默认角度
        'RL_thigh_joint': 1.0,  # 后左大腿关节的默认角度
        'FR_thigh_joint': 0.8,  # 前右大腿关节的默认角度
        'RR_thigh_joint': 1.0,  # 后右大腿关节的默认角度

        'FL_calf_joint': -1.5,  # 前左小腿关节的默认角度
        'RL_calf_joint': -1.5,  # 后左小腿关节的默认角度
        'FR_calf_joint': -1.5,  # 前右小腿关节的默认角度
        'RR_calf_joint': -1.5,  # 后右小腿关节的默认角度
    }

    # 配置控制参数
    _ = Cnfg.control
    _.control_type = 'P'          # 控制类型，这里设置为比例控制（P）
    _.stiffness = {'joint': 20.}  # 关节的刚度，单位为牛顿米每弧度（N*m/rad）
    _.damping = {'joint': 0.5}    # 关节的阻尼，单位为牛顿米秒每弧度（N*m*s/rad）
    _.action_scale = 0.25         # 动作比例因子，目标角度 = 动作比例因子 * 动作 + 默认角度
    _.decimation = 4              # 降采样因子，仿真时间步长内策略时间步长的控制动作更新次数

    # 配置资产（如URDF文件）
    _ = Cnfg.asset
    _.file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'  # URDF文件的路径，适应A1机器人
    _.name = 'a1'
    _.foot_name = "foot"                          # 足部的名称
    _.penalize_contacts_on = ["thigh", "calf"]    # 在大腿和小腿上的接触会受到惩罚
    _.terminate_after_contacts_on = ["base"]      # 在基座上的接触会导致仿真终止
    _.self_collisions = 0                         # 自碰撞设置，1表示禁用，0表示启用

    # 配置奖励机制
    _ = Cnfg.rewards
    _.soft_dof_pos_limit = 0.9    # 软关节位置限制的奖励因子，与案例代码一致
    _.base_height_target = 0.25   # 基座的目标高度，单位为米（m），这是与案例代码的区别之一，更适合A1的配置

    # 配置奖励缩放
    _ = Cnfg.reward_scales
    _.torques = -0.0001  # 扭矩
    _.action_rate = -0.01  # 动作率
    _.dof_pos_limits = -10.0  # 关节位置限制
    _.orientation = -5.  # 方向
    _.base_height = -30.  # 基座高度

    # 配置地形
    _ = Cnfg.terrain
    _.mesh_type = 'trimesh'  # 网格类型
    _.measure_heights = False  # 测量高度
    _.terrain_noise_magnitude = 0.0  # 地形噪声幅度
    _.teleport_robots = True  # 传送机器人
    _.border_size = 50  # 边界大小
    _.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]  # 地形比例
    _.curriculum = False  # 课程制

    # 配置环境
    _ = Cnfg.env
    _.num_observations = 42  # 观察数量
    _.observe_vel = False  # 观察速度
    _.num_envs = 4000  # 环境数量

    # 配置命令
    _ = Cnfg.commands
    _.lin_vel_x = [-1.0, 1.0]  # 线速度X范围
    _.lin_vel_y = [-1.0, 1.0]  # 线速度Y范围
    _.heading_command = False  # 航向命令
    _.resampling_time = 10.0  # 重采样时间
    _.command_curriculum = True  # 命令课程制
    _.num_lin_vel_bins = 30  # 线速度分箱数
    _.num_ang_vel_bins = 30  # 角速度分箱数
    _.lin_vel_x = [-0.6, 0.6]  # 线速度X范围
    _.lin_vel_y = [-0.6, 0.6]  # 线速度Y范围
    _.ang_vel_yaw = [-1, 1]  # 角速度偏航范围

    # 配置域随机化
    _ = Cnfg.domain_rand
    _.randomize_base_mass = True  # 随机化基座质量
    _.added_mass_range = [-1, 3]  # 添加质量范围
    _.push_robots = False  # 推动机器人
    _.max_push_vel_xy = 0.5  # 最大推动速度
    _.randomize_friction = True  # 随机化摩擦
    _.friction_range = [0.05, 4.5]  # 摩擦范围
    _.randomize_restitution = True  # 随机化弹性恢复
    _.restitution_range = [0.0, 1.0]  # 弹性恢复范围
    _.restitution = 0.5  # 默认地形弹性恢复
    _.randomize_com_displacement = True  # 随机化质心位移
    _.com_displacement_range = [-0.1, 0.1]  # 质心位移范围
    _.randomize_motor_strength = True  # 随机化电机强度
    _.motor_strength_range = [0.9, 1.1]  # 电机强度范围
    _.randomize_Kp_factor = False  # 随机化比例增益因子
    _.Kp_factor_range = [0.8, 1.3]  # 比例增益因子范围
    _.randomize_Kd_factor = False  # 随机化微分增益因子
    _.Kd_factor_range = [0.5, 1.5]  # 微分增益因子范围
    _.rand_interval_s = 6  # 随机化间隔（秒）


