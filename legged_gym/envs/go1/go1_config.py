from typing import Union

from params_proto import Meta

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg as Cfg


def config_go1(Cnfg: Union[Cfg, Meta]):
    # 配置 go1 实体的函数。接受 Cnfg 参数，可以是 Cfg 或 Meta 类型。

    # 配置初始状态
    _ = Cnfg.init_state
    _.pos = [0.0, 0.0, 0.34]  # 初始位置 x,y,z（米）
    _.default_joint_angles = {  # 默认关节角度（弧度），当动作为 0.0 时的目标角度
        'FL_hip_joint': 0.1, 'RL_hip_joint': 0.1,  # 前左和后左髋关节
        'FR_hip_joint': -0.1, 'RR_hip_joint': -0.1,  # 前右和后右髋关节
        'FL_thigh_joint': 0.8, 'RL_thigh_joint': 1.,  # 前左和后左大腿关节
        'FR_thigh_joint': 0.8, 'RR_thigh_joint': 1.,  # 前右和后右大腿关节
        'FL_calf_joint': -1.5, 'RL_calf_joint': -1.5,  # 前左和后左小腿关节
        'FR_calf_joint': -1.5, 'RR_calf_joint': -1.5  # 前右和后右小腿关节
    }

    # 配置控制参数
    _ = Cnfg.control
    _.control_type = 'P'  # 控制类型（P表示比例控制）
    _.stiffness = {'joint': 20.}  # 关节刚度（牛顿米/弧度）
    _.damping = {'joint': 0.5}  # 关节阻尼（牛顿米秒/弧度）
    _.action_scale = 0.25  # 动作缩放：目标角度 = 动作缩放 * 动作 + 默认角度
    _.hip_scale_reduction = 0.5  # 髋关节缩放减少
    _.decimation = 4  # 控制动作更新的减频率

    # 配置资产（如URDF文件）
    _ = Cnfg.asset
    _.file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'  # URDF文件路径
    _.name = 'go1'
    _.foot_name = "foot"  # 脚部名称
    _.penalize_contacts_on = ["thigh", "calf"]  # 联系处罚部位
    _.terminate_after_contacts_on = ["base"]  # 联系终止部位
    _.self_collisions = 0  # 自身碰撞（1禁用，0启用）
    _.flip_visual_attachments = False  # 翻转视觉附件
    _.fix_base_link = False  # 固定基座链接

    # 配置奖励机制
    _ = Cnfg.rewards
    _.soft_dof_pos_limit = 0.9  # 软关节位置限制
    _.base_height_target = 0.34  # 基座高度目标

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
    _.measure_heights = True  # 测量高度
    _.terrain_noise_magnitude = 0.1  # 地形噪声幅度
    _.teleport_robots = True  # 传送机器人
    _.border_size = 50  # 边界大小
    _.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]  # 地形比例
    _.curriculum = True  # 课程制

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
