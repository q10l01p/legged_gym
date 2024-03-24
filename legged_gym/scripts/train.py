import isaacgym
assert isaacgym

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg as Cfg
from legged_gym.envs import config_go1, config_a1
from legged_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from ml_logger import logger
from legged_gym.envs.wrappers.history_wrapper import HistoryWrapper

from legged_gym.algorithms.ppo import Runner
from legged_gym.algorithms.ppo.actor_critic import AC_Args
from legged_gym.algorithms.ppo.ppo import PPO_Args
from legged_gym.algorithms.ppo import RunnerArgs

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def cfg():
    # 初始化配置，使用名为Cfg的配置对象
    config_go1(Cfg)
    # config_a1(Cfg)

    Cfg.env.num_envs = 4096  # 设置环境数量为3000

    Cfg.commands.num_lin_vel_bins = 30  # 设置线性速度的分割区间数量为30
    Cfg.commands.num_ang_vel_bins = 30  # 设置角速度的分割区间数量为30
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7  # 设置跟踪角速度的阈值为0.7
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8  # 设置跟踪线性速度的阈值为0.8
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90  # 设置跟踪接触速度的阈值为0.90
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90  # 设置跟踪接触力的阈值为0.90

    Cfg.commands.distributional_commands = True  # 启用分布式命令

    Cfg.domain_rand.lag_timesteps = 6  # 设置延迟时间步数为6
    Cfg.domain_rand.randomize_lag_timesteps = True  # 启用延迟时间步数的随机化
    Cfg.control.control_type = "actuator_net"  # 设置控制类型为"actuator_net"

    Cfg.domain_rand.randomize_rigids_after_start = False  # 启动后不随机化刚体
    Cfg.env.priv_observe_motion = False  # 不观察运动
    Cfg.env.priv_observe_gravity_transformed_motion = False  # 不观察受重力影响的运动
    Cfg.domain_rand.randomize_friction_indep = False  # 不随机化独立摩擦
    Cfg.env.priv_observe_friction_indep = False  # 不观察独立摩擦
    Cfg.domain_rand.randomize_friction = True  # 随机化摩擦
    Cfg.env.priv_observe_friction = True  # 观察摩擦
    Cfg.domain_rand.friction_range = [0.1, 3.0]  # 设置摩擦范围为0.1到3.0
    Cfg.domain_rand.randomize_restitution = True  # 随机化恢复系数
    Cfg.env.priv_observe_restitution = True  # 观察恢复系数
    Cfg.domain_rand.restitution_range = [0.0, 0.4]  # 设置恢复系数范围为0.0到0.4
    Cfg.domain_rand.randomize_base_mass = True  # 随机化基础质量
    Cfg.env.priv_observe_base_mass = False  # 不观察基础质量
    Cfg.domain_rand.added_mass_range = [-1.0, 3.0]  # 设置额外质量范围为-1.0到3.0
    Cfg.domain_rand.randomize_gravity = True  # 随机化重力
    Cfg.domain_rand.gravity_range = [-1.0, 1.0]  # 设置重力范围为-1.0到1.0
    Cfg.domain_rand.gravity_rand_interval_s = 8.0  # 设置重力随机化间隔为8秒
    Cfg.domain_rand.gravity_impulse_duration = 0.99  # 设置重力冲量持续时间为0.99秒
    Cfg.env.priv_observe_gravity = False  # 不观察重力
    Cfg.domain_rand.randomize_com_displacement = False  # 不随机化质心位移
    Cfg.domain_rand.com_displacement_range = [-0.15, 0.15]  # 设置质心位移范围为-0.15到0.15
    Cfg.env.priv_observe_com_displacement = False  # 不观察质心位移
    Cfg.domain_rand.randomize_ground_friction = True  # 随机化地面摩擦
    Cfg.env.priv_observe_ground_friction = False  # 不观察地面摩擦
    Cfg.env.priv_observe_ground_friction_per_foot = False  # 不逐足观察地面摩擦
    Cfg.domain_rand.ground_friction_range = [0.0, 0.0]  # 设置地面摩擦范围为0.0到0.0
    Cfg.domain_rand.randomize_motor_strength = True  # 随机化电机强度
    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]  # 设置电机强度范围为0.9到1.1
    Cfg.env.priv_observe_motor_strength = False  # 不观察电机强度
    Cfg.domain_rand.randomize_motor_offset = True  # 随机化电机偏移
    Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]  # 设置电机偏移范围为-0.02到0.02
    Cfg.env.priv_observe_motor_offset = False  # 不观察电机偏移
    Cfg.domain_rand.push_robots = False  # 不推动机器人
    Cfg.domain_rand.randomize_Kp_factor = False  # 不随机化比例增益因子Kp
    Cfg.env.priv_observe_Kp_factor = False  # 不观察比例增益因子Kp
    Cfg.domain_rand.randomize_Kd_factor = False  # 不随机化微分增益因子Kd
    Cfg.env.priv_observe_Kd_factor = False  # 不观察微分增益因子Kd
    Cfg.env.priv_observe_body_velocity = False  # 不观察身体速度
    Cfg.env.priv_observe_body_height = False  # 不观察身体高度
    Cfg.env.priv_observe_desired_contact_states = False  # 不观察期望的接触状态
    Cfg.env.priv_observe_contact_forces = False  # 不观察接触力
    Cfg.env.priv_observe_foot_displacement = False  # 不观察足部位移
    Cfg.env.priv_observe_gravity_transformed_foot_displacement = False  # 不观察受重力影响的足部位移
    Cfg.env.priv_observe_heights = True  # 观察地形高度

    Cfg.env.num_privileged_obs = 2 + 36  # 设置特权观测的数量为2
    Cfg.env.num_observation_history = 1  # 设置观测历史的数量为30
    Cfg.reward_scales.feet_contact_forces = 0.0  # 设置脚部接触力的奖励比例为0.0

    Cfg.domain_rand.rand_interval_s = 4  # 设置领域随机化的间隔时间为4秒
    Cfg.commands.num_commands = 15  # 设置命令的数量为15
    Cfg.env.observe_two_prev_actions = True  # 设置环境观测前两个动作
    Cfg.env.observe_yaw = False  # 设置环境不观测偏航角
    Cfg.env.num_observations = 70  # 设置环境的观测数量为70
    Cfg.env.num_scalar_observations = 70  # 设置标量观测的数量为70
    Cfg.env.observe_gait_commands = True  # 设置环境观测步态命令
    Cfg.env.observe_timing_parameter = False  # 设置环境不观测时间参数
    Cfg.env.observe_clock_inputs = True  # 设置环境观测时钟输入

    Cfg.domain_rand.tile_height_range = [-0.0, 0.0]  # 设置地砖高度范围为-0.0到0.0
    Cfg.domain_rand.tile_height_curriculum = True  # 设置地砖高度不使用课程学习
    Cfg.domain_rand.tile_height_update_interval = 1000000  # 设置地砖高度更新间隔为1000000步
    Cfg.domain_rand.tile_height_curriculum_step = 0.01  # 设置地砖高度课程学习步长为0.01
    Cfg.terrain.border_size = 10.0  # 设置地形边界大小为0.0
    Cfg.terrain.mesh_type = "trimesh"  # 设置地形的网格类型为三角网格
    Cfg.terrain.num_cols = 20  # 设置地形的列数为30
    Cfg.terrain.num_rows = 20  # 设置地形的行数为30
    Cfg.terrain.terrain_width = 5.0  # 设置地形的宽度为5.0
    Cfg.terrain.terrain_length = 5.0  # 设置地形的长度为5.0
    Cfg.terrain.x_init_range = 0.2  # 设置地形X方向初始范围为0.2
    Cfg.terrain.y_init_range = 0.2  # 设置地形Y方向初始范围为0.2
    Cfg.terrain.teleport_thresh = 0.3  # 设置地形传送阈值为0.3
    Cfg.terrain.teleport_robots = True  # 设置不传送机器人
    Cfg.terrain.center_robots = False  # 设置将机器人置于中心
    Cfg.terrain.center_span = 4  # 设置中心区域范围为4
    Cfg.terrain.horizontal_scale = 0.10  # 设置地形的水平缩放为0.10
    Cfg.rewards.use_terminal_foot_height = False  # 设置不使用脚部高度作为终止奖励
    Cfg.rewards.use_terminal_body_height = False  # 设置使用身体高度作为终止奖励
    Cfg.rewards.terminal_body_height = 0.05  # 设置终止时的身体高度奖励阈值为0.05
    Cfg.rewards.use_terminal_roll_pitch = True  # 设置使用终止时的滚动和俯仰作为奖励
    Cfg.rewards.terminal_body_ori = 1.6  # 设置终止时身体方向的奖励阈值为1.6

    Cfg.commands.resampling_time = 10  # 设置命令重采样时间为10

    # 奖励和惩罚的比例设置
    Cfg.reward_scales.feet_slip = -0.04  # 脚部滑动的惩罚比例
    Cfg.reward_scales.action_smoothness_1 = -0.1  # 动作平滑度1的惩罚比例
    Cfg.reward_scales.action_smoothness_2 = -0.1  # 动作平滑度2的惩罚比例
    Cfg.reward_scales.dof_vel = -1e-4  # 自由度速度的惩罚比例
    Cfg.reward_scales.dof_pos = -0.0  # 自由度位置的惩罚比例
    Cfg.reward_scales.jump = 10.0  # 跳跃的奖励比例
    Cfg.reward_scales.base_height = 0.0  # 基础高度的奖励比例
    Cfg.rewards.base_height_target = 0.30  # 目标基础高度为0.30
    Cfg.reward_scales.estimation_bonus = 0.0  # 估计奖金的比例
    Cfg.reward_scales.raibert_heuristic = -10.0  # Raibert启发式的惩罚比例
    Cfg.reward_scales.feet_impact_vel = -0.0  # 脚部冲击速度的惩罚比例
    Cfg.reward_scales.feet_clearance = -0.0  # 脚部间隙的惩罚比例
    Cfg.reward_scales.feet_clearance_cmd = -0.0  # 脚部间隙命令的惩罚比例
    Cfg.reward_scales.feet_clearance_cmd_linear = -30.0  # 脚部间隙线性命令的惩罚比例
    Cfg.reward_scales.orientation = 0.0  # 方向的奖励比例
    Cfg.reward_scales.orientation_control = -5.0  # 方向控制的惩罚比例
    Cfg.reward_scales.tracking_stance_width = -0.0  # 跟踪站姿宽度的惩罚比例
    Cfg.reward_scales.tracking_stance_length = -0.0  # 跟踪站姿长度的惩罚比例
    Cfg.reward_scales.lin_vel_z = -0.02  # Z轴线性速度的惩罚比例
    Cfg.reward_scales.ang_vel_xy = -0.001  # XY轴角速度的惩罚比例
    Cfg.reward_scales.feet_air_time = 0.0  # 脚部空中时间的奖励比例
    Cfg.reward_scales.hop_symmetry = 0.0  # 跳跃对称性的奖励比例
    Cfg.rewards.kappa_gait_probs = 0.07  # 步态概率的kappa值
    Cfg.rewards.gait_force_sigma = 100.  # 步态力sigma值
    Cfg.rewards.gait_vel_sigma = 10.  # 步态速度sigma值
    Cfg.reward_scales.tracking_contacts_shaped_force = 4.0  # 跟踪接触形状力的奖励比例
    Cfg.reward_scales.tracking_contacts_shaped_vel = 4.0  # 跟踪接触形状速度的奖励比例
    Cfg.reward_scales.collision = -5.0  # 碰撞的惩罚比例

    # 奖励容器和奖励策略设置
    Cfg.rewards.reward_container_name = "CoRLRewards"  # 奖励容器名称
    Cfg.rewards.only_positive_rewards = False  # 是否只有正的奖励
    Cfg.rewards.only_positive_rewards_ji22_style = True  # 是否采用ji22风格的仅正奖励策略
    Cfg.rewards.sigma_rew_neg = 0.02  # 负奖励的sigma值

    # 设置命令参数的范围
    Cfg.commands.lin_vel_x = [-1.0, 1.0]  # 线性速度X的范围
    Cfg.commands.lin_vel_y = [-0.6, 0.6]  # 线性速度Y的范围
    Cfg.commands.ang_vel_yaw = [-1.0, 1.0]  # 角速度偏航的范围
    Cfg.commands.body_height_cmd = [-0.25, 0.15]  # 身体高度命令的范围
    Cfg.commands.gait_frequency_cmd_range = [2.0, 4.0]  # 步态频率命令的范围
    Cfg.commands.gait_phase_cmd_range = [0.0, 1.0]  # 步态相位命令的范围
    Cfg.commands.gait_offset_cmd_range = [0.0, 1.0]  # 步态偏移命令的范围
    Cfg.commands.gait_bound_cmd_range = [0.0, 1.0]  # 步态边界命令的范围
    Cfg.commands.gait_duration_cmd_range = [0.5, 0.5]  # 步态持续时间命令的范围
    Cfg.commands.footswing_height_range = [0.03, 0.35]  # 脚摆高度的范围
    Cfg.commands.body_pitch_range = [-0.4, 0.4]  # 身体俯仰的范围
    Cfg.commands.body_roll_range = [-0.0, 0.0]  # 身体翻滚的范围
    Cfg.commands.stance_width_range = [0.10, 0.45]  # 站姿宽度的范围
    Cfg.commands.stance_length_range = [0.35, 0.45]  # 站姿长度的范围

    # 设置命令参数的限制
    Cfg.commands.limit_vel_x = [-5.0, 5.0]  # 线性速度X的限制
    Cfg.commands.limit_vel_y = [-0.6, 0.6]  # 线性速度Y的限制
    Cfg.commands.limit_vel_yaw = [-5.0, 5.0]  # 角速度偏航的限制
    Cfg.commands.limit_body_height = [-0.25, 0.15]  # 身体高度的限制
    Cfg.commands.limit_gait_frequency = [2.0, 4.0]  # 步态频率的限制
    Cfg.commands.limit_gait_phase = [0.0, 1.0]  # 步态相位的限制
    Cfg.commands.limit_gait_offset = [0.0, 1.0]  # 步态偏移的限制
    Cfg.commands.limit_gait_bound = [0.0, 1.0]  # 步态边界的限制
    Cfg.commands.limit_gait_duration = [0.5, 0.5]  # 步态持续时间的限制
    Cfg.commands.limit_footswing_height = [0.03, 0.35]  # 脚摆高度的限制
    Cfg.commands.limit_body_pitch = [-0.4, 0.4]  # 身体俯仰的限制
    Cfg.commands.limit_body_roll = [-0.0, 0.0]  # 身体翻滚的限制
    Cfg.commands.limit_stance_width = [0.10, 0.45]  # 站姿宽度的限制
    Cfg.commands.limit_stance_length = [0.35, 0.45]  # 站姿长度的限制

    # 配置命令的分箱数量
    Cfg.commands.num_bins_vel_x = 21  # 线性速度X的分箱数量
    Cfg.commands.num_bins_vel_y = 1  # 线性速度Y的分箱数量
    Cfg.commands.num_bins_vel_yaw = 21  # 角速度偏航的分箱数量
    Cfg.commands.num_bins_body_height = 1  # 身体高度的分箱数量
    Cfg.commands.num_bins_gait_frequency = 1  # 步态频率的分箱数量
    Cfg.commands.num_bins_gait_phase = 1  # 步态相位的分箱数量
    Cfg.commands.num_bins_gait_offset = 1  # 步态偏移的分箱数量
    Cfg.commands.num_bins_gait_bound = 1  # 步态边界的分箱数量
    Cfg.commands.num_bins_gait_duration = 1  # 步态持续时间的分箱数量
    Cfg.commands.num_bins_footswing_height = 1  # 脚摆高度的分箱数量
    Cfg.commands.num_bins_body_roll = 1  # 身体翻滚的分箱数量
    Cfg.commands.num_bins_body_pitch = 1  # 身体俯仰的分箱数量
    Cfg.commands.num_bins_stance_width = 1  # 站姿宽度的分箱数量

    # 标准化和地形参数设置
    Cfg.normalization.friction_range = [0, 1]  # 摩擦系数的标准化范围
    Cfg.normalization.ground_friction_range = [0, 1]  # 地面摩擦系数的标准化范围
    Cfg.terrain.yaw_init_range = 3.14  # 初始偏航角范围
    Cfg.normalization.clip_actions = 10.0  # 动作剪辑值

    # 命令参数的其他配置
    Cfg.commands.exclusive_phase_offset = False  # 独占相位偏移关闭
    Cfg.commands.pacing_offset = False  # 步速偏移关闭
    Cfg.commands.binary_phases = True  # 二进制相位开启
    Cfg.commands.gaitwise_curricula = True  # 步态课程学习开启


def train(headless=False):
    # 初始化仿真环境
    env = VelocityTrackingEasyEnv(sim_device='cuda', headless=True, cfg=Cfg)
    # 创建VelocityTrackingEasyEnv环境实例，指定模拟设备为CUDA（GPU加速），headless为True（无头模式，不显示界面），并传入配置Cfg

    # 记录实验参数
    logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                       Cfg=vars(Cfg))
    # 使用logger记录不同组件的参数。AC_Args, PPO_Args, 和 RunnerArgs 可能是与学习算法相关的不同配置对象。
    # vars() 函数用于获取对象的属性和属性值，类似于转换为字典。

    # 环境包装
    env = HistoryWrapper(env)
    # 对环境应用历史包装器（HistoryWrapper），这可能是为了增加历史信息的追踪能力，有助于学习算法更好地理解环境的动态。

    # 学习过程的初始化
    runner = Runner(env, device="cuda")
    # 创建Runner对象，用于执行学习过程。Runner可能是一个封装了学习算法的类，这里指定运行设备为第一个GPU。

    # 启动学习过程
    runner.learn(num_learning_iterations=100000, init_at_random_ep_len=True, eval_freq=1000, cfg=Cfg)


if __name__ == '__main__':
    # 导入所需的模块。Path用于文件路径操作，logger用于日志记录，MINI_GYM_ROOT_DIR可能是一个定义好的环境根目录常量。
    from pathlib import Path
    from ml_logger import logger
    from legged_gym import MINI_GYM_ROOT_DIR

    cfg()

    # 获取当前脚本文件的基础名（不包含后缀）。
    stem = Path(__file__).stem
    # 配置logger，设置日志记录的格式和存储位置。使用当前时间和脚本基础名作为日志文件的一部分路径。
    logger.configure(logger.utcnow(f'gait-conditioned-agility/{Cfg.asset.name}/%Y-%m-%d/{stem}/%H%M%S.%f'),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
    # 使用logger记录一段文本，这段文本可能是用于配置图表或报告的YAML格式内容。dedent=True表明去除文本前面的空白。
    logger.log_text("""
                charts: # 图表配置列表
                - yKey: train/episode/rew_total/mean # 平均总奖励
                  xKey: iterations # 迭代次数
                - yKey: train/episode/rew_tracking_lin_vel/mean # 跟踪线速度的平均奖励
                  xKey: iterations # 迭代次数
                - yKey: train/episode/rew_tracking_contacts_shaped_force/mean # 根据接触力调整的平均奖励
                  xKey: iterations # 迭代次数
                - yKey: train/episode/rew_action_smoothness_1/mean # 动作平滑度1的平均奖励
                  xKey: iterations # 迭代次数
                - yKey: train/episode/rew_action_smoothness_2/mean # 动作平滑度2的平均奖励
                  xKey: iterations # 迭代次数
                - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean # 根据接触速度调整的平均奖励
                  xKey: iterations # 迭代次数
                - yKey: train/episode/rew_orientation_control/mean # 方向控制的平均奖励
                  xKey: iterations # 迭代次数
                - yKey: train/episode/command_area_trot/mean # 指挥区域跑步的平均奖励
                  xKey: iterations # 迭代次数
                - yKey: adaptation_loss/mean # 适应性损失的平均值
                  xKey: iterations # 迭代次数
                - yKey: student_loss/mean # 学生策略损失的平均值
                  xKey: iterations # 迭代次数
                - type: video # 视频类型
                  glob: "videos/*.mp4" # 视频文件匹配模式
                """, filename=".charts.yml", dedent=True)  # 将文本记录到.charts.yml文件，自动减少缩进

    # 调用train_go1函数开始训练过程。参数headless=False表明在训练时会显示环境的渲染（即可视化训练过程）。
    train(headless=False)
