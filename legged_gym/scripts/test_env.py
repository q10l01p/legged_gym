import isaacgym

assert isaacgym
import matplotlib.pyplot as plt
import torch
from tqdm import trange

from go1_gym.envs import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg as Cfg
from legged_gym.envs import config_go1, config_a1
from legged_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv


def run_env(render=False, headless=False):
    # 定义一个函数run_env，接受两个参数：render和headless。这些参数控制环境的视觉渲染和无头模式。

    # 准备环境配置
    config_go1(Cfg)
    # 调用函数config_go1来配置环境，使用Cfg作为配置对象。

    # 配置线性速度和角速度的离散分布数量
    Cfg.commands.num_lin_vel_bins = 30
    # 设置线速度的离散分布数量为30个区间。

    Cfg.commands.num_ang_vel_bins = 30
    # 设置角速度的离散分布数量为30个区间。

    # 设置跟踪速度和接触力的阈值
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    # 设置角速度跟踪的阈值为0.7。

    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    # 设置线速度跟踪的阈值为0.8。

    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.9
    # 设置接触速度的阈值为0.9。

    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.9
    # 设置接触力的阈值为0.9。

    # 启用分布式命令
    Cfg.commands.distributional_commands = True
    # 设置分布式命令为激活状态。

    # 配置观察参数和环境随机化
    Cfg.env.priv_observe_motion = False
    # 设置观察动作为非私有。

    Cfg.env.priv_observe_gravity_transformed_motion = True
    # 设置观察经重力转换的动作为私有。

    Cfg.domain_rand.randomize_friction_indep = False
    # 关闭独立摩擦力的随机化。

    Cfg.env.priv_observe_friction_indep = False
    # 设置独立摩擦力的观察为非私有。

    Cfg.domain_rand.randomize_friction = True
    # 开启摩擦力的随机化。

    Cfg.env.priv_observe_friction = False
    # 设置摩擦力的观察为非私有。

    Cfg.domain_rand.friction_range = [0.0, 0.0]
    # 设置摩擦力的随机化范围。

    Cfg.domain_rand.randomize_restitution = True
    # 开启恢复系数的随机化。

    Cfg.env.priv_observe_restitution = False
    # 设置恢复系数的观察为非私有。

    Cfg.domain_rand.restitution_range = [0.0, 1.0]
    # 设置恢复系数的随机化范围。

    Cfg.domain_rand.randomize_base_mass = True
    # 开启基础质量的随机化。

    Cfg.env.priv_observe_base_mass = False
    # 设置基础质量的观察为非私有。

    Cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    # 设置添加质量的随机化范围。

    Cfg.domain_rand.randomize_gravity = True
    # 开启重力的随机化。

    Cfg.domain_rand.gravity_range = [-1.0, 1.0]
    # 设置重力的随机化范围。

    Cfg.domain_rand.gravity_rand_interval_s = 2.0
    # 设置重力随机化的时间间隔。

    Cfg.domain_rand.gravity_impulse_duration = 0.5
    # 设置重力冲击的持续时间。

    Cfg.env.priv_observe_gravity = True
    # 设置重力的观察为私有。

    Cfg.domain_rand.randomize_com_displacement = False
    # 关闭质心位移的随机化。

    Cfg.domain_rand.com_displacement_range = [-0.15, 0.15]
    # 设置质心位移的随机化范围。

    Cfg.env.priv_observe_com_displacement = False
    # 设置质心位移的观察为非私有。

    Cfg.domain_rand.randomize_ground_friction = True
    # 开启地面摩擦力的随机化。

    Cfg.env.priv_observe_ground_friction = False
    # 设置地面摩擦力的观察为非私有。

    Cfg.env.priv_observe_ground_friction_per_foot = False
    # 设置每只脚的地面摩擦力观察为非私有。

    Cfg.domain_rand.ground_friction_range = [0.3, 2.0]
    # 设置地面摩擦力的随机化范围。

    Cfg.domain_rand.randomize_motor_strength = True
    # 开启电机强度的随机化。

    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    # 设置电机强度的随机化范围。

    Cfg.env.priv_observe_motor_strength = False
    # 设置电机强度的观察为非私有。

    Cfg.domain_rand.randomize_motor_offset = True
    # 开启电机偏移的随机化。

    Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    # 设置电机偏移的随机化范围。

    Cfg.env.priv_observe_motor_offset = False
    # 设置电机偏移的观察为非私有。

    # TODO: 是否推动机器人
    Cfg.domain_rand.push_robots = True
    # 设置是否推动机器人。

    Cfg.domain_rand.randomize_Kp_factor = False
    # 关闭比例增益Kp因子的随机化。

    Cfg.env.priv_observe_Kp_factor = False
    # 设置比例增益Kp因子的观察为非私有。

    Cfg.domain_rand.randomize_Kd_factor = False
    # 关闭微分增益Kd因子的随机化。

    Cfg.env.priv_observe_Kd_factor = False
    # 设置微分增益Kd因子的观察为非私有。

    Cfg.env.priv_observe_body_velocity = False
    # 设置身体速度的观察为非私有。

    Cfg.env.priv_observe_body_height = False
    # 设置身体高度的观察为非私有。

    Cfg.env.priv_observe_desired_contact_states = False
    # 设置期望的接触状态的观察为非私有。

    Cfg.env.priv_observe_contact_forces = False
    # 设置接触力的观察为非私有。

    Cfg.env.priv_observe_foot_displacement = False
    # 设置脚的位移观察为非私有。

    # 配置其他环境参数
    Cfg.env.num_privileged_obs = 3
    # 设置私有观察数量为3。

    Cfg.env.num_observation_history = 30
    # 设置观察历史的数量为30。

    Cfg.reward_scales.feet_contact_forces = 0.0
    # 设置脚接触力的奖励比例为0。

    Cfg.domain_rand.rand_interval_s = 4
    # 设置环境随机化参数的更新间隔为4秒。

    Cfg.commands.num_commands = 15
    # 设置仿真环境中可用的命令数量为15。

    Cfg.env.observe_two_prev_actions = True
    # 设置仿真环境以观察前两个动作，这有助于理解动作之间的关系。

    Cfg.env.observe_yaw = True
    # 设置仿真环境以观察偏航角，这是旋转运动的一个方面。

    Cfg.env.num_observations = 71
    # 设置仿真环境中观察的总数为71。

    Cfg.env.num_scalar_observations = 71
    # 设置仿真环境中标量观察的数量也为71。

    Cfg.env.observe_gait_commands = True
    # 设置仿真环境以观察步态命令，这涉及到机器人或模型的移动方式。

    Cfg.env.observe_timing_parameter = False
    # 设置仿真环境不观察时间参数，这可能涉及到动作或事件的定时。

    Cfg.env.observe_clock_inputs = True
    # 设置仿真环境以观察时钟输入，这有助于同步或定时事件。

    Cfg.domain_rand.tile_height_range = [-0.0, 0.0]
    # 设置地面瓦片高度的随机化范围，这里设置为不变。

    Cfg.domain_rand.tile_height_curriculum = False
    # 设置是否使用地面瓦片高度的逐步学习（课程学习），这里设置为不使用。

    Cfg.domain_rand.tile_height_update_interval = 1000, 3000
    # 设置地面瓦片高度更新的间隔时间范围。

    Cfg.domain_rand.tile_height_curriculum_step = 0.01
    # 设置地面瓦片高度课程学习的步长。

    Cfg.terrain.border_size = 0.0
    # 设置仿真环境中地形边界的大小。

    Cfg.commands.resampling_time = 10
    # 设置命令重采样的时间间隔为10秒。

    Cfg.reward_scales.feet_slip = -0.04
    # 设置足部滑动的奖励比例，这里为负值表示滑动为不良行为。

    Cfg.reward_scales.action_smoothness_1 = -0.1
    # 设置动作平滑性的奖励比例，鼓励更平滑的动作。

    Cfg.reward_scales.action_smoothness_2 = -0.1
    # 设置另一项动作平滑性的奖励比例。

    Cfg.reward_scales.dof_vel = -1e-4
    # 设置关节速度的奖励比例。

    Cfg.reward_scales.dof_pos = -0.05
    # 设置关节位置的奖励比例。

    Cfg.reward_scales.jump = 10.0
    # 设置跳跃的奖励比例，鼓励跳跃行为。

    Cfg.reward_scales.base_height = 0.0
    # 设置基础高度的奖励比例。

    Cfg.rewards.base_height_target = 0.30
    # 设置基础高度的目标值。

    Cfg.reward_scales.estimation_bonus = 0.0
    # 设置估计奖励的比例。

    Cfg.reward_scales.feet_impact_vel = -0.0
    # 设置足部冲击速度的奖励比例。

    # rewards.footswing_height = 0.09
    # 设置脚摆动高度的奖励比例。

    Cfg.reward_scales.feet_clearance = -0.0
    # 设置足部清除的奖励比例。

    Cfg.reward_scales.feet_clearance_cmd = -15.
    # 设置足部清除命令的奖励比例。

    # reward_scales.feet_contact_forces = -0.01
    # 设置足部接触力的奖励比例。

    Cfg.rewards.reward_container_name = "CoRLRewards"
    # 设置奖励容器的名称为"CoRLRewards"。

    Cfg.rewards.only_positive_rewards = False
    # 设置奖励机制不仅限于正向奖励。

    Cfg.rewards.only_positive_rewards_ji22_style = True
    # 设置使用ji22风格的仅正向奖励机制。

    Cfg.rewards.sigma_rew_neg = 0.02
    # 设置负向奖励的标准差为0.02。

    Cfg.reward_scales.hop_symmetry = 0.0
    # 设置跳跃对称性的奖励比例。

    Cfg.rewards.kappa_gait_probs = 0.07
    # 设置步态概率的kappa值为0.07。

    Cfg.rewards.gait_force_sigma = 100.
    # 设置步态力的标准差为100。

    Cfg.rewards.gait_vel_sigma = 10.
    # 设置步态速度的标准差为10。

    Cfg.reward_scales.tracking_contacts_shaped_force = 4.0
    # 设置跟踪接触形状力的奖励比例为4.0。

    Cfg.reward_scales.tracking_contacts_shaped_vel = 4.0
    # 设置跟踪接触形状速度的奖励比例为4.0。

    Cfg.reward_scales.collision = -5.0
    # 设置碰撞的奖励比例为-5.0，表示碰撞是不期望的。

    Cfg.commands.lin_vel_x = [-1.0, 1.0]
    # 设置x方向线速度的范围。

    Cfg.commands.lin_vel_y = [-0.6, 0.6]
    # 设置y方向线速度的范围。

    Cfg.commands.ang_vel_yaw = [-1.0, 1.0]
    # 设置偏航角速度的范围。

    Cfg.commands.body_height_cmd = [-0.25, 0.15]
    # 设置身体高度命令的范围。

    Cfg.commands.gait_frequency_cmd_range = [1.5, 4.0]
    # 设置步态频率命令的范围。

    Cfg.commands.gait_phase_cmd_range = [0.0, 1.0]
    # 设置步态阶段命令的范围。

    Cfg.commands.gait_offset_cmd_range = [0.0, 1.0]
    # 设置步态偏移命令的范围。

    Cfg.commands.gait_bound_cmd_range = [0.0, 1.0]
    # 设置步态边界命令的范围。

    Cfg.commands.gait_duration_cmd_range = [0.5, 0.5]
    # 设置步态持续时间命令的范围。

    Cfg.commands.footswing_height_range = [0.03, 0.25]
    # 设置脚摆动高度的范围。

    Cfg.reward_scales.lin_vel_z = -0.02
    # 设置z方向线速度的奖励比例。

    Cfg.reward_scales.ang_vel_xy = -0.001
    # 设置xy平面内角速度的奖励比例。

    Cfg.reward_scales.base_height = 0.0
    # 设置基础高度的奖励比例。

    Cfg.reward_scales.feet_air_time = 0.0
    # 设置脚部空中时间的奖励比例。

    Cfg.commands.limit_vel_x = [-5.0, 5.0]
    # 设置x方向速度的限制范围。

    Cfg.commands.limit_vel_y = [-0.6, 0.6]
    # 设置y方向速度的限制范围。

    Cfg.commands.limit_vel_yaw = [-5.0, 5.0]
    # 设置偏航角速度的限制范围。

    Cfg.commands.limit_body_height = [-0.25, 0.15]
    # 设置身体高度的限制范围。

    Cfg.commands.limit_gait_frequency = [1.5, 4.0]
    # 设置步态频率的限制范围。

    Cfg.commands.limit_gait_phase = [0.0, 1.0]
    # 设置步态阶段的限制范围。

    Cfg.commands.limit_gait_offset = [0.0, 1.0]
    # 设置步态偏移的限制范围。

    Cfg.commands.limit_gait_bound = [0.0, 1.0]
    # 设置步态边界的限制范围。

    Cfg.commands.limit_gait_duration = [0.5, 0.5]
    # 设置步态持续时间的限制范围。

    Cfg.commands.limit_footswing_height = [0.03, 0.25]
    # 设置脚摆动高度的限制范围。

    Cfg.commands.num_bins_vel_x = 21
    # 设置x方向速度的离散化区间数为21。

    Cfg.commands.num_bins_vel_y = 1
    # 设置y方向速度的离散化区间数为1。

    Cfg.commands.num_bins_vel_yaw = 21
    # 设置偏航角速度的离散化区间数为21。

    Cfg.commands.num_bins_body_height = 1
    # 设置身体高度的离散化区间数为1。

    Cfg.commands.num_bins_gait_frequency = 1
    # 设置步态频率的离散化区间数为1。

    Cfg.commands.num_bins_gait_phase = 1
    # 设置步态阶段的离散化区间数为1。

    Cfg.commands.num_bins_gait_offset = 1
    # 设置步态偏移的离散化区间数为1。

    Cfg.commands.num_bins_gait_bound = 1
    # 设置步态边界的离散化区间数为1。

    Cfg.commands.num_bins_gait_duration = 1
    # 设置步态持续时间的离散化区间数为1。

    Cfg.commands.num_bins_footswing_height = 1
    # 设置脚摆动高度的离散化区间数为1。

    Cfg.normalization.friction_range = [0, 1]
    # 设置摩擦力的标准化范围。

    Cfg.normalization.ground_friction_range = [0, 1]
    # 设置地面摩擦力的标准化范围。

    Cfg.commands.exclusive_phase_offset = False
    # 设置步态阶段偏移不是互斥的。

    Cfg.commands.pacing_offset = False
    # 设置步态步速偏移为禁用。

    Cfg.commands.binary_phases = True
    # 启用二元步态阶段。

    Cfg.commands.gaitwise_curricula = True
    # 启用针对不同步态的课程学习。

    Cfg.env.num_envs = 3
    # 设置环境数量为3。

    Cfg.domain_rand.push_interval_s = 1
    # 设置推动间隔为1秒。

    Cfg.terrain.num_rows = 10
    # 设置地形的行数为3。

    Cfg.terrain.num_cols = 10
    # 设置地形的列数为5。

    Cfg.terrain.border_size = 0
    # 设置地形边界的大小为0。

    Cfg.domain_rand.randomize_friction = True
    # 启用摩擦力的随机化。

    Cfg.domain_rand.friction_range = [1.0, 1.01]
    # 设置摩擦力的随机化范围。

    Cfg.domain_rand.randomize_base_mass = True
    # 启用基础质量的随机化。

    Cfg.domain_rand.added_mass_range = [0., 6.]
    # 设置额外质量的随机化范围。

    Cfg.terrain.terrain_noise_magnitude = 0.01
    # 设置地形噪声的大小为0。

    Cfg.domain_rand.lag_timesteps = 6
    # 设置延迟的时间步数为6。

    Cfg.domain_rand.randomize_lag_timesteps = True
    # 启用延迟时间步数的随机化。

    Cfg.control.control_type = "actuator_net"
    # 设置控制类型为"actuator_net"。

    # 创建VelocityTrackingEasyEnv仿真环境实例
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    # 使用GPU（cuda:0）来加速仿真，并根据Cfg配置环境。

    # 重置环境到初始状态
    env.reset()

    # 如果需要渲染并且处于无头模式
    if render and headless:
        # 渲染环境并获取图像
        img = env.render(mode="rgb_array")
        # 使用matplotlib显示图像
        plt.imshow(img)
        plt.show()
        # 打印信息并退出程序
        print("Show the first frame and exit.")
        exit()

    # 循环1000次
    for i in trange(1000, desc="Running"):
        # 创建一个全零的动作数组
        actions = 0. * torch.ones(env.num_envs, env.num_actions, device=env.device)
        # 向环境发送动作，获取观察结果、奖励、完成状态和其他信息
        obs, rew, done, info = env.step(actions)

    # 当循环结束时打印"Done"
    print("Done")

# 这段代码检查是否直接运行该脚本
if __name__ == '__main__':
    # 如果是直接运行，则调用run_env函数
    run_env(render=True, headless=True)