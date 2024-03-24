import isaacgym

assert isaacgym
import os
import torch
import numpy as np

import glob
import pickle as pkl

from legged_gym.envs import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg as Cfg
from legged_gym.envs import config_go1, config_a1
from legged_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm
from ml_logger import logger
from legged_gym.envs.base.legged_robot import LeggedRobot


def load_policy(logdir):
    # 定义一个函数load_policy，接收一个参数logdir，这个参数是日志目录的路径。

    body = torch.jit.load(logdir + '/checkpoints/teacher_body_latest.jit').to("cuda")
    # 加载一个名为'body_latest.jit'的预训练模型。这个模型可能是一个神经网络，用于决定动作或行为。

    import os
    # 导入os模块，通常用于处理文件路径和操作系统功能。

    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit').to("cuda")

    # 加载另一个名为'adaptation_module_latest.jit'的预训练模型。这个模型可能用于根据观察结果调整策略。

    def policy(obs, info={}):
        # 定义一个嵌套函数policy，接收观察结果obs和一个可选的info字典。
        latent = adaptation_module.forward(
            torch.cat((obs["obs_history"], obs["acts_history"]), dim=-1))
        # 使用adaptation_module模型处理观察历史，生成潜在状态（latent）。这里将观察结果移动到CPU进行处理。

        ret = body.forward(torch.cat((obs["obs_history"], latent), dim=-1))
        action = ret[:, 3:]
        command = abs(torch.tanh(ret[:, :3]))
        # 使用body模型结合观察结果和潜在状态来决定下一个动作。这里也是将数据移动到CPU。

        info['latent'] = latent
        # 将潜在状态存储在info字典中，可能用于后续分析或调试。

        return action, command
        # 返回计算出的动作。

    return policy
    # 返回policy函数。这个函数现在可以用来根据观察结果生成动作。


def load_env(label, headless=False):
    # 定义一个函数load_env，接收两个参数：label和headless。label用于指定要加载的环境配置。
    current_work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dirs = glob.glob(f"{current_work_dir}/runs/{label}/*/*/train/*")
    # 使用glob模块搜索与指定label匹配的目录。
    logdir = sorted(dirs)[-1]
    # 对找到的目录进行排序，并选择第一个目录。这通常是最新的或特定的配置目录。
    with open(logdir + "/parameters.pkl", 'rb') as file:
        # 打开包含参数的Pickle文件。Pickle用于序列化和反序列化Python对象结构。
        pkl_cfg = pkl.load(file)
        # 从Pickle文件中加载配置。
        print(pkl_cfg.keys())
        # 打印配置字典的键，这有助于了解存储了哪些参数。
        # cfg = config_a1
        cfg = pkl_cfg["Cfg"]
        # 从加载的配置中获取特定的配置部分，通常是模型或环境的主要配置。
        print(cfg.keys())
        # 打印配置部分的键。

        for key, value in cfg.items():
            # 遍历配置字典的键值对。
            if hasattr(Cfg, key):
                # 检查全局配置对象Cfg是否有与当前键相同的属性。
                for key2, value2 in cfg[key].items():
                    # 遍历内部配置字典的键值对。
                    setattr(getattr(Cfg, key), key2, value2)
                    # 更新全局配置对象Cfg的属性。这里使用getattr和setattr来动态访问和设置属性。

    # 关闭领域随机化(DR)以进行评估
    Cfg.domain_rand.push_robots = False
    # 关闭推动机器人的随机化。

    Cfg.domain_rand.randomize_friction = False
    # 关闭摩擦力的随机化。

    Cfg.domain_rand.randomize_gravity = False
    # 关闭重力的随机化。

    Cfg.domain_rand.randomize_restitution = False
    # 关闭恢复系数（弹性）的随机化。

    Cfg.domain_rand.randomize_motor_offset = False
    # 关闭电机偏移的随机化。

    Cfg.domain_rand.randomize_motor_strength = False
    # 关闭电机强度的随机化。

    Cfg.domain_rand.randomize_friction_indep = False
    # 关闭独立摩擦力的随机化。

    Cfg.domain_rand.randomize_ground_friction = False
    # 关闭地面摩擦力的随机化。

    Cfg.domain_rand.randomize_base_mass = False
    # 关闭基础质量的随机化。

    Cfg.domain_rand.randomize_Kd_factor = False
    # 关闭Kd因子（微分增益）的随机化。

    Cfg.domain_rand.randomize_Kp_factor = False
    # 关闭Kp因子（比例增益）的随机化。

    Cfg.domain_rand.randomize_joint_friction = False
    # 关闭关节摩擦力的随机化。

    Cfg.domain_rand.randomize_com_displacement = False
    # 关闭质心位移的随机化。

    # 设置环境配置
    Cfg.env.num_recording_envs = 1
    # 设置记录环境的数量为1。

    Cfg.env.num_envs = 1
    # 设置环境数量为1。

    Cfg.terrain.mesh_type = 'trimesh'  # 地形网格类型，可选值有'none'、'plane'、'heightfield'或'trimesh'
    # 地形类型的比例：[平滑斜坡, 粗糙斜坡, 上楼梯, 下楼梯, 离散]
    Cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]

    Cfg.terrain.num_rows = 10
    # 设置地形的行数为5。

    Cfg.terrain.num_cols = 10
    # 设置地形的列数为5。

    Cfg.terrain.border_size = 0
    # 设置地形边界的大小为0。

    Cfg.terrain.center_robots = False
    # 设置机器人在地形中心。

    Cfg.terrain.center_span = 1
    # 设置地形中心的跨度为1。

    Cfg.terrain.teleport_robots = True
    # 允许机器人瞬移。

    Cfg.domain_rand.lag_timesteps = 6
    # 设置延迟时间步数为6。

    Cfg.domain_rand.randomize_lag_timesteps = True
    # 开启延迟时间步数的随机化。

    Cfg.control.control_type = "actuator_net"
    # 设置控制类型为"actuator_net"。

    from legged_gym.envs.wrappers.history_wrapper import HistoryWrapper
    # 导入HistoryWrapper，这可能是一个用于环绑包装器，用于记录并处理环境的历史信息。

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    # 创建一个VelocityTrackingEasyEnv环境实例，使用GPU加速（'cuda:0'），并根据前面设置的Cfg配置环境。

    env = HistoryWrapper(env)
    # 使用HistoryWrapper包装器包装环境，增加了处理历史信息的功能。

    policy = load_policy(logdir)
    # 调用前面定义的load_policy函数来加载策略。logdir是包含策略模型的目录。

    return env, policy
    # 返回配置好的环境和加载的策略。


def play_go1(headless=True):
    # play_go1 函数，用于执行 go1 实体的仿真。可以在无头模式下运行。

    label = "gait-conditioned-agility"  # 仿真实验的标签

    # 加载环境和策略
    env, policy = load_env(label, headless=headless)

    num_eval_steps = 800  # 评估步数

    # 定义不同步态的命令
    gaits = {
        "pronking": [0, 0, 0],
        "trotting": [0.5, 0, 0],
        "bounding": [0, 0.5, 0],
        "pacing": [0, 0, 0.5],
    }

    # 设置运动命令
    # todo:x = 5 时崩溃
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, .0, .0  # 速度命令
    body_height_cmd = 0  # 身体高度命令
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])  # 选择步态（这里使用“pronking”）
    footswing_height_cmd = 0.5  # 脚摆高度命令
    pitch_cmd = 0.0  # 俯仰角命令
    roll_cmd = 0.0  # 翻滚角命令
    stance_width_cmd = 0.25  # 站姿宽度命令

    # 初始化记录仿真数据的数组
    measured_x_vels = np.zeros(num_eval_steps)  # 实测X速度
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd  # 目标X速度
    joint_positions = np.zeros((num_eval_steps, 12))  # 关节位置

    print("START RECORDING")
    env.start_recording()

    # 重置环境，开始仿真
    obs = env.reset()

    env.start_recording()

    for i in tqdm(range(num_eval_steps)):
        # 循环执行仿真步骤，使用 tqdm 来显示进度条
        if i == 498:
            i = 0

        with torch.no_grad():
            actions, command = policy(obs)  # 使用策略生成动作，不计算梯度

        # print(command)

        # 设置各种运动命令
        env.commands[:, 0] = x_vel_cmd  # X轴速度命令
        env.commands[:, 1] = y_vel_cmd  # Y轴速度命令
        env.commands[:, 2] = yaw_vel_cmd  # 偏航速度命令
        env.commands[:, 3] = body_height_cmd  # 身体高度命令
        env.commands[:, 4] = step_frequency_cmd  # 步频命令
        env.commands[:, 5:8] = gait  # 步态命令
        # env.commands[:, 5:8] = torch.clip(command, 0, 1)  # 步态命令
        env.commands[:, 8] = 0.5  # 未指定命令，可能是默认值
        env.commands[:, 9] = footswing_height_cmd  # 脚摆高度命令
        env.commands[:, 10] = pitch_cmd  # 俯仰角命令
        env.commands[:, 11] = roll_cmd  # 翻滚角命令
        env.commands[:, 12] = stance_width_cmd  # 站姿宽度命令

        obs, rew, done, info = env.step(actions)  # 执行一步仿真
        env.recording()

        # 记录仿真数据
        measured_x_vels[i] = env.base_lin_vel[0, 0]  # 记录实测X轴速度
        joint_positions[i] = env.dof_pos[0, :].cpu()  # 记录关节位置

    print("LOGGING VIDEO")
    frames = env.get_frames()
    logger.save_video(frames, f"./videos/trotting.mp4", fps=2 / env.dt)

    # 使用 matplotlib 绘制图表
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))

    # 绘制前进线速度
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-",
                label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--",
                label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity Pacing")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    # 绘制关节位置
    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-",
                label="Measured")
    axs[1].set_title("Joint Positions")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
