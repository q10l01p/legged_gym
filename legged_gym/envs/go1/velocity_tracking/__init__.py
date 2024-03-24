from isaacgym import gymutil, gymapi
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg as Cfg


class VelocityTrackingEasyEnv(LeggedRobot):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):
        # 如果提供了环境数量，则更新配置中的环境数量。
        if num_envs is not None:
            cfg.env.num_envs = num_envs

        # 初始化模拟参数。
        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)

        # 调用基类的构造函数，初始化基类。
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, eval_cfg, initial_dynamics_dict)

    def step(self, actions):
        """
        在环境中执行一个动作，并返回结果。

        参数:
        - actions: 要执行的动作数组。

        返回:
        - obs_buf: 观测数据缓冲区。
        - rew_buf: 奖励数据缓冲区。
        - reset_buf: 环境重置信号的缓冲区。
        - extras: 包含额外信息的字典。

        功能:
        - 使用给定的动作更新环境状态。
        - 计算和记录额外的环境信息，如关节位置、速度等。
        """

        # 使用基类的 step 方法来更新环境状态。
        self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras = super().step(actions)

        # 计算脚部的位置信息。
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]

        # 更新额外信息到 extras 字典中。
        self.extras.update({
            "privileged_obs": self.privileged_obs_buf,
            "joint_pos": self.dof_pos.cpu().numpy(),  # 关节位置
            "joint_vel": self.dof_vel.cpu().numpy(),  # 关节速度
            "joint_pos_target": self.joint_pos_target.cpu().detach().numpy(),  # 目标关节位置
            "joint_vel_target": torch.zeros(12),  # 目标关节速度
            "body_linear_vel": self.base_lin_vel.cpu().detach().numpy(),  # 机体线速度
            "body_angular_vel": self.base_ang_vel.cpu().detach().numpy(),  # 机体角速度
            "body_linear_vel_cmd": self.commands.cpu().numpy()[:, 0:2],  # 线速度命令
            "body_angular_vel_cmd": self.commands.cpu().numpy()[:, 2:],  # 角速度命令
            "contact_states": (self.contact_forces[:, self.feet_indices, 2] > 1.).detach().cpu().numpy().copy(),  # 接触状态
            "foot_positions": self.foot_positions.detach().cpu().numpy().copy(),  # 脚部位置
            "body_pos": self.root_states[:, 0:3].detach().cpu().numpy(),  # 机体位置
            "torques": self.torques.detach().cpu().numpy()  # 扭矩
        })

        # 返回观测数据、奖励、重置信号和额外信息。
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        """
        重置环境到初始状态。

        返回:
        - obs: 重置后的环境观测数据。

        功能:
        - 重置环境中的所有实例。
        - 获取重置后的初始观测数据。
        """

        # 重置指定数量的环境实例。这是通过向 reset_idx 函数传递一个包含所有环境索引的数组来完成的。
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # 执行一步动作（这里是零动作）来获取初始状态的观测数据。
        # 这里的动作是零，表示在重置时不对环境施加任何特定的动作影响。
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))

        # 返回重置后的观测数据。
        return obs

