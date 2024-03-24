import os
from typing import Dict
import torch
import torch.nn.functional as F
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

from legged_gym import MINI_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, get_scale_shift
from legged_gym.utils.terrain import Terrain
from .legged_robot_config import LeggedRobotCfg as Cfg

assert gymtorch


class LeggedRobot(BaseTask):
    def __init__(self, cfg: Cfg, sim_params, physics_engine, sim_device,
                 headless, eval_cfg=None, initial_dynamics_dict=None):

        self.cfg = cfg                  # 将环境配置文件保存到实例变量中
        self.eval_cfg = eval_cfg  # 保存评估配置
        self.sim_params = sim_params    # 将模拟环境的参数保存到实例变量中
        self.height_samples = None      # 初始化高度样本变量为None
        self.debug_viz = False          # 设置调试可视化标志为False
        self.init_done = False          # 设置初始化完成标志为False
        self.initial_dynamics_dict = initial_dynamics_dict  # 保存初始动态参数字典
        self._parse_cfg(self.cfg)       # 解析配置文件

        # 调用父类的初始化函数
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless, self.eval_cfg)

        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))  # 初始化命令分布

        if not self.headless:               # 如果不是无头模式
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)                # 设置相机位置和观察点
        self._init_buffers()                # 初始化PyTorch缓冲区

        self._prepare_reward_function()     # 准备奖励函数
        self.init_done = True               # 标记初始化完成
        self.record_now = False             # 设置记录标志为False
        self.record_eval_now = False        # 设置评估记录标志为False
        self.collecting_evaluation = False  # 设置正在收集评估标志为False
        self.num_still_evaluating = 0       # 设置仍在评估的数量为0

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions  # 获取配置中的动作裁剪参数
        # 将动作裁剪到指定范围并转移到设备上
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # 保存当前的基础位置、基础四元数、基础线速度和脚部速度
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        self.render_gui()  # 渲染GUI

        for _ in range(self.cfg.control.decimation):        # 根据减频参数重复模拟
            # 计算扭矩并调整其形状以匹配扭矩张量
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # 设置关节的驱动力
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)                     # 进行物理模拟
            self.gym.fetch_results(self.sim, True)      # 获取模拟结果
            self.gym.refresh_dof_state_tensor(self.sim)     # 刷新关节状态张量
        self.post_physics_step()                            # 调用后处理方法

        # 裁剪观测值并返回
        clip_obs = self.cfg.normalization.clip_observations # 获取观测值裁剪参数
        # 裁剪观测值
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:             # 如果有特权观测值
            # 裁剪特权观测值
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        # 返回观测值、特权观测值、奖励、重置标志和额外信息
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """
        检查终止条件，计算观测值和奖励。
        调用self._post_physics_step_callback()进行通用计算。
        如果需要，调用self._draw_debug_vis()进行调试可视化。
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)      # 刷新演员根状态张量
        self.gym.refresh_net_contact_force_tensor(self.sim)     # 刷新网络接触力张量
        self.gym.refresh_rigid_body_state_tensor(self.sim)      # 刷新刚体状态张量

        if self.record_now:  # 如果当前处于记录模式
            self.gym.step_graphics(self.sim)  # 进行图形步骤
            self.gym.render_all_camera_sensors(self.sim)  # 渲染所有相机传感器

        self.episode_length_buf += 1    # 更新回合长度缓冲区
        self.common_step_counter += 1   # 更新通用步数计数器

        # 准备基础位置、基础四元数、基础线速度、基础角速度、投影重力、脚部速度和脚部位置等量
        self.base_pos[:] = self.root_states[:self.num_envs, 0:3]
        self.base_quat[:] = self.root_states[:self.num_envs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs,
                                                          self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs,
                                                         self.num_bodies, 13)[:, self.feet_indices, 0:3]

        self._post_physics_step_callback()  # 调用通用后处理计算方法

        # 计算观测值、奖励、重置等
        self.check_termination()    # 检查终止条件
        self.compute_reward()       # 计算奖励
        # 获取需要重置的环境ID
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)         # 处理重置
        self.compute_observations()     # 计算观测值

        # 更新上一步和上上一步的动作、关节位置目标、关节速度和根速度
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # 如果需要，进行调试可视化
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()  # 绘制调试信息

        self._render_headless()  # 进行无头渲染

    def check_termination(self):
        """
        检查是否有环境需要重置。
        """
        # 检查接触力是否超过阈值，超过则标记为需要重置
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1
        )
        # 检查是否达到最大回合长度，达到则标记为需要重置
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        # 更新重置缓冲区，合并接触力和时间超时的重置条件
        self.reset_buf |= self.time_out_buf
        # 如果使用了终止体高度作为奖励
        if self.cfg.rewards.use_terminal_body_height:
            # 计算体高度，如果体高度小于终止体高度，则设置对应的体高度标志为True，并将重置标志设置为True
            self.body_height_buf = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) \
                                < self.cfg.rewards.terminal_body_height
            self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)

    def reset_idx(self, env_ids):
        """
        重置一些环境。
        调用self._reset_dofs(env_ids)、self._reset_root_states(env_ids)和self._resample_commands(env_ids)。
        [可选]调用self._update_terrain_curriculum(env_ids)、self.update_command_curriculum(env_ids)。
        记录回合信息。
        重置一些缓冲区。
        """
        if len(env_ids) == 0:  # 如果没有环境需要重置，则直接返回
            return

        # 重置机器人状态
        self._update_terrain_curriculum(env_ids)
        self._resample_commands(env_ids)                                            # 重新采样命令(步态课程设置)
        self._call_train_eval(self._randomize_dof_props, env_ids)                   # 随机化自由度属性
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)        # 如果配置中设定了在开始后随机化刚体，则执行刚体属性随机化
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)    # 刷新刚体形状属性

        self._call_train_eval(self._reset_dofs, env_ids)                            # 重置自由度
        self._call_train_eval(self._reset_root_states, env_ids)                     # 重置根状态

        # 重置缓冲区
        self.last_actions[env_ids] = 0.         # 将最后的动作缓冲区中对应环境ID的部分重置为0
        self.last_last_actions[env_ids] = 0.    # 将最后的最后的动作缓冲区中对应环境ID的部分重置为0
        self.last_dof_vel[env_ids] = 0.         # 将最后的自由度速度缓冲区中对应环境ID的部分重置为0
        self.feet_air_time[env_ids] = 0.        # 将脚部空中时间缓冲区中对应环境ID的部分重置为0
        self.episode_length_buf[env_ids] = 0    # 将剧集长度缓冲区中对应环境ID的部分重置为0
        self.reset_buf[env_ids] = 1             # 将重置缓冲区中对应环境ID的部分设定为1，表示已经重置

        # 填充额外信息
        train_env_ids = env_ids[env_ids < self.num_train_envs]      # 获取训练环境的ID
        if len(train_env_ids) > 0:                                  # 如果训练环境ID列表不为空
            self.extras["train/episode"] = {}                       # 在额外信息中添加训练环境的剧集信息
            for key in self.episode_sums.keys():                    # 遍历剧集总和的键
                # 计算对应键的剧集总和的平均值，并存储到额外信息的训练环境剧集信息中
                self.extras["train/episode"]['rew_' + key] = torch.mean(self.episode_sums[key][train_env_ids])
                self.episode_sums[key][train_env_ids] = 0.          # 重置对应键的剧集总和

        eval_env_ids = env_ids[env_ids >= self.num_train_envs]      # 获取评估环境的ID
        # 如果评估环境ID列表不为空
        if len(eval_env_ids) > 0:
            self.extras["eval/episode"] = {}                        # 在额外信息中添加评估环境的剧集信息
            # 遍历剧集总和的键
            for key in self.episode_sums.keys():
                # 保存未设置的评估环境的剧集总和
                unset_eval_envs = eval_env_ids[self.episode_sums_eval[key][eval_env_ids] == -1]
                self.episode_sums_eval[key][unset_eval_envs] = self.episode_sums[key][unset_eval_envs]
                self.episode_sums[key][eval_env_ids] = 0.           # 重置对应键的剧集总和

        # 记录额外的课程信息
        # 如果配置中设定了地形课程
        if self.cfg.terrain.curriculum:
            # 计算训练环境的地形等级的平均值，并存储到额外信息的训练环境剧集信息中
            self.extras["train/episode"]["terrain_level"] = torch.mean(
                self.terrain_levels[:self.num_train_envs].float())
        # 如果配置中的命令课程为真
        if self.cfg.commands.command_curriculum:
            # 将环境命令分箱数据转为张量，并取前num_train_envs个
            self.extras["env_bins"] = torch.Tensor(self.env_command_bins)[:self.num_train_envs]
            # 计算并存储命令中的最小持续时间
            self.extras["train/episode"]["min_command_duration"] = torch.min(self.commands[:, 8])
            # 计算并存储命令中的最大持续时间
            self.extras["train/episode"]["max_command_duration"] = torch.max(self.commands[:, 8])
            # 计算并存储命令中的最小边界值
            self.extras["train/episode"]["min_command_bound"] = torch.min(self.commands[:, 7])
            # 计算并存储命令中的最大边界值
            self.extras["train/episode"]["max_command_bound"] = torch.max(self.commands[:, 7])
            # 计算并存储命令中的最小偏移值
            self.extras["train/episode"]["min_command_offset"] = torch.min(self.commands[:, 6])
            # 计算并存储命令中的最大偏移值
            self.extras["train/episode"]["max_command_offset"] = torch.max(self.commands[:, 6])
            # 计算并存储命令中的最小相位值
            self.extras["train/episode"]["min_command_phase"] = torch.min(self.commands[:, 5])
            # 计算并存储命令中的最大相位值
            self.extras["train/episode"]["max_command_phase"] = torch.max(self.commands[:, 5])
            # 计算并存储命令中的最小频率值
            self.extras["train/episode"]["min_command_freq"] = torch.min(self.commands[:, 4])
            # 计算并存储命令中的最大频率值
            self.extras["train/episode"]["max_command_freq"] = torch.max(self.commands[:, 4])
            # 计算并存储命令中的最小x方向速度
            self.extras["train/episode"]["min_command_x_vel"] = torch.min(self.commands[:, 0])
            # 计算并存储命令中的最大x方向速度
            self.extras["train/episode"]["max_command_x_vel"] = torch.max(self.commands[:, 0])
            # 计算并存储命令中的最小y方向速度
            self.extras["train/episode"]["min_command_y_vel"] = torch.min(self.commands[:, 1])
            # 计算并存储命令中的最大y方向速度
            self.extras["train/episode"]["max_command_y_vel"] = torch.max(self.commands[:, 1])
            # 计算并存储命令中的最小偏航速度
            self.extras["train/episode"]["min_command_yaw_vel"] = torch.min(self.commands[:, 2])
            # 计算并存储命令中的最大偏航速度
            self.extras["train/episode"]["max_command_yaw_vel"] = torch.max(self.commands[:, 2])
            # 如果命令的数量大于9
            if self.cfg.commands.num_commands > 9:
                # 计算并存储命令中的最小摆动高度
                self.extras["train/episode"]["min_command_swing_height"] = torch.min(self.commands[:, 9])
                # 计算并存储命令中的最大摆动高度
                self.extras["train/episode"]["max_command_swing_height"] = torch.max(self.commands[:, 9])
            # 对于每个课程和类别
            for curriculum, category in zip(self.curricula, self.category_names):
                # 计算并存储命令区域的平均权重
                self.extras["train/episode"][f"command_area_{category}"] = np.sum(curriculum.weights) / \
                                                                           curriculum.weights.shape[0]

            # 计算并存储动作的最小值
            self.extras["train/episode"]["min_action"] = torch.min(self.actions)
            # 计算并存储动作的最大值
            self.extras["train/episode"]["max_action"] = torch.max(self.actions)

            # 初始化课程分布字典
            self.extras["curriculum/distribution"] = {}
            # 对于每个课程和类别
            for curriculum, category in zip(self.curricula, self.category_names):
                # 存储权重
                self.extras[f"curriculum/distribution"][f"weights_{category}"] = curriculum.weights
                # 存储网格
                self.extras[f"curriculum/distribution"][f"grid_{category}"] = curriculum.grid
        # 如果环境配置中的send_timeouts为真
        if self.cfg.env.send_timeouts:
            # 将超时缓冲区的前num_train_envs个元素存储到extras的"time_outs"键下
            self.extras["time_outs"] = self.time_out_buf[:self.num_train_envs]

        # 将步态索引中对应环境ID的元素设为0
        self.gait_indices[env_ids] = 0

        # 遍历滞后缓冲区
        for i in range(len(self.lag_buffer)):
            # 将滞后缓冲区中对应环境ID的元素设为0
            self.lag_buffer[i][env_ids, :] = 0

    def set_idx_pose(self, env_ids, dof_pos, base_state):
        """
        设置指定环境ID的姿态
        """
        # 如果没有指定环境ID，则直接返回
        if len(env_ids) == 0:
            return
        # 将环境ID转换为int32类型，并移到设备上
        env_ids_int32 = env_ids.to(dtype=torch.int32).to(self.device)
        # 如果提供了自由度位置
        if dof_pos is not None:
            # 更新对应环境ID的自由度位置
            self.dof_pos[env_ids] = dof_pos
            # 将对应环境ID的自由度速度设为0
            self.dof_vel[env_ids] = 0.
            # 在模拟器中设置对应的自由度状态
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # 更新对应环境ID的根状态
        self.root_states[env_ids] = base_state.to(self.device)
        # 在模拟器中设置对应的根状态
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def compute_reward(self):
        """
        计算奖励。
        调用每个非零比例的奖励函数（在self._prepare_reward_function()中处理）。
        将每个项加到回合总和和总奖励中。
        """
        # 初始化奖励缓冲区
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        # 对于每个奖励函数
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            # 计算奖励并乘以对应的比例
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # 更新奖励缓冲区
            self.rew_buf += rew
            # 如果奖励为正，更新正奖励缓冲区
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            # 如果奖励为负，更新负奖励缓冲区
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            # 更新episode sums
            self.episode_sums[name] += rew
            # 更新command sums
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
        # 如果只有正奖励，将奖励裁剪为非负
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # 如果只有正奖励（ji22 style），将奖励设置为正奖励乘以负奖励的指数
        elif self.cfg.rewards.only_positive_rewards_ji22_style:
            #TODO: update
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)
        # 更新总奖励
        self.episode_sums["total"] += self.rew_buf
        # 在裁剪后添加终止奖励
        # 如果"termination"在奖励比例中
        if "termination" in self.reward_scales:
            # 计算终止奖励并乘以对应的比例
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            # 更新奖励缓冲区
            self.rew_buf += rew
            # 更新终止奖励的episode sums
            self.episode_sums["termination"] += rew
            # 更新终止奖励的command sums
            self.command_sums["termination"] += rew

        # 更新线速度原始值的command sums
        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        # 更新角速度原始值的command sums
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        # 更新线速度残差的command sums
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        # 更新角速度残差的command sums
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        # 更新episode timesteps的command sums
        self.command_sums["ep_timesteps"] += 1

    def compute_observations(self):
        """ Computes observations
        """
        # 将重力投影、动作自由度（DOF）位置与默认位置的差值经过缩放、DOF速度经过缩放以及动作合并到观察缓冲区
        self.obs_buf = torch.cat((self.projected_gravity,                                       # 将重力投影
                                  (self.dof_pos[:, :self.num_actuated_dof] -
                                   self.default_dof_pos[:, :self.num_actuated_dof])
                                  * self.obs_scales.dof_pos,                                            # 动作自由度（DOF）位置
                                  self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,    # 默认位置的差值
                                  self.actions                                                          # 动作
                                  ), dim=-1)

        # 如果配置了观察命令而没有观察高度命令，则将重力投影、命令经过缩放、DOF位置差值经过缩放、DOF速度经过缩放以及动作合并到观察缓冲区
        # if self.cfg.env.observe_command and not self.cfg.env.observe_height_command:
        #     self.obs_buf = torch.cat((self.projected_gravity,
        #                               self.commands[:, :3] * self.commands_scale,
        #                               (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
        #                               self.dof_vel * self.obs_scales.dof_vel,
        #                               self.actions
        #                               ), dim=-1)

        # 如果配置了观察命令，将重力投影、命令经过缩放、DOF位置差值经过缩放、DOF速度经过缩放以及动作合并到观察缓冲区
        if self.cfg.env.observe_command:
            self.obs_buf = torch.cat((self.projected_gravity,
                                      self.commands * self.commands_scale,
                                      (self.dof_pos[:, :self.num_actuated_dof] -
                                       self.default_dof_pos[:,:self.num_actuated_dof]) * self.obs_scales.dof_pos,
                                      self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,
                                      self.actions
                                      ), dim=-1)

        # 如果配置了观察前两个动作，将这些动作加入到观察缓冲区
        if self.cfg.env.observe_two_prev_actions:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.last_actions), dim=-1)

        # 如果配置了观察步态索引（可能是用于控制不同的运动模式），将步态索引加入到观察缓冲区
        if self.cfg.env.observe_timing_parameter:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.gait_indices.unsqueeze(1)), dim=-1)

        # 如果配置了观察时钟输入（可能用于控制周期性动作），将时钟输入加入到观察缓冲区
        if self.cfg.env.observe_clock_inputs:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.clock_inputs), dim=-1)

        # 如果配置了观察期望接触状态，将期望接触状态加入到观察缓冲区
        # if self.cfg.env.observe_desired_contact_states:
        #     self.obs_buf = torch.cat((self.obs_buf,
        #                               self.desired_contact_states), dim=-1)

        # 如果配置了观察速度
        if self.cfg.env.observe_vel:
            # 如果命令使用全局参考，将根状态的线性速度、基座的角速度加入到观察缓冲区的前面
            if self.cfg.commands.global_reference:
                self.obs_buf = torch.cat((self.root_states[:self.num_envs, 7:10] * self.obs_scales.lin_vel,
                                          self.base_ang_vel * self.obs_scales.ang_vel,
                                          self.obs_buf), dim=-1)
            else:
                # 否则，将基座的线性速度和角速度加入到观察缓冲区的前面
                self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                          self.base_ang_vel * self.obs_scales.ang_vel,
                                          self.obs_buf), dim=-1)

        # 如果配置了只观察角速度，将基座的角速度加入到观察缓冲区的前面
        if self.cfg.env.observe_only_ang_vel:
            self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                      self.obs_buf), dim=-1)

        # 如果配置了只观察线速度，将基座的线速度加入到观察缓冲区的前面
        if self.cfg.env.observe_only_lin_vel:
            self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                      self.obs_buf), dim=-1)

        # 如果配置了观察偏航角（yaw角）
        if self.cfg.env.observe_yaw:
            # 计算前方向量，使用四元数将基础前向向量转换到当前姿态
            forward = quat_apply(self.base_quat, self.forward_vec)
            # 计算偏航角，即基础向量在水平面上的角度，并将其增加一个维度以匹配观察缓冲区的形状
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            # 计算了偏航角误差，并将其增加一个维度以匹配观察缓冲区的形状
            # heading_error = torch.clip(0.5 * wrap_to_pi(heading), -1., 1.).unsqueeze(1)
            # 将偏航角加入到观察缓冲区
            self.obs_buf = torch.cat((self.obs_buf,
                                      heading), dim=-1)

        # 如果配置了观察接触状态
        if self.cfg.env.observe_contact_states:
            # 计算脚部接触力是否超过阈值（这里是1），并将其转换为二进制值（接触或不接触），然后加入到观察缓冲区
            self.obs_buf = torch.cat((self.obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.).view(
                self.num_envs,
                -1) * 1.0), dim=1)

        # 如果需要添加噪声
        if self.add_noise:
            # 在观察缓冲区中添加均匀分布的噪声，噪声级别由`noise_scale_vec`决定
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # 初始化特权观察缓冲区
        self.privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)
        self.next_privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        # 检查是否配置了观察摩擦系数，并进行相应的处理
        if self.cfg.env.priv_observe_friction:
            # 使用先前定义的get_scale_shift函数计算摩擦系数的缩放比例和平移量
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            # 将处理后的摩擦系数添加到privileged_obs_buf
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.friction_coeffs[:, 0].unsqueeze(
                                                     1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                dim=1)
            # 对next_privileged_obs_buf执行相同的处理
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.friction_coeffs[:, 0].unsqueeze(
                                                          1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                     dim=1)

        # 处理地面摩擦系数
        # if self.cfg.env.priv_observe_ground_friction:
        #     # 计算地面摩擦系数
        #     self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
        #     # 计算缩放比例和平移量
        #     ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(
        #         self.cfg.normalization.ground_friction_range)
        #     # 更新privileged_obs_buf
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.ground_friction_coeffs.unsqueeze(1) -
        #                                           ground_friction_coeffs_shift) * ground_friction_coeffs_scale),
        #                                         dim=1)
        #     # 更新next_privileged_obs_buf
        #     self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
        #                                               (self.ground_friction_coeffs.unsqueeze(1) -
        #                                                ground_friction_coeffs_shift) * ground_friction_coeffs_scale),
        #                                              dim=1)

        # 如果配置了观察弹性系数（restitution）
        if self.cfg.env.priv_observe_restitution:
            # 使用get_scale_shift函数根据配置的弹性系数范围计算归一化所需的缩放比例和平移量
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            # 将归一化后的弹性系数加入到当前和下一个时间步的特权观察缓冲区
            # 先通过unsqueeze增加一个维度，然后减去平移量后乘以缩放比例
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.restitutions[:, 0].unsqueeze(
                                                     1) - restitutions_shift) * restitutions_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.restitutions[:, 0].unsqueeze(
                                                          1) - restitutions_shift) * restitutions_scale),
                                                     dim=1)

        # 如果配置了观察基础质量
        if self.cfg.env.priv_observe_base_mass:
            # 使用get_scale_shift函数根据配置的额外质量范围计算归一化所需的缩放比例和平移量
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            # 将归一化后的额外质量加入到当前和下一个时间步的特权观察缓冲区
            # 先通过unsqueeze增加一个维度，然后减去平移量后乘以缩放比例
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                     dim=1)

        # 如果配置了观察质心位移
        if self.cfg.env.priv_observe_com_displacement:
            # 根据配置的质心位移范围计算归一化所需的缩放比例和平移量
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.normalization.com_displacement_range)
            # 将归一化后的质心位移加入到当前和下一个时间步的特权观察缓冲区
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.com_displacements - com_displacements_shift)
                                                 * com_displacements_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      ( self.com_displacements - com_displacements_shift)
                                                      * com_displacements_scale),
                                                     dim=1)

        if self.cfg.env.priv_observe_body_velocity:
            # 如果配置中设置要观察身体速度，则执行以下操作

            # 获取身体速度的缩放比例和偏移量
            body_velocity_scale, body_velocity_shift = get_scale_shift(self.cfg.normalization.body_velocity_range)

            # 将处理后的速度数据添加到当前的特权观测缓冲区(self.privileged_obs_buf)
            self.privileged_obs_buf = torch.cat((
                self.privileged_obs_buf,  # 当前的特权观测数据
                ((self.base_lin_vel).view(self.num_envs, -1) - body_velocity_shift) * body_velocity_scale
            # 将基础线性速度self.base_lin_vel处理后的结果
            ), dim=1)

            # 将处理后的速度数据添加到下一个时间点的特权观测缓冲区(self.next_privileged_obs_buf)
            self.next_privileged_obs_buf = torch.cat((
                self.next_privileged_obs_buf,  # 下一个时间点的特权观测数据
                ((self.base_lin_vel).view(self.num_envs, -1) - body_velocity_shift) * body_velocity_scale
            # 将基础线性速度self.base_lin_vel处理后的结果
            ), dim=1)

        # 如果配置了观察电机强度
        if self.cfg.env.priv_observe_motor_strength:
            # 根据配置的电机强度范围计算归一化所需的缩放比例和平移量
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
            # 将归一化后的电机强度加入到当前和下一个时间步的特权观察缓冲区
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.motor_strengths - motor_strengths_shift)
                                                 * motor_strengths_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.motor_strengths - motor_strengths_shift)
                                                      * motor_strengths_scale),
                                                     dim=1)

        # 如果配置了观察电机偏移
        if self.cfg.env.priv_observe_motor_offset:
            # 根据配置的电机偏移范围计算归一化所需的缩放比例和平移量
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
            # 将归一化后的电机偏移加入到当前和下一个时间步的特权观察缓冲区
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                      (self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                     dim=1)

        # 如果配置了观察身体高度
        if self.cfg.env.priv_observe_body_height:
            # 根据配置的身体高度范围计算归一化所需的缩放比例和平移量
            body_height_scale, body_height_shift = get_scale_shift(self.cfg.normalization.body_height_range)
            # 将归一化后的身体高度加入到当前和下一个时间步的特权观察缓冲区
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 ((self.root_states[:self.num_envs, 2]).view(
                                                     self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      ((self.root_states[:self.num_envs, 2]).view(
                                                          self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                     dim=1)


        # 如果配置了观察重力
        if self.cfg.env.priv_observe_gravity:
            # 根据配置的重力范围计算归一化所需的缩放比例和平移量
            gravity_scale, gravity_shift = get_scale_shift(self.cfg.normalization.gravity_range)
            # 将归一化后的重力加入到当前和下一个时间步的特权观察缓冲区
            # 注意此处是通过除以缩放比例进行归一化，而非乘以缩放比例
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.gravities - gravity_shift) / gravity_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.gravities - gravity_shift) / gravity_scale), dim=1)

        # 如果配置了观察时钟输入
        if self.cfg.env.priv_observe_clock_inputs:
            # 直接将时钟输入加入到当前时间步的特权观察缓冲区，无需归一化处理
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.clock_inputs), dim=-1)

        # 如果配置了观察期望的接触状态
        if self.cfg.env.priv_observe_desired_contact_states:
            # 直接将期望接触状态加入到当前时间步的特权观察缓冲区，无需归一化处理
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.desired_contact_states), dim=-1)

        if self.cfg.env.priv_observe_heights:
            F_F_z = torch.zeros(self.num_envs, 36, dtype=torch.float, device=self.device, requires_grad=False)
            heights = torch.from_numpy(self.terrain.get_height()).to(self.device)
            heights = heights.unsqueeze(0).float()
            heights = F.pad(heights, (1, 1, 1, 1), mode='replicate').squeeze(0).to(torch.int16)

            # 将foot_positions数据一次性传输到GPU上
            foot_positions_gpu = self.foot_positions.to(self.device)

            # 使用PyTorch的高级索引功能
            F_F_x = (foot_positions_gpu[:, 0, 0] // 1 * 10).long()
            F_F_y = (foot_positions_gpu[:, 0, 1] // 1 * 10).long()
            F_R_x = (foot_positions_gpu[:, 1, 0] // 1 * 10).long()
            F_R_y = (foot_positions_gpu[:, 1, 1] // 1 * 10).long()
            R_F_x = (foot_positions_gpu[:, 2, 0] // 1 * 10).long()
            R_F_y = (foot_positions_gpu[:, 2, 1] // 1 * 10).long()
            R_R_x = (foot_positions_gpu[:, 3, 0] // 1 * 10).long()
            R_R_y = (foot_positions_gpu[:, 3, 1] // 1 * 10).long()

            # 使用PyTorch的高级索引功能，避免使用循环
            F_F_z = torch.stack([
                heights[F_F_x - 1, F_F_y - 1], heights[F_F_x, F_F_y - 1], heights[F_F_x + 1, F_F_y - 1],
                heights[F_F_x - 1, F_F_y], heights[F_F_x, F_F_y], heights[F_F_x + 1, F_F_y],
                heights[F_F_x - 1, F_F_y + 1], heights[F_F_x, F_F_y + 1], heights[F_F_x + 1, F_F_y + 1],

                heights[F_R_x - 1, F_R_y - 1], heights[F_R_x, F_R_y - 1], heights[F_R_x + 1, F_R_y - 1],
                heights[F_R_x - 1, F_R_y], heights[F_R_x, F_R_y], heights[F_R_x + 1, F_R_y],
                heights[F_R_x - 1, F_R_y + 1], heights[F_R_x, F_R_y + 1], heights[F_R_x + 1, F_R_y + 1],

                heights[R_F_x - 1, R_F_y - 1], heights[R_F_x, R_F_y - 1], heights[R_F_x + 1, R_F_y - 1],
                heights[R_F_x - 1, R_F_y], heights[R_F_x, R_F_y], heights[R_F_x + 1, R_F_y],
                heights[R_F_x - 1, R_F_y + 1], heights[R_F_x, R_F_y + 1], heights[R_F_x + 1, R_F_y + 1],

                heights[R_R_x - 1, R_R_y - 1], heights[R_R_x, R_R_y - 1], heights[R_R_x + 1, R_R_y - 1],
                heights[R_R_x - 1, R_R_y], heights[R_R_x, R_R_y], heights[R_R_x + 1, R_R_y],
                heights[R_R_x - 1, R_R_y + 1], heights[R_R_x, R_R_y + 1], heights[R_R_x + 1, R_R_y + 1]
            ], dim=1)

            F_F_z = F_F_z.to(self.device)

            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, F_F_z), dim=-1)

        # 断言检查以确保特权观察缓冲区的维度符合预期
        assert self.privileged_obs_buf.shape[1] == self.cfg.env.num_privileged_obs, \
            (f"num_privileged_obs ({self.cfg.env.num_privileged_obs}) != the number of privileged observations "
             f"({self.privileged_obs_buf.shape[1]}), you will discard data from the student!")

    def create_sim(self):
        """
        创建模拟环境、地形和环境。
        """
        self.up_axis_idx = 2  # 设置上轴索引为2，表示z轴是上方，相应地调整重力方向
        # 创建模拟环境，指定设备ID、图形引擎、物理引擎和模拟参数
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)

        mesh_type = self.cfg.terrain.mesh_type  # 获取地形的网格类型
        # 如果地形类型是高度场或三角网格
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)  # 创建地形实例

        # 根据地形网格类型创建相应的地形
        if mesh_type == 'plane':
            self._create_ground_plane()     # 创建平面地形
        elif mesh_type == 'heightfield':
            self._create_heightfield()      # 创建高度场地形
        elif mesh_type == 'trimesh':
            self._create_trimesh()          # 创建三角网格地形
        elif mesh_type is not None:
            # 如果地形类型不是已知类型，则抛出错误
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

        self._create_envs()  # 创建环境实例

    def set_camera(self, position, lookat):
        """
        设置相机的位置和观察方向。
        """
        # 创建相机位置的Vec3对象
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        # 创建相机观察目标点的Vec3对象
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        # 设置仿真环境中的相机视角
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def set_main_agent_pose(self, loc, quat):
        """
        设置主代理的位置和旋转姿态。

        参数:
        - loc: 位置，一个包含三个元素的列表或元组，分别对应于XYZ轴的位置。
        - quat: 旋转，一个包含四个元素的列表或元组，表示代理的四元数旋转（通常用于表示3D空间中的旋转）。

        此函数将主代理的位置和旋转姿态更新为给定的参数值，并将这些新的状态应用到仿真环境中。
        """

        # 更新位置：将传入的位置参数loc转换为PyTorch张量，并设置为root_states的前三个元素。
        # root_states是一个存储所有代理状态的PyTorch张量，第一个代理（通常是主代理）的状态被存储在第一行。
        self.root_states[0, 0:3] = torch.Tensor(loc)

        # 更新旋转：同样地，将传入的旋转参数quat转换为PyTorch张量，并设置为root_states的接下来四个元素。
        self.root_states[0, 3:7] = torch.Tensor(quat)

        # 应用更新：使用NVIDIA Gym的set_actor_root_state_tensor方法将更新后的root_states应用到仿真环境的代理上。
        # gymtorch.unwrap_tensor是一个辅助函数，用于将PyTorch张量转换回Gym可以处理的格式。
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _call_train_eval(self, func, env_ids):
        """
        在训练和评估环境中调用特定的函数。

        参数:
        - func: 需要被调用的函数，该函数接收环境ID列表和配置作为参数。
        - env_ids: 一个包含环境ID的数组，这些ID表示应该在哪些环境上调用函数。

        返回值:
        - ret: func在所有指定环境上调用的结果。如果既有训练环境也有评估环境的结果，则这两个结果会被合并。
        """

        # 根据环境ID将环境分为训练集和评估集
        env_ids_train = env_ids[env_ids < self.num_train_envs]
        env_ids_eval = env_ids[env_ids >= self.num_train_envs]

        # 初始化返回值
        ret, ret_eval = None, None

        # 如果训练环境ID列表不为空，则在这些环境上调用func函数
        if len(env_ids_train) > 0:
            ret = func(env_ids_train, self.cfg)

        # 如果评估环境ID列表不为空，则在这些环境上调用func函数，但使用评估配置
        if len(env_ids_eval) > 0:
            ret_eval = func(env_ids_eval, self.eval_cfg)

        # 如果在训练环境和评估环境上都有返回结果，则将这两个结果合并
        if ret is not None and ret_eval is not None:
            ret = torch.cat((ret, ret_eval), axis=-1)

        return ret

    def _randomize_gravity(self, external_force=None):
        """
        随机化或设置仿真环境的重力。

        参数:
        - external_force: 可选，一个三元素的张量，指定应用于仿真环境的外部力（重力）。

        此方法根据配置随机生成一个重力向量，或使用提供的外部力更新仿真环境的重力。
        """

        # 如果提供了外部力，则使用这个外部力作为重力向量
        if external_force is not None:
            self.gravities[:, :] = external_force.unsqueeze(0)
        # 否则，如果配置了随机化重力，根据配置的范围随机生成一个重力向量
        elif self.cfg.domain_rand.randomize_gravity:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity

            self.gravities[:, :] = external_force.unsqueeze(0)

        # 获取当前仿真的参数
        sim_params = self.gym.get_sim_params(self.sim)

        # 计算新的重力向量，这里将随机生成的向量或提供的向量与地球的重力（向下9.8m/s²）相加
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)

        # 更新重力向量的单位向量，用于可能的后续计算
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)

        # 更新仿真环境的重力参数
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)

    def _process_rigid_shape_props(self, props, env_id):
        """
        存储、更改或随机化每个环境中刚体形状属性的回调函数。
        在环境创建期间被调用。
        基本行为：随机化每个环境的摩擦系数。

        参数:
        - props (List[gymapi.RigidShapeProperties]): 资产每个形状的属性列表。
        - env_id (int): 环境ID。

        返回值:
        - [List[gymapi.RigidShapeProperties]]: 修改后的刚体形状属性列表。
        """

        # 遍历所有形状的属性
        for s in range(len(props)):
            # 为每个形状设置摩擦系数，该值来自于当前环境ID对应的friction_coeffs数组中的值
            props[s].friction = self.friction_coeffs[env_id, 0]
            # 为每个形状设置弹性系数，该值来自于当前环境ID对应的restitutions数组中的值
            props[s].restitution = self.restitutions[env_id, 0]

        # 返回修改后的属性列表
        return props

    def _process_dof_props(self, props, env_id):
        """
        在每个环境创建期间调用的回调函数，用于存储、更改或随机化每个环境的关节自由度(DOF)属性。
        基本行为：存储在URDF中定义的位置、速度和扭矩限制。

        参数:
        - props: 资产每个DOF的属性，类型为numpy.array
        - env_id: 环境ID，类型为int

        返回值:
        - 修改后的DOF属性，类型为numpy.array

        注释:
        - 如果是第一个环境，这个回调将会存储每个DOF的位置、速度和扭矩限制。
        - 还会根据配置文件中的设置调整软限制。
        """
        # 如果是第一个环境
        if env_id == 0:
            # 初始化位置限制
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            # 初始化速度限制
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            # 初始化扭矩限制
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            # 遍历所有DOF的属性
            for i in range(len(props)):
                # 存储每个DOF的位置限制
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                # 存储每个DOF的速度限制
                self.dof_vel_limits[i] = props["velocity"][i].item()
                # 存储每个DOF的扭矩限制
                self.torque_limits[i] = props["effort"][i].item()
                # 计算并设置软限制
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2  # 计算中点
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]  # 计算范围
                # 根据配置文件中的软限制比例调整位置限制
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        # 返回修改后的DOF属性
        return props

    def _randomize_rigid_body_props(self, env_ids, cfg):
        """
        随机化指定环境中刚体的物理属性。

        参数:
        - env_ids: 环境ID的列表，指明哪些环境需要被随机化。
        - cfg: 配置对象，包含随机化参数的范围和开关。

        功能:
        根据配置，随机化每个指定环境的基础质量、质心位移、摩擦系数和弹性系数。
        """

        # 随机化基础质量
        if cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = cfg.domain_rand.added_mass_range
            # 为指定环境生成在指定范围内的随机质量
            self.payloads[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (max_payload - min_payload) + min_payload

        # 随机化质心位移
        if cfg.domain_rand.randomize_com_displacement:
            min_com_displacement, max_com_displacement = cfg.domain_rand.com_displacement_range
            # 为指定环境生成在指定范围内的随机质心位移向量
            self.com_displacements[env_ids, :] = (torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                             requires_grad=False) *
                                                  (max_com_displacement - min_com_displacement) + min_com_displacement)

        # 随机化摩擦系数
        if cfg.domain_rand.randomize_friction:
            min_friction, max_friction = cfg.domain_rand.friction_range
            # 为指定环境生成在指定范围内的随机摩擦系数
            self.friction_coeffs[env_ids, :] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                          requires_grad=False) * (
                                                           max_friction - min_friction) + min_friction

        # 随机化弹性系数
        if cfg.domain_rand.randomize_restitution:
            min_restitution, max_restitution = cfg.domain_rand.restitution_range
            # 为指定环境生成在指定范围内的随机弹性系数
            self.restitutions[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                    requires_grad=False) * (
                                                     max_restitution - min_restitution) + min_restitution

    def refresh_actor_rigid_shape_props(self, env_ids, cfg):
        """
        更新指定环境中演员的刚体形状属性。

        参数:
        - env_ids: 要更新的环境ID列表。
        - cfg: 配置对象，虽然在这个方法中未直接使用，但保留这个参数为了一致性或未来的扩展。

        功能:
        遍历指定的每个环境，更新每个演员的摩擦系数和弹性系数。
        """

        # 遍历所有指定的环境ID
        for env_id in env_ids:
            # 获取当前环境中第一个演员的所有刚体形状属性
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            # 遍历所有刚体（假设它们的数量等于自由度的数量）
            for i in range(self.num_dof):
                # 设置每个刚体形状的摩擦系数为当前环境对应的值
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                # 设置每个刚体形状的弹性系数为当前环境对应的值
                rigid_shape_props[i].restitution = self.restitutions[env_id, 0]

            # 将更新后的刚体形状属性设置回该环境的演员中
            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _randomize_dof_props(self, env_ids, cfg):
        """
        随机化指定环境中的关节自由度属性。

        参数:
        - env_ids: 环境ID列表，指定哪些环境的关节自由度属性需要被随机化。
        - cfg: 配置对象，包含随机化参数的范围。

        功能:
        根据配置，随机化每个指定环境的电机强度、电机偏移、Kp因子和Kd因子。
        """
        # 随机化电机强度
        if cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                          requires_grad=False).unsqueeze(1) * (
                                                           max_strength - min_strength) + min_strength
        # 随机化电机偏移
        if cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False) * (
                                                         max_offset - min_offset) + min_offset
        # 随机化Kp因子
        if cfg.domain_rand.randomize_Kp_factor:
            min_Kp_factor, max_Kp_factor = cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                      max_Kp_factor - min_Kp_factor) + min_Kp_factor
        # 随机化Kd因子
        if cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                      max_Kd_factor - min_Kd_factor) + min_Kd_factor

    def _process_rigid_body_props(self, props, env_id):
        """
        修改指定环境中刚体的质量和质心位置。

        参数:
        - props: 刚体属性的列表，每个元素是一个gymapi.RigidShapeProperties对象。
        - env_id: 环境的ID，用于索引特定环境的属性值。

        返回值:
        - 修改后的刚体属性列表。
        """

        # 保存第一个刚体的默认质量
        self.default_body_mass = props[0].mass

        # 修改第一个刚体的质量，增加相应的额外质量（payload）
        props[0].mass = self.default_body_mass + self.payloads[env_id]

        # 设置第一个刚体的质心位置，根据env_id索引的位移量进行设置
        props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1],
                                   self.com_displacements[env_id, 2])

        # 返回修改后的刚体属性列表
        return props

    def _post_physics_step_callback(self):
        # 防止机器人掉落边缘，通过将它们传送回场地内部
        self._call_train_eval(self._teleport_robots, torch.arange(self.num_envs, device=self.device))

        # 重新采样命令，在达到特定的采样间隔时
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._step_contact_targets()

        # 如果配置了测量地形高度，则进行测量
        # if self.cfg.terrain.measure_heights:
        #     self.measured_heights = self._get_heights()

        # 随机推动机器人，以增加训练的难度和多样性
        self._call_train_eval(self._push_robots, torch.arange(self.num_envs, device=self.device))

        # 随机化关节自由度（DoF）属性，根据设定的随机化间隔
        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero(
            as_tuple=False).flatten()
        self._call_train_eval(self._randomize_dof_props, env_ids)

        # 根据设定的间隔随机化重力
        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        # 在随机化持续时间结束后重置重力
        if int(self.common_step_counter - self.cfg.domain_rand.gravity_rand_duration) % int(
                self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity(torch.tensor([0, 0, 0]))

        # 如果配置了在仿真开始后随机化刚体属性，则执行随机化操作，并刷新演员的刚体形状属性
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)

    def _resample_commands(self, env_ids):
        # 如果env_ids为空，则直接返回，不执行任何操作
        if len(env_ids) == 0:
            return

        # 计算重新采样时间步长
        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        # 确定每个周期的长度，取环境允许的最大长度和时间步长的最小值
        ep_len = min(self.cfg.env.max_episode_length, timesteps)

        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            # 根据当前类别的索引，获取属于该类别的环境ID
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            # 如果env_ids_in_category是单个布尔值（表示只有一个环境ID），或者是长度为1的数组，
            # 那么将它转换为包含一个元素的Tensor，类型为布尔
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            # 如果该类别没有任何环境ID，跳过当前循环迭代
            elif len(env_ids_in_category) == 0:
                continue

            # 使用之前筛选出的类别，从全部环境ID中获取对应类别的环境ID
            env_ids_in_category = env_ids[env_ids_in_category]

            # 初始化任务奖励和成功阈值的空列表
            task_rewards, success_thresholds = [], []
            # 遍历特定的任务关键字，这些任务关键字对应不同的性能指标
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force",
                        "tracking_contacts_shaped_vel"]:
                # 如果当前任务关键字在命令总和的键中，说明我们需要追踪此项任务
                if key in self.command_sums.keys():
                    # 计算并收集每个环境中指定任务的平均奖励，这是通过将总奖励除以周期长度(ep_len)得到的
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                    # 计算并收集对应任务的成功阈值，这是通过乘以特定任务的奖励缩放因子来实现的
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])

            # 获取各环境当前所属的分类bin
            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            # 如果存在成功阈值，意味着有一定的任务完成标准，进而更新课程大纲
            if len(success_thresholds) > 0:
                # 更新课程，传入原始bin、任务奖励、成功阈值以及本地范围
                # 本地范围可能代表了各种参数在课程更新中的权重或重要性
                curriculum.update(old_bins, task_rewards, success_thresholds, local_range=np.array(
                    [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

            # 生成每个环境随机的浮点数，用于后续的环境类别分配
            random_env_floats = torch.rand(len(env_ids), device=self.device)
            # 计算每个类别的概率，这里假定每个类别被选择的概率是均等的
            probability_per_category = 1. / len(self.category_names)
            # 根据生成的随机浮点数，为每个环境重新分配类别
            category_env_ids = [
                env_ids[
                    torch.logical_and(
                        probability_per_category * i <= random_env_floats,
                        random_env_floats < probability_per_category * (i + 1))
                ]
                for i in range(len(self.category_names))
            ]

        # 遍历每个类别及其对应的环境ID和教程
        for i, (category, env_ids_in_category, curriculum) in enumerate(
                zip(self.category_names, category_env_ids, self.curricula)):

            # 获取该类别下环境的数量
            batch_size = len(env_ids_in_category)
            # 如果没有环境属于这个类别，继续下一个循环
            if batch_size == 0:
                continue

            # 从当前教程中采样新命令和新的bin索引，采样数量等于批次大小
            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)

            # 更新环境的命令bins和类别
            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i

            # 更新环境的命令，只选择配置文件中指定的命令数量
            # 并将新命令转换成适当的数据类型和设备
            self.commands[env_ids_in_category, :] = torch.Tensor(new_commands[:, :self.cfg.commands.num_commands]).to(
                self.device)

        # 如果命令的数量超过5个
        if self.cfg.commands.num_commands > 5:
            # 检查是否设置了针对不同步态的课程
            if self.cfg.commands.gaitwise_curricula:
                # 遍历每个类别及其对应的环境ID
                for i, (category, env_ids_in_category) in enumerate(zip(self.category_names, category_env_ids)):
                    # 对于“pronk”步态，调整命令6、7、8的值，使其在0到1之间循环，具体为原值的一半减去0.25后模1
                    if category == "pronk":
                        self.commands[env_ids_in_category, 5] = (self.commands[env_ids_in_category, 5] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 6] = (self.commands[env_ids_in_category, 6] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 7] = (self.commands[env_ids_in_category, 7] / 2 - 0.25) % 1
                    # 对于“trot”步态，将命令6的值调整为原值的一半加上0.25，命令7和8的值设为0
                    elif category == "trot":
                        self.commands[env_ids_in_category, 5] = self.commands[env_ids_in_category, 5] / 2 + 0.25
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = 0
                    # 对于“pace”步态，将命令7的值调整为原值的一半加上0.25，命令6和8的值设为0
                    elif category == "pace":
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = self.commands[env_ids_in_category, 6] / 2 + 0.25
                        self.commands[env_ids_in_category, 7] = 0
                    # 对于“bound”步态，将命令8的值调整为原值的一半加上0.25，命令6和7的值设为0
                    elif category == "bound":
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = self.commands[env_ids_in_category, 7] / 2 + 0.25

            # 根据配置调整步态分布，确保不同步态的均衡分布
            if self.cfg.commands.balance_gait_distribution:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                # 根据随机值分配步态
                pronking_envs = env_ids[random_env_floats <= 0.25]
                trotting_envs = env_ids[torch.logical_and(0.25 <= random_env_floats, random_env_floats < 0.50)]
                pacing_envs = env_ids[torch.logical_and(0.50 <= random_env_floats, random_env_floats < 0.75)]
                bounding_envs = env_ids[0.75 <= random_env_floats]
                # 调整命令以符合不同的步态
                self.commands[pronking_envs, 5] = (self.commands[pronking_envs, 5] / 2 - 0.25) % 1
                self.commands[trotting_envs, 5] = self.commands[trotting_envs, 5] / 2 + 0.25
                self.commands[pacing_envs, 6] = self.commands[pacing_envs, 6] / 2 + 0.25
                self.commands[bounding_envs, 7] = self.commands[bounding_envs, 7] / 2 + 0.25

        # 如果命令存在二元相位，则对命令进行调整
        if self.cfg.commands.binary_phases:
            self.commands[env_ids, 5] = (torch.round(2 * self.commands[env_ids, 5])) / 2.0 % 1
            self.commands[env_ids, 6] = (torch.round(2 * self.commands[env_ids, 6])) / 2.0 % 1
            self.commands[env_ids, 7] = (torch.round(2 * self.commands[env_ids, 7])) / 2.0 % 1

        # 设置较小的命令值为零
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # 重置命令累计值
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0

    def _step_contact_targets(self):
        # 如果配置中要求观察步态指令，则进行以下处理
        if self.cfg.env.observe_gait_commands:
            # 从命令中提取频率、相位、偏移量、边界和持续时间
            frequencies = self.commands[:, 4]
            phases = self.commands[:, 5]
            offsets = self.commands[:, 6]
            bounds = self.commands[:, 7]
            durations = self.commands[:, 8]

            # 根据时间增量和频率更新步态索引
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

            # 根据配置中的步态偏移量来调整足部指标的计算
            if self.cfg.commands.pacing_offset:
                foot_indices = [self.gait_indices + phases + offsets + bounds,
                                self.gait_indices + bounds,
                                self.gait_indices + offsets,
                                self.gait_indices + phases]
            else:
                foot_indices = [self.gait_indices + phases + offsets + bounds,
                                self.gait_indices + offsets,
                                self.gait_indices + bounds,
                                self.gait_indices + phases]

            # 将足部指标合并为一个张量，并确保所有值都在0到1之间
            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

            # 对每个足部指标进行遍历，分别计算站立和摆动状态的指标
            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = (0.5 + (torch.remainder(idxs[swing_idxs], 1) -
                                           durations[swing_idxs]) * (0.5 / (1 - durations[swing_idxs])))

            # 为四个足部指标计算时钟输入信号
            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            # 为四个足部指标计算双倍频率的时钟输入信号
            self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
            self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
            self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
            self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

            # 为四个足部指标计算半频率的时钟输入信号
            self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
            self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
            self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
            self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

            # 使用冯·米塞斯分布来平滑期望接触状态
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

            # 计算每个足部的平滑乘数
            smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) *
                                       (1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) *
                                       (1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))

            smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) *
                                       (1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) *
                                       (1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))

            smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) *
                                       (1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) *
                                       (1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))

            smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) *
                                       (1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) *
                                       (1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

            # 设置四个足部的期望接触状态
            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        # 如果命令数量超过9，则设置期望的足部摆动高度
        if self.cfg.commands.num_commands > 9:
            self.desired_footswing_height = self.commands[:, 9]

    def _compute_torques(self, actions):
        """
        从动作中计算扭矩。
        动作可以解释为给PD控制器的位置或速度目标，或者直接解释为缩放后的扭矩。
        注意：扭矩的维度必须与自由度的数量相同，即使某些自由度未被驱动。

        参数:
            actions (torch.Tensor): 动作

        返回值:
            [torch.Tensor]: 发送到仿真的扭矩
        """

        # PD控制器部分
        actions_scaled = actions[:, :12] * self.cfg.control.action_scale  # 缩放前12个动作
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # 缩减髋关节屈伸范围的缩放

        # 如果配置中启用了时延随机化，更新滞后缓冲区并计算目标关节位置
        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
        else:
            self.joint_pos_target = actions_scaled + self.default_dof_pos  # 否则直接计算目标关节位置

        control_type = self.cfg.control.control_type  # 控制类型

        # 根据控制类型计算扭矩
        if control_type == "actuator_net":
            # 如果是执行器网络类型，根据执行器网络计算扭矩
            self.joint_pos_err = self.dof_pos - self.joint_pos_target + self.motor_offsets  # 计算关节位置误差
            self.joint_vel = self.dof_vel  # 获取关节速度
            # 使用执行器网络计算扭矩
            torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                                            self.joint_vel, self.joint_vel_last, self.joint_vel_last_last)
            # 更新误差和速度的历史值
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_pos_err_last = torch.clone(self.joint_pos_err)
            self.joint_vel_last_last = torch.clone(self.joint_vel_last)
            self.joint_vel_last = torch.clone(self.joint_vel)
        elif control_type == "P":
            # 如果是P控制类型，使用比例增益计算扭矩
            torques = (self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos + self.motor_offsets) -
                       self.d_gains * self.Kd_factors * self.dof_vel)
        else:
            # 未知控制类型抛出异常
            raise NameError(f"Unknown controller type: {control_type}")

        # 根据电机强度调整扭矩，并限制扭矩值在允许的范围内
        torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids, cfg):
        """
        重置特定环境ID的关节位置和速度

        参数:
        - env_ids: 一个包含环境ID的张量，指定哪些环境的关节状态需要被重置

        注解:
        1. 将关节位置设置为默认位置的随机缩放值。
        2. 将关节速度设置为零。
        3. 使用Gym API更新仿真环境中的关节状态。

        注释:
        - 使用torch_rand_float函数生成随机缩放因子，用于初始化关节位置。
        - 将env_ids张量转换为int32类型，以符合Gym API的要求。
        - 使用gym.set_dof_state_tensor_indexed函数根据env_ids更新仿真环境中的关节状态。
        """
        # 将关节位置设置为默认位置的随机缩放值
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            0.5, 1.5, (len(env_ids), self.num_dof), device=self.device
        )
        # 将关节速度设置为零
        self.dof_vel[env_ids] = 0.

        # 将env_ids张量转换为int32类型
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # 使用Gym API更新仿真环境中的关节状态
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids, cfg):
        """
        重置选定环境的根状态位置和速度。
        基于课程设置基础位置，并在-0.5到0.5[m/s, rad/s]范围内随机选择基础速度。

        参数:
            env_ids (List[int]): 环境ID列表
        """

        # 基础位置设置
        if self.custom_origins:
            # 如果自定义起点被启用
            self.root_states[env_ids] = self.base_init_state  # 将选定环境的根状态设置为基础初始状态
            self.root_states[env_ids, :3] += self.env_origins[env_ids]  # 加上环境起点偏移量
            # 随机添加x和y方向的初始位置偏移
            self.root_states[env_ids, 0:1] += torch_rand_float(-cfg.terrain.x_init_range, cfg.terrain.x_init_range,
                                                               (len(env_ids), 1), device=self.device)
            self.root_states[env_ids, 1:2] += torch_rand_float(-cfg.terrain.y_init_range, cfg.terrain.y_init_range,
                                                               (len(env_ids), 1), device=self.device)
            # 添加x和y方向的初始位置偏移常量
            self.root_states[env_ids, 0] += cfg.terrain.x_init_offset
            self.root_states[env_ids, 1] += cfg.terrain.y_init_offset
        else:
            # 如果未启用自定义起点
            self.root_states[env_ids] = self.base_init_state  # 设置基础初始状态
            self.root_states[env_ids, :3] += self.env_origins[env_ids]  # 加上环境起点偏移量

        # 基础偏航角
        init_yaws = torch_rand_float(-cfg.terrain.yaw_init_range, cfg.terrain.yaw_init_range, (len(env_ids), 1),
                                     device=self.device)  # 随机生成初始偏航角
        quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]  # 将偏航角转换为四元数
        self.root_states[env_ids, 3:7] = quat  # 更新根状态的旋转部分

        # 基础速度
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # 随机生成线性和角速度

        # 将根状态更新到仿真中
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # 视频录制处理
        if cfg.env.record_video and 0 in env_ids:
            # 如果需要录制视频且包含ID为0的环境
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames = self.video_frames[:]  # 保存已有的视频帧
            self.video_frames = []  # 重置视频帧列表

        if cfg.env.record_video and self.eval_cfg is not None and self.num_train_envs in env_ids:
            # 如果在评估模式下需要录制视频且包含训练环境的ID
            if self.complete_video_frames_eval is None:
                self.complete_video_frames_eval = []
            else:
                self.complete_video_frames_eval = self.video_frames_eval[:]  # 保存已有的评估视频帧
            self.video_frames_eval = []  # 重置评估视频帧列表

    def _push_robots(self, env_ids, cfg):
        """
        随机对机器人施加推力。通过设置随机的基础速度来模拟冲击效果。
        """

        # 如果配置中启用了对机器人施加推力
        if cfg.domain_rand.push_robots:
            # 根据推力间隔筛选出应施加推力的环境ID
            env_ids = env_ids[self.episode_length_buf[env_ids] % int(cfg.domain_rand.push_interval) == 0]

            max_vel = cfg.domain_rand.max_push_vel_xy  # 获取最大推力速度限制

            # 为选定的环境ID设置随机的线性速度（x/y方向）
            self.root_states[env_ids, 7:9] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 2), device=self.device)

            # 将更新后的根状态应用到仿真中
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _teleport_robots(self, env_ids, cfg):
        """
        如果机器人过于靠近边缘，则将它们瞬移到另一侧。
        """

        # 如果配置中启用了机器人的瞬移功能
        if cfg.terrain.teleport_robots:
            thresh = cfg.terrain.teleport_thresh  # 获取瞬移阈值

            x_offset = int(cfg.terrain.x_offset * cfg.terrain.horizontal_scale)  # 计算x轴偏移量

            # 筛选出需要在x轴正方向上瞬移的环境ID
            low_x_ids = env_ids[self.root_states[env_ids, 0] < thresh + x_offset]
            # 对这些环境的机器人在x轴上进行正向瞬移
            self.root_states[low_x_ids, 0] += cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            # 筛选出需要在x轴负方向上瞬移的环境ID
            high_x_ids = env_ids[
                self.root_states[env_ids, 0] > cfg.terrain.terrain_length * cfg.terrain.num_rows - thresh + x_offset]
            # 对这些环境的机器人在x轴上进行负向瞬移
            self.root_states[high_x_ids, 0] -= cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            # 筛选出需要在y轴正方向上瞬移的环境ID
            low_y_ids = env_ids[self.root_states[env_ids, 1] < thresh]
            # 对这些环境的机器人在y轴上进行正向瞬移
            self.root_states[low_y_ids, 1] += cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            # 筛选出需要在y轴负方向上瞬移的环境ID
            high_y_ids = env_ids[
                self.root_states[env_ids, 1] > cfg.terrain.terrain_width * cfg.terrain.num_cols - thresh]
            # 对这些环境的机器人在y轴上进行负向瞬移
            self.root_states[high_y_ids, 1] -= cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            # 更新仿真中的机器人根状态
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            # 刷新仿真中的机器人根状态信息
            self.gym.refresh_actor_root_state_tensor(self.sim)

    def _update_terrain_curriculum(self, env_ids):
        """
        根据机器人的行走距离更新地形难度等级

        参数:
        - env_ids: 一个包含环境ID的张量，指定哪些环境的地形难度等级需要更新

        注解:
        1. 如果初始化还未完成，则不进行更新。
        2. 计算机器人从起点走过的距离。
        3. 根据走过的距离决定地形难度等级的提升或降低。
        4. 对于解决了最高难度等级的机器人，随机分配一个新的难度等级。
        5. 更新环境原点以匹配新的地形难度等级。

        注释:
        - 使用torch.norm函数计算机器人的行走距离。
        - 使用逻辑运算符确定哪些机器人应该升级或降级地形难度等级。
        - 使用torch.where和torch.clip函数确保地形难度等级在有效范围内。
        - 更新env_origins以反映地形难度等级的变化。
        """
        # 实施地形课程
        if not self.init_done:
            # 如果初始化还未完成，则不进行更新
            return
        # 计算机器人从起点走过的距离
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # 走得足够远的机器人进入更难的地形
        move_up = distance > self.terrain.cfg.env_length / 2
        # 走得不到一半所需距离的机器人进入更简单的地形
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        # 更新地形难度等级
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # 解决了最高难度等级的机器人被随机分配一个新的难度等级
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.cfg.terrain.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.cfg.terrain.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0)  # 最低难度等级为零
        )
        # 更新环境原点以匹配新的地形难度等级
        self.env_origins[env_ids] = self.cfg.terrain.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """
        根据跟踪线速度的奖励表现来更新命令范围

        参数:
        - env_ids: 一个包含环境ID的张量，指定哪些环境的命令范围需要更新

        注解:
        1. 如果跟踪线速度的平均奖励超过最大奖励的80%，则增加命令的范围。
        2. 使用numpy的clip函数确保命令范围在配置允许的最大课程范围内。

        注释:
        - 使用torch.mean函数计算特定环境ID的跟踪线速度奖励的平均值。
        - 使用self.max_episode_length和self.reward_scales来计算奖励的比例。
        - 根据奖励的表现调整命令范围。
        - 使用np.clip函数确保命令范围的更新在合理的限制内。
        """
        # 如果跟踪奖励超过最大值的80%，增加命令范围
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            # 减少线速度x的最小命令范围，但不低于配置中的最大课程负值
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5,
                -self.cfg.commands.max_forward_curriculum,
                0.
            )
            # 增加线速度x的最大命令范围，但不超过配置中的最大课程值
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5,
                0.,
                self.cfg.commands.max_forward_curriculum
            )

    def _get_noise_scale_vec(self, cfg):
        """
        设置用于缩放添加到观察结果中的噪声的向量。
        注意：当改变观察结果结构时，必须适配此向量。

        参数:
            cfg (Dict): 环境配置文件

        返回值:
            [torch.Tensor]: 用于乘以[-1, 1]中的均匀分布的缩放向量
        """

        # 初始化噪声添加标志和噪声缩放参数
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise_scales
        noise_level = self.cfg.noise.noise_level

        # 构造基础噪声向量
        noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,  # 重力噪声
                               torch.ones(self.num_actuated_dof) * noise_scales.dof_pos *
                               noise_level * self.obs_scales.dof_pos,
                               # 关节位置噪声
                               torch.ones(self.num_actuated_dof) * noise_scales.dof_vel *
                               noise_level * self.obs_scales.dof_vel,
                               # 关节速度噪声
                               torch.zeros(self.num_actions),  # 动作噪声（初始化为0）
                               ), dim=0)

        # 根据不同的观察配置添加噪声
        if self.cfg.env.observe_command:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,
                                   torch.zeros(self.cfg.commands.num_commands),  # 命令噪声
                                   torch.ones(self.num_actuated_dof) * noise_scales.dof_pos * noise_level *
                                   self.obs_scales.dof_pos, torch.ones(self.num_actuated_dof) *
                                   noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                                   torch.zeros(self.num_actions),
                                   ), dim=0)

        # 如果观察到过去两次动作
        if self.cfg.env.observe_two_prev_actions:
            noise_vec = torch.cat((noise_vec, torch.zeros(self.num_actions)), dim=0)

        # 如果观察到时间参数
        if self.cfg.env.observe_timing_parameter:
            noise_vec = torch.cat((noise_vec, torch.zeros(1)), dim=0)

        # 如果观察到时钟输入
        if self.cfg.env.observe_clock_inputs:
            noise_vec = torch.cat((noise_vec, torch.zeros(4)), dim=0)

        # 如果观察到线速度和角速度
        if self.cfg.env.observe_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   torch.ones(3) * noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel,
                                   noise_vec
                                   ), dim=0)

        # 如果仅观察到线速度
        if self.cfg.env.observe_only_lin_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   noise_vec
                                   ), dim=0)

        # 如果观察到偏航角
        if self.cfg.env.observe_yaw:
            noise_vec = torch.cat((noise_vec, torch.zeros(1)), dim=0)

        # 如果观察到接触状态
        if self.cfg.env.observe_contact_states:
            noise_vec = torch.cat((noise_vec, torch.ones(4) * noise_scales.contact_states * noise_level), dim=0)

        # 将噪声向量移至设定的设备上
        noise_vec = noise_vec.to(self.device)

        return noise_vec

    def _init_buffers(self):
        """ 初始化将包含仿真状态和处理量的PyTorch张量 """
        # 获取Gym环境的GPU状态张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # 获取参与者根状态张量
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # 获取自由度状态张量
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)  # 获取接触力张量
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  # 获取刚体状态张量

        # 刷新这些张量以确保它们包含最新的仿真数据
        self.gym.refresh_dof_state_tensor(self.sim)  # 刷新自由度状态张量
        self.gym.refresh_actor_root_state_tensor(self.sim)  # 刷新参与者根状态张量
        self.gym.refresh_net_contact_force_tensor(self.sim)  # 刷新接触力张量
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # 刷新刚体状态张量

        # 渲染所有相机传感器的数据
        self.gym.render_all_camera_sensors(self.sim)  # 渲染所有相机传感器

        # 创建不同切片的包装张量
        self.root_states = gymtorch.wrap_tensor(actor_root_state)  # 将参与者根状态张量包装为GymTorch张量
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)  # 将自由度状态张量包装为GymTorch张量
        # 将接触力张量包装为GymTorch张量，并取前num_envs * num_bodies行
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]  # 从自由度状态中提取位置信息
        self.base_pos = self.root_states[:self.num_envs, 0:3]  # 提取每个环境的基座位置
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]  # 从自由度状态中提取速度信息
        self.base_quat = self.root_states[:self.num_envs, 3:7]  # 提取每个环境的基座四元数
        # 将刚体状态张量包装为GymTorch张量，并取前num_envs * num_bodies行
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies, :]
        # 从刚体状态中提取脚部速度，首先将刚体状态张量变形，然后取出特定脚部索引的速度信息
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                               7:10]
        # 从刚体状态中提取脚部位置，同样首先变形，然后取特定脚部索引的位置信息
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.prev_base_pos = self.base_pos.clone()  # 克隆当前的基座位置，用于之后的比较
        self.prev_foot_velocities = self.foot_velocities.clone()  # 克隆当前的脚部速度

        # 初始化滞后缓冲区，用于存储过去的自由度位置信息
        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps + 1)]

        # 将接触力张量重新包装并变形为(num_envs, num_bodies, xyz轴)的形状，便于处理
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :].view(
            self.num_envs, -1, 3)

        # 初始化一些稍后将使用的数据
        self.common_step_counter = 0  # 通用步骤计数器
        self.extras = {}  # 用于存储额外数据的字典

        # 如果配置要求测量高度，则初始化高度测量点
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0  # 初始化测量到的高度为0

        # 获取噪声缩放向量
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        # 初始化重力向量，并重复num_envs次以匹配环境数量
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        # 初始化前进向量，并重复num_envs次
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        # 初始化力矩、P增益、D增益和动作张量，全部设置为零
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        # 初始化上一次和上上次的动作张量，以及关节位置目标张量
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                            requires_grad=False)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                                      device=self.device, requires_grad=False)
        # 初始化上一次的关节速度和根部速度
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        # 初始化命令值和命令张量，以及命令缩放因子
        self.commands_value = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                          device=self.device, requires_grad=False)
        self.commands = torch.zeros_like(self.commands_value)  # 初始化命令张量
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel,
                                            self.obs_scales.lin_vel,
                                            self.obs_scales.ang_vel,
                                            self.obs_scales.body_height_cmd,
                                            self.obs_scales.gait_freq_cmd,
                                            self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.footswing_height_cmd,
                                            self.obs_scales.body_pitch_cmd,
                                            self.obs_scales.body_roll_cmd,
                                            self.obs_scales.stance_width_cmd,
                                            self.obs_scales.stance_length_cmd,
                                            self.obs_scales.aux_reward_cmd],
                                           device=self.device, requires_grad=False)[:self.cfg.commands.num_commands]
        # 初始化期望接触状态张量
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False)

        # 初始化脚部空中时间和上一次接触状态
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.last_contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool,
                                             device=self.device, requires_grad=False)
        # 初始化基座的线速度和角速度
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])
        # 计算重力在基座坐标系中的投影
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 初始化关节位置偏移和PD增益
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):  # 遍历所有自由度
            name = self.dof_names[i]  # 获取当前自由度的名称
            angle = self.cfg.init_state.default_joint_angles[name]  # 获取默认的关节角度
            self.default_dof_pos[i] = angle  # 设置默认的关节位置
            found = False  # 初始化找到标志为False
            for dof_name in self.cfg.control.stiffness.keys():  # 遍历所有设定的刚度值
                if dof_name in name:  # 如果当前自由度名称包含在刚度键中
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]  # 设置比例增益
                    self.d_gains[i] = self.cfg.control.damping[dof_name]  # 设置微分增益
                    found = True  # 标记为找到
            if not found:  # 如果没有找到对应的增益
                self.p_gains[i] = 0.  # 设置比例增益为0
                self.d_gains[i] = 0.  # 设置微分增益为0
                if self.cfg.control.control_type in ["P", "V"]:  # 如果控制类型为P或V
                    print(f"PD gain of joint {name} were not defined, setting them to zero")  # 打印未定义PD增益的警告
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)  # 将默认关节位置张量增加一个维度

        # 如果控制类型为"actuator_net"，加载执行器网络
        if self.cfg.control.control_type == "actuator_net":
            # 构建执行器网络的路径
            actuator_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/actuator_nets/unitree_go1.pt'
            actuator_network = torch.jit.load(actuator_path).to(self.device)  # 加载执行器网络并移动到设备上

            # 定义执行器网络的求值函数
            def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last):
                # 将各个状态张量拼接
                xs = torch.cat((joint_pos.unsqueeze(-1),
                                joint_pos_last.unsqueeze(-1),
                                joint_pos_last_last.unsqueeze(-1),
                                joint_vel.unsqueeze(-1),
                                joint_vel_last.unsqueeze(-1),
                                joint_vel_last_last.unsqueeze(-1)), dim=-1)
                # 使用执行器网络计算力矩
                torques = actuator_network(xs.view(self.num_envs * 12, 6))
                # 将力矩张量变形回原始维度
                return torques.view(self.num_envs, 12)

            # 将求值函数赋值给self.actuator_network
            self.actuator_network = eval_actuator_network

            # 初始化关节位置误差和关节速度的历史记录张量
            self.joint_pos_err_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_pos_err_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last = torch.zeros((self.num_envs, 12), device=self.device)

    def _init_custom_buffers__(self):
        # 领域随机化属性
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 4, dtype=torch.float,
                                                                  device=self.device, requires_grad=False)  # 初始化摩擦系数
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, 4, dtype=torch.float,
                                                                  device=self.device, requires_grad=False)  # 初始化恢复系数
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)  # 初始化负载量
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)  # 初始化质心偏移量
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)  # 初始化电机强度
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)  # 初始化电机偏移量
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)  # 初始化比例增益因子
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)  # 初始化微分增益因子
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                     requires_grad=False)  # 初始化重力向量
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))  # 根据上轴索引计算重力向量并重复

        # 如果在初始化中传入了自定义动力学参数值，则在此处设置它们
        dynamics_params = ["friction_coeffs", "restitutions", "payloads", "com_displacements", "motor_strengths",
                           "Kp_factors", "Kd_factors"]
        if self.initial_dynamics_dict is not None:  # 如果传入了初始动力学参数字典
            for k, v in self.initial_dynamics_dict.items():  # 遍历字典中的每一项
                if k in dynamics_params:  # 如果键在动力学参数列表中
                    setattr(self, k, v.to(self.device))  # 设置对应属性的值，并确保其位于正确的设备上

        # 其他动力学参数
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)  # 初始化步态指标
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)  # 初始化时钟输入
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                   requires_grad=False)  # 初始化双倍时钟输入
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                 requires_grad=False)  # 初始化半时钟输入

    def _init_command_distribution(self, env_ids):
        # 初始化命令分布配置
        # 设置默认的类别名称为['nominal']
        self.category_names = ['nominal']

        # 如果配置中指定了按步态分的课程（gaitwise_curricula），则更新类别名称
        if self.cfg.commands.gaitwise_curricula:
            self.category_names = ['pronk', 'trot', 'pace', 'bound']

        # 如果配置的课程类型为"RewardThresholdCurriculum"，则导入相应的课程类
        if self.cfg.commands.curriculum_type == "RewardThresholdCurriculum":
            from .curriculum import RewardThresholdCurriculum
            CurriculumClass = RewardThresholdCurriculum

        # 初始化空的课程列表
        self.curricula = []

        # 针对每一个类别名称，创建课程实例并添加到课程列表中
        for category in self.category_names:
            self.curricula += [CurriculumClass(
                seed=self.cfg.commands.curriculum_seed,  # 课程随机种子
                x_vel=(self.cfg.commands.limit_vel_x[0],  # X轴速度限制(最小值, 最大值, 分箱数量)
                       self.cfg.commands.limit_vel_x[1],
                       self.cfg.commands.num_bins_vel_x),
                y_vel=(self.cfg.commands.limit_vel_y[0],  # Y轴速度限制(最小值, 最大值, 分箱数量)
                       self.cfg.commands.limit_vel_y[1],
                       self.cfg.commands.num_bins_vel_y),
                yaw_vel=(self.cfg.commands.limit_vel_yaw[0],  # 偏航速度限制(最小值, 最大值, 分箱数量)
                         self.cfg.commands.limit_vel_yaw[1],
                         self.cfg.commands.num_bins_vel_yaw),
                body_height=(self.cfg.commands.limit_body_height[0],  # 身体高度限制(最小值, 最大值, 分箱数量)
                             self.cfg.commands.limit_body_height[1],
                             self.cfg.commands.num_bins_body_height),
                gait_frequency=(self.cfg.commands.limit_gait_frequency[0],  # 步态频率限制(最小值, 最大值, 分箱数量)
                                self.cfg.commands.limit_gait_frequency[1],
                                self.cfg.commands.num_bins_gait_frequency),
                gait_phase=(self.cfg.commands.limit_gait_phase[0],  # 步态相限制(最小值, 最大值, 分箱数量)
                            self.cfg.commands.limit_gait_phase[1],
                            self.cfg.commands.num_bins_gait_phase),
                gait_offset=(self.cfg.commands.limit_gait_offset[0],  # 步态偏移限制(最小值, 最大值, 分箱数量)
                             self.cfg.commands.limit_gait_offset[1],
                             self.cfg.commands.num_bins_gait_offset),
                gait_bounds=(self.cfg.commands.limit_gait_bound[0],  # 步态界限限制(最小值, 最大值, 分箱数量)
                             self.cfg.commands.limit_gait_bound[1],
                             self.cfg.commands.num_bins_gait_bound),
                gait_duration=(self.cfg.commands.limit_gait_duration[0],  # 步态持续时间限制(最小值, 最大值, 分箱数量)
                               self.cfg.commands.limit_gait_duration[1],
                               self.cfg.commands.num_bins_gait_duration),
                footswing_height=(self.cfg.commands.limit_footswing_height[0],  # 足摆高度限制(最小值, 最大值, 分箱数量)
                                  self.cfg.commands.limit_footswing_height[1],
                                  self.cfg.commands.num_bins_footswing_height),
                body_pitch=(self.cfg.commands.limit_body_pitch[0],  # 身体俯仰限制(最小值, 最大值, 分箱数量)
                            self.cfg.commands.limit_body_pitch[1],
                            self.cfg.commands.num_bins_body_pitch),
                body_roll=(self.cfg.commands.limit_body_roll[0],  # 身体翻滚限制(最小值, 最大值, 分箱数量)
                           self.cfg.commands.limit_body_roll[1],
                           self.cfg.commands.num_bins_body_roll),
                stance_width=(self.cfg.commands.limit_stance_width[0],  # 站姿宽度限制(最小值, 最大值, 分箱数量)
                              self.cfg.commands.limit_stance_width[1],
                              self.cfg.commands.num_bins_stance_width),
                stance_length=(self.cfg.commands.limit_stance_length[0],  # 站姿长度限制(最小值, 最大值, 分箱数量)
                               self.cfg.commands.limit_stance_length[1],
                               self.cfg.commands.num_bins_stance_length),
                aux_reward_coef=(self.cfg.commands.limit_aux_reward_coef[0],  # 辅助奖励系数限制(最小值, 最大值, 分箱数量)
                                 self.cfg.commands.limit_aux_reward_coef[1],
                                 self.cfg.commands.num_bins_aux_reward_coef),
            )]

        # 如果课程类型设置为"LipschitzCurriculum"
        if self.cfg.commands.curriculum_type == "LipschitzCurriculum":
            # 遍历之前创建的课程列表
            for curriculum in self.curricula:
                # 为每个课程设置Lipschitz阈值和二进制阶段参数
                curriculum.set_params(lipschitz_threshold=self.cfg.commands.lipschitz_threshold,
                                      binary_phases=self.cfg.commands.binary_phases)

        # 初始化一个数组来存储环境命令的分箱编号，数组长度等于环境ID的数量
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int64)
        # 初始化一个数组来存储环境命令的类别编号，数组长度也等于环境ID的数量
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int64)

        low = np.array([
            self.cfg.commands.lin_vel_x[0],  # 线性速度X轴的最小值
            self.cfg.commands.lin_vel_y[0],  # 线性速度Y轴的最小值
            self.cfg.commands.ang_vel_yaw[0],  # 角速度偏航角的最小值
            self.cfg.commands.body_height_cmd[0],  # 身体高度命令的最小值
            self.cfg.commands.gait_frequency_cmd_range[0],  # 步态频率命令的范围最小值
            self.cfg.commands.gait_phase_cmd_range[0],  # 步态相位命令的范围最小值
            self.cfg.commands.gait_offset_cmd_range[0],  # 步态偏移命令的范围最小值
            self.cfg.commands.gait_bound_cmd_range[0],  # 步态边界命令的范围最小值
            self.cfg.commands.gait_duration_cmd_range[0],  # 步态持续时间命令的范围最小值
            self.cfg.commands.footswing_height_range[0],  # 足摆高度范围的最小值
            self.cfg.commands.body_pitch_range[0],  # 身体俯仰范围的最小值
            self.cfg.commands.body_roll_range[0],  # 身体翻滚范围的最小值
            self.cfg.commands.stance_width_range[0],  # 站姿宽度范围的最小值
            self.cfg.commands.stance_length_range[0],  # 站姿长度范围的最小值
            self.cfg.commands.aux_reward_coef_range[0],  # 辅助奖励系数范围的最小值
        ])

        high = np.array([
            self.cfg.commands.lin_vel_x[1],  # 线性速度X轴的最大值
            self.cfg.commands.lin_vel_y[1],  # 线性速度Y轴的最大值
            self.cfg.commands.ang_vel_yaw[1],  # 角速度偏航角的最大值
            self.cfg.commands.body_height_cmd[1],  # 身体高度命令的最大值
            self.cfg.commands.gait_frequency_cmd_range[1],  # 步态频率命令的范围最大值
            self.cfg.commands.gait_phase_cmd_range[1],  # 步态相位命令的范围最大值
            self.cfg.commands.gait_offset_cmd_range[1],  # 步态偏移命令的范围最大值
            self.cfg.commands.gait_bound_cmd_range[1],  # 步态边界命令的范围最大值
            self.cfg.commands.gait_duration_cmd_range[1],  # 步态持续时间命令的范围最大值
            self.cfg.commands.footswing_height_range[1],  # 足摆高度范围的最大值
            self.cfg.commands.body_pitch_range[1],  # 身体俯仰范围的最大值
            self.cfg.commands.body_roll_range[1],  # 身体翻滚范围的最大值
            self.cfg.commands.stance_width_range[1],  # 站姿宽度范围的最大值
            self.cfg.commands.stance_length_range[1],  # 站姿长度范围的最大值
            self.cfg.commands.aux_reward_coef_range[1],  # 辅助奖励系数范围的最大值
        ])

        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

    def _prepare_reward_function(self):
        """
        准备用于计算总奖励的奖励函数列表。
        该函数查找所有非零奖励比例的奖励名称对应的奖励函数。
        """

        # 引入奖励容器
        from go1_gym.envs.rewards.corl_rewards import CoRLRewards
        reward_containers = {"CoRLRewards": CoRLRewards}
        # 根据配置文件中的奖励容器名称，实例化奖励容器
        self.reward_container = reward_containers[self.cfg.rewards.reward_container_name](self)

        # 移除奖励比例为0的项，并将非零项乘以时间差(dt)
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)  # 如果奖励比例为0，则移除
            else:
                self.reward_scales[key] *= self.dt  # 如果奖励比例非零，乘以dt

        # 准备奖励函数列表
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue  # 如果是终止奖励，则跳过

            # 检查是否存在对应的奖励函数
            if not hasattr(self.reward_container, '_reward_' + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)  # 添加奖励名称
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))  # 添加奖励函数

        # 初始化奖励累计值
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)

        # 初始化评估模式下的奖励累计值
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)

        # 初始化命令累计值
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}

    def _create_ground_plane(self):
        """
        创建仿真环境中的地面平面

        注解:
        1. 设置地面平面的参数。
        2. 将地面平面添加到仿真环境中。

        注释:
        - 这个函数负责在仿真环境中添加一个地面平面，并设置其物理属性。
        """

        plane_params = gymapi.PlaneParams()  # 创建一个新的平面参数对象
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)  # 设置平面的法向量，表示垂直向上
        plane_params.static_friction = self.cfg.terrain.static_friction  # 设置平面的静摩擦系数
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction  # 设置平面的动摩擦系数
        plane_params.restitution = self.cfg.terrain.restitution  # 设置平面的恢复系数（弹性系数）

        self.gym.add_ground(self.sim, plane_params)  # 将地面平面添加到仿真环境中

    def _create_heightfield(self):
        """
        创建仿真环境中的高度场

        注解:
        1. 设置高度场的参数。
        2. 将高度场添加到仿真环境中。
        3. 将高度样本转换为张量并发送到设备。

        注释:
        - 这个函数负责在仿真环境中添加一个高度场，并设置其物理属性和变换。
        """

        hf_params = gymapi.HeightFieldParams()  # 创建一个新的高度场参数对象
        hf_params.column_scale = self.terrain.cfg.horizontal_scale  # 设置列的水平缩放比例
        hf_params.row_scale = self.terrain.cfg.horizontal_scale  # 设置行的水平缩放比例
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale  # 设置垂直缩放比例
        hf_params.nbRows = self.terrain.tot_cols  # 设置高度场的总行数
        hf_params.nbColumns = self.terrain.tot_rows  # 设置高度场的总列数
        hf_params.transform.p.x = -self.terrain.cfg.border_size  # 设置高度场的x轴位置（平移）
        hf_params.transform.p.y = -self.terrain.cfg.border_size  # 设置高度场的y轴位置（平移）
        hf_params.transform.p.z = 0.0  # 设置高度场的z轴位置（平移）
        hf_params.static_friction = self.cfg.terrain.static_friction  # 设置高度场的静摩擦系数
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction  # 设置高度场的动摩擦系数
        hf_params.restitution = self.cfg.terrain.restitution  # 设置高度场的恢复系数（弹性系数）

        print(self.terrain.heightsamples.shape, hf_params.nbRows, hf_params.nbColumns)

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)  # 将高度场添加到仿真环境中
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)  # 将高度样本转换为张量，并发送到设备

    def _create_trimesh(self):
        """
        创建仿真环境中的三角形网格

        注解:
        1. 设置三角形网格的参数。
        2. 将三角形网格添加到仿真环境中。
        3. 将高度样本转换为张量并发送到设备。

        注释:
        - 这个函数负责在仿真环境中添加一个三角形网格，并设置其物理属性和变换。
        """

        tm_params = gymapi.TriangleMeshParams()  # 创建一个新的三角形网格参数对象
        tm_params.nb_vertices = self.terrain.vertices.shape[0]  # 设置网格的顶点数量
        tm_params.nb_triangles = self.terrain.triangles.shape[0]  # 设置网格的三角形数量

        tm_params.transform.p.x = -self.terrain.cfg.border_size  # 设置网格的x轴位置（平移）
        tm_params.transform.p.y = -self.terrain.cfg.border_size  # 设置网格的y轴位置（平移）
        tm_params.transform.p.z = 0.0  # 设置网格的z轴位置（平移）
        tm_params.static_friction = self.cfg.terrain.static_friction  # 设置网格的静摩擦系数
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction  # 设置网格的动摩擦系数
        tm_params.restitution = self.cfg.terrain.restitution  # 设置网格的恢复系数（弹性系数）

        # 将三角形网格添加到仿真环境中
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)

        # 将高度样本转换为张量，并发送到设备
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """
        创建模拟环境和机器人actor

        注解:
        1. 根据配置文件中的资产路径加载机器人模型。
        2. 设置资产选项，如驱动模式、固定关节折叠、替换圆柱体等。
        3. 加载资产并获取机器人的自由度和刚体数量。
        4. 获取资产的自由度属性和刚体形状属性。
        5. 保存资产中的刚体名称和自由度名称。
        6. 根据配置文件中的初始状态设置基础初始状态。
        7. 调用_get_env_origins函数初始化环境原点。
        8. 遍历每个环境，创建环境实例并在其中创建机器人actor。
        9. 对每个环境的刚体形状属性和自由度属性进行处理并应用。
        10. 保存脚部和接触惩罚部位的索引。
        11. 保存接触终止部位的索引。

        注释:
        - 该函数负责根据配置文件创建模拟环境，并在其中放置机器人actor。
        """
        # 根据配置文件中定义的资产文件路径格式和根目录，构建完整的资产文件路径
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        # 获取资产文件的根目录路径
        asset_root = os.path.dirname(asset_path)
        # 获取资产文件的名称
        asset_file = os.path.basename(asset_path)

        # 创建资产选项实例，用于配置资产的加载选项
        asset_options = gymapi.AssetOptions()
        # 设置默认的关节驱动模式
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        # 是否合并固定关节，以减少计算成本
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        # 是否将圆柱体替换为胶囊体，通常用于改善碰撞检测
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        # 是否翻转视觉附件，这与模型的可视化表现有关
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        # 是否固定基础链接，这对于某些类型的仿真可能是必要的
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        # 设置资产的密度，影响物理仿真的真实性
        asset_options.density = self.cfg.asset.density
        # 设置资产的角阻尼，影响旋转运动的阻力
        asset_options.angular_damping = self.cfg.asset.angular_damping
        # 设置资产的线阻尼，影响直线运动的阻力
        asset_options.linear_damping = self.cfg.asset.linear_damping
        # 设置资产的最大角速度，限制旋转速度
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        # 设置资产的最大线速度，限制直线移动速度
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        # 设置资产的肌腱模拟参数，影响关节的弹性
        asset_options.armature = self.cfg.asset.armature
        # 设置资产的厚度，影响视觉和碰撞模型的表现
        asset_options.thickness = self.cfg.asset.thickness
        # 是否禁用重力，对于需要在无重力环境中仿真的资产很有用
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # 加载配置的资产文件到仿真环境中，使用前面设置的资产选项
        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        # 获取资产的自由度（DoF）数量，即资产中可动部分（如关节）的数量
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        # 将可执行动作的数量设置为资产的自由度数量
        self.num_actuated_dof = self.num_actions
        # 获取资产的刚体数量，即构成资产的固体部分的数量
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        # 获取资产的关节属性，如关节的运动范围、力矩限制等
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        # 获取资产的刚体形状属性，包括刚体的大小、质量、摩擦系数等
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

        # 获取资产中所有刚体的名称
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        # 获取资产中所有自由度（DoF）的名称
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        # 更新资产的刚体数量，确保与实际匹配
        self.num_bodies = len(body_names)
        # 更新资产的自由度（DoF）数量，确保与实际匹配
        self.num_dofs = len(self.dof_names)
        # 从所有刚体名称中筛选出符合“脚”命名条件的刚体名称，用于后续处理（如平衡控制、步态分析）
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        # 初始化将被应用接触惩罚的刚体名称列表
        penalized_contact_names = []
        # 遍历配置文件中指定的需要应用接触惩罚的刚体标识符，筛选出对应的刚体名称
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        # 初始化触发仿真终止的接触刚体名称列表
        termination_contact_names = []
        # 遍历配置文件中指定的在接触后终止仿真的刚体标识符，筛选出对应的刚体名称
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # 设置基础初始状态
        # 将配置文件中定义的初始位置、旋转、线速度和角速度合并成一个列表
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + \
                               self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        # 将上述初始状态列表转换为PyTorch张量，并指定存储在指定的设备上，不需要计算梯度
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        # 创建一个仿真环境中的变换对象，用于设置资产的初始姿态
        start_pose = gymapi.Transform()
        # 将初始状态中的位置信息（前三个元素）设置为变换对象的位置部分
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # 初始化环境原点
        # 初始化每个仿真环境原点的张量，形状为[num_envs, 3]，并设置为0，表示所有环境的原点都在坐标系的原点
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # 初始化每个环境的地形水平为0，这可能表示地形的高度或类型的一个标识符
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        # 初始化每个环境的地形原点，与env_origins类似，也设置为0
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # 初始化每个环境的地形类型为0，这也可能表示不同类型的地形或地面条件
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        # 调用一个函数（假设是获取环境原点的函数），这里没有给出具体实现，但它可能会根据仿真需求更新env_origins的值
        self._call_train_eval(self._get_env_origins, torch.arange(self.num_envs, device=self.device))
        # 初始化环境边界的下限和上限，这里都设置为(0, 0, 0)，可能用于后续定义环境的大小或范围
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        # 初始化用于存储每个环境中的仿真实体句柄的列表
        self.actor_handles = []
        # 初始化用于存储每个环境中IMU传感器句柄的列表
        self.imu_sensor_handles = []
        # 初始化用于存储仿真环境容器的列表
        self.envs = []

        # 从之前获取的刚体形状属性中，设置默认的摩擦系数为第二个刚体的摩擦系数
        # 注意这里的索引是1，假设是因为Python的索引从0开始，这意味着使用的是第二个刚体的属性
        self.default_friction = rigid_shape_props_asset[1].friction
        # 同样地，设置默认的恢复系数为第二个刚体的恢复系数
        self.default_restitution = rigid_shape_props_asset[1].restitution
        # 调用一个自定义的初始化函数，该函数可能用于初始化一些自定义的缓冲区或其他数据结构
        # 具体实现细节未给出，但这通常是为了准备仿真运行所需的特定数据结构
        self._init_custom_buffers__()
        # 调用一个函数来随机化每个仿真环境中的刚体属性，如质量、摩擦系数等
        # 这里假设`_randomize_rigid_body_props`是一个自定义函数，用于增加仿真的多样性
        # 通过传递所有环境的索引，这个函数可能会为每个环境的刚体设置不同的物理属性
        self._call_train_eval(self._randomize_rigid_body_props, torch.arange(self.num_envs, device=self.device))
        # 调用一个函数来随机化重力设置，这也是为了增加仿真环境的多样性
        # `self._randomize_gravity`可能会改变重力的大小或方向，模拟不同的重力条件
        self._randomize_gravity()

        for i in range(self.num_envs):
            # 在仿真中为每个环境创建一个新的环境实例
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # 克隆初始位置，并根据配置中的初始X和Y范围随机调整位置
            pos = self.env_origins[i].clone()
            # 随机调整X位置
            pos[0:1] += torch_rand_float(-self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            # 随机调整Y位置
            pos[1:2] += torch_rand_float(-self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            # 设置起始位置
            start_pose.p = gymapi.Vec3(*pos)
            # 处理刚体形状属性，并设置给机器人资产
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            # 在环境实例中创建一个机器人或仿生体角色
            anymal_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, "anymal", i,
                                                  self.cfg.asset.self_collisions, 0)
            # 处理并设置关节属性
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            # 获取并处理刚体属性，然后更新机器人或仿生体的刚体属性
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            # 将环境句柄添加到环境列表中，用于后续操作
            self.envs.append(env_handle)
            # 将创建的机器人或仿生体角色句柄添加到列表中
            self.actor_handles.append(anymal_handle)

        # 保存脚部索引
        # 创建一个零张量来存储脚部索引，长度为脚部名称列表的长度
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):  # 遍历脚部名称列表
            # 查找并保存每个脚部对应的刚体句柄索引
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                         self.actor_handles[0], feet_names[i])

        # 保存接触惩罚部位索引
        # 创建一个零张量来存储接触惩罚部位索引
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long,
                                                     device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):  # 遍历接触惩罚部位名称列表
            # 查找并保存每个接触惩罚部位对应的刚体句柄索引
            self.penalised_contact_indices[i] = (self.gym.find_actor_rigid_body_handle
                                                 (self.envs[0], self.actor_handles[0], penalized_contact_names[i]))

        # 保存接触终止部位索引
        # 创建一个零张量来存储接触终止部位索引
        self.termination_contact_indices = torch.zeros(len(termination_contact_names),
                                                       dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):  # 遍历接触终止部位名称列表
            # 查找并保存每个接触终止部位对应的刚体句柄索引
            self.termination_contact_indices[i] = (self.gym.find_actor_rigid_body_handle
                                                   (self.envs[0], self.actor_handles[0], termination_contact_names[i]))

        # 如果配置了录制视频，设置相机
        if self.cfg.env.record_video:
            # 创建一个摄像头属性对象
            self.camera_props = gymapi.CameraProperties()
            # 设置摄像头的宽度为360像素
            self.camera_props.width = 512
            # 设置摄像头的高度为240像素
            self.camera_props.height = 512
            # 为环境中的第一个环境创建摄像头传感器，并保存到self.rendering_camera
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
            # 设置摄像头的位置和朝向。摄像头位于(1.5, 1, 3.0)，朝向(0, 0, 0)点
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(1.5, 1, 3.0),
                                         gymapi.Vec3(0, 0, 0))
            # 如果有评估配置(eval_cfg)，则为评估环境也创建并设置摄像头
            if self.eval_cfg is not None:
                # 为评估环境创建摄像头传感器，评估环境是训练环境之后的第一个环境
                self.rendering_camera_eval = self.gym.create_camera_sensor(self.envs[self.num_train_envs],
                                                                           self.camera_props)
                # 设置评估环境的摄像头的位置和朝向。位置和朝向与训练环境的相同
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(1.5, 1, 3.0),
                                             gymapi.Vec3(0, 0, 0))

        # 初始化视频写入器为None，表示开始时没有正在进行的视频写入
        self.video_writer = None
        # 初始化存储视频帧的列表
        self.video_frames = []  # 存储训练环境的视频帧
        self.video_frames_eval = []  # 存储评估环境的视频帧
        self.complete_video_frames = []  # 存储完成的训练视频帧
        self.complete_video_frames_eval = []  # 存储完成的评估视频帧

    def render(self, mode="rgb_array"):
        # 确保渲染模式为rgb_array，这是获取图像数据的常用模式
        assert mode == "rgb_array"
        # 从root_states中获取根对象的x, y, z位置
        bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
        # 更新摄像头的位置和朝向，以便跟随根对象移动。摄像头设置为根对象后面1.0单位，上方1.0单位的位置
        self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                     gymapi.Vec3(bx, by, bz))
        # 更新图形，进行一步图形渲染计算
        self.gym.step_graphics(self.sim)
        # 渲染所有摄像头传感器的图像
        self.gym.render_all_camera_sensors(self.sim)
        # 获取指定摄像头的图像，图像类型为彩色图(IMAGE_COLOR)
        img = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR)
        # 获取图像的宽度和高度
        w, h = img.shape
        # 重塑图像数组的形状为[w, h//4, 4]，通常用于调整图像数据的格式以便于处理
        # 注意：这里的reshape可能意味着图像数据的某种特定格式调整，需根据实际情况理解
        return img.reshape([w, h // 4, 4])

    def _render_headless(self):
        # 如果当前在记录视频，并且完整视频帧列表存在且为空
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            # 获取根状态的位置坐标
            bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
            # 设置训练环境摄像头的位置和朝向
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                         gymapi.Vec3(bx, by, bz))
            # 获取当前摄像头的图像，并将其存储为视频帧
            self.video_frame = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera,
                                                         gymapi.IMAGE_COLOR)
            # 调整图像的形状以匹配摄像头属性指定的尺寸
            self.video_frame = self.video_frame.reshape((self.camera_props.height, self.camera_props.width, 4))
            # 将视频帧添加到视频帧列表中
            self.video_frames.append(self.video_frame)

        # 如果当前在记录评估环境的视频，并且评估环境的完整视频帧列表存在且为空
        if self.record_eval_now and self.complete_video_frames_eval is not None and len(
                self.complete_video_frames_eval) == 0:
            # 如果评估配置存在
            if self.eval_cfg is not None:
                # 获取评估环境的根状态位置坐标
                bx, by, bz = self.root_states[self.num_train_envs, 0], self.root_states[self.num_train_envs, 1], \
                    self.root_states[self.num_train_envs, 2]
                # 设置评估环境摄像头的位置和朝向
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                             gymapi.Vec3(bx, by, bz))
                # 获取评估环境摄像头的图像，并将其调整形状后存储为视频帧
                self.video_frame_eval = self.gym.get_camera_image(self.sim, self.envs[self.num_train_envs],
                                                                  self.rendering_camera_eval,
                                                                  gymapi.IMAGE_COLOR)
                self.video_frame_eval = self.video_frame_eval.reshape(
                    (self.camera_props.height, self.camera_props.width, 4))
                # 将视频帧添加到评估环境的视频帧列表中
                self.video_frames_eval.append(self.video_frame_eval)

    def start_recording(self):
        # 初始化完成的视频帧列表为None，表示开始新的视频录制
        self.complete_video_frames = None
        # 设置录制标志为True，开始录制视频
        self.record_now = True

    def recording(self):
        self.complete_video_frames = []
        self.record_now = True
        self._render_headless()

    def start_recording_eval(self):
        # 对于评估环境，初始化完成的视频帧列表为None，表示开始新的视频录制
        self.complete_video_frames_eval = None
        # 设置评估环境的录制标志为True，开始录制视频
        self.record_eval_now = True

    def pause_recording(self):
        # 初始化完成的视频帧列表为空列表，表示暂停视频录制并清空当前录制内容
        self.complete_video_frames = []
        # 清空视频帧列表，以便重新开始录制
        self.video_frames = []
        # 设置录制标志为False，停止录制视频
        self.record_now = False

    def pause_recording_eval(self):
        # 对于评估环境，同样初始化完成的视频帧列表为空列表，表示暂停录制并清空当前录制内容
        self.complete_video_frames_eval = []
        # 清空评估环境的视频帧列表，以便重新开始录制
        self.video_frames_eval = []
        # 设置评估环境的录制标志为False，停止录制视频
        self.record_eval_now = False

    def get_complete_frames(self):
        # 如果完成的视频帧列表为None，返回空列表
        if self.complete_video_frames is None:
            return []
        # 否则返回完成的视频帧列表
        return self.complete_video_frames

    def get_frames(self):
        self.complete_video_frames = self.video_frames[:]
        return self.complete_video_frames

    def get_complete_frames_eval(self):
        # 对于评估环境，如果完成的视频帧列表为None，返回空列表
        if self.complete_video_frames_eval is None:
            return []
        # 否则返回评估环境的完成的视频帧列表
        return self.complete_video_frames_eval

    def _get_env_origins(self, env_ids, cfg):
        """
        设置环境起始点。在粗糙地形上，起始点由地形平台定义。否则，创建一个网格。

        参数:
        - env_ids: 环境的ID列表。
        - cfg: 包含环境配置的配置对象。
        """

        # 如果地形类型是高度场(heightfield)或三角形网格(trimesh)，则按地形来设置起始点
        if cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True  # 标记为使用自定义起始点

            # 根据配置，设置机器人的初始位置范围
            max_init_level = cfg.terrain.max_init_terrain_level
            min_init_level = cfg.terrain.min_init_terrain_level
            # 如果未使用课程学习，则设置为固定的最大和最小初始水平
            if not cfg.terrain.curriculum: max_init_level = cfg.terrain.num_rows - 1
            if not cfg.terrain.curriculum: min_init_level = 0

            # 如果设置了机器人中心化
            if cfg.terrain.center_robots:
                # 计算中心化时的行列范围
                min_terrain_level = cfg.terrain.num_rows // 2 - cfg.terrain.center_span
                max_terrain_level = cfg.terrain.num_rows // 2 + cfg.terrain.center_span - 1
                min_terrain_type = cfg.terrain.num_cols // 2 - cfg.terrain.center_span
                max_terrain_type = cfg.terrain.num_cols // 2 + cfg.terrain.center_span - 1

                # 随机选择起始点的行列位置
                self.terrain_levels[env_ids] = torch.randint(min_terrain_level, max_terrain_level + 1, (len(env_ids),),
                                                             device=self.device)
                self.terrain_types[env_ids] = torch.randint(min_terrain_type, max_terrain_type + 1, (len(env_ids),),
                                                            device=self.device)
            else:
                # 如果没有中心化，则随机选择初始水平和类型
                self.terrain_levels[env_ids] = torch.randint(min_init_level, max_init_level + 1, (len(env_ids),),
                                                             device=self.device)
                self.terrain_types[env_ids] = torch.div(torch.arange(len(env_ids), device=self.device),
                                                        (len(env_ids) / cfg.terrain.num_cols),
                                                        rounding_mode='floor').to(torch.long)
            # 更新地形的最大水平为行数
            cfg.terrain.max_terrain_level = cfg.terrain.num_rows
            # 将numpy数组转换为torch张量并设置为地形起始点
            cfg.terrain.terrain_origins = torch.from_numpy(cfg.terrain.env_origins).to(self.device).to(torch.float)
            # 根据计算出的地形水平和类型获取实际的环境起始点
            self.env_origins[env_ids] = cfg.terrain.terrain_origins[
                self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        else:
            self.custom_origins = False  # 未使用自定义起始点，即使用网格布局

            # 计算网格的列数和行数
            num_cols = np.floor(np.sqrt(len(env_ids)))
            num_rows = np.ceil(self.num_envs / num_cols)
            # 创建一个网格
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            xx, yy = xx.to(self.device), yy.to(self.device)
            spacing = cfg.env.env_spacing  # 获取网格间距
            # 设置环境起始点的x和y坐标，z坐标固定为0
            self.env_origins[env_ids, 0] = spacing * xx.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 1] = spacing * yy.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 2] = 0

    def _parse_cfg(self, cfg):
        # 计算模拟步长。它是控制决策间隔和模拟器参数中的时间步长(dt)的乘积。
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        # 从配置中获取观测的缩放系数。
        self.obs_scales = self.cfg.obs_scales
        # 使用vars函数将配置中的奖励缩放系数转换为字典。
        self.reward_scales = vars(self.cfg.reward_scales)
        # 使用vars函数将配置中的课程学习阈值转换为字典。
        self.curriculum_thresholds = vars(self.cfg.curriculum_thresholds)
        # 将命令范围配置转换为字典。
        cfg.command_ranges = vars(cfg.commands)
        # 如果地形类型不是高度场(heightfield)或三角形网格(trimesh)，则关闭课程学习。
        if cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            cfg.terrain.curriculum = False
        # 根据环境中设置的每集最长时间和时间步长计算最大的回合长度。
        self.max_episode_length_s = cfg.env.episode_length_s
        cfg.env.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        # 更新实例变量以存储最大回合长度。
        self.max_episode_length = cfg.env.max_episode_length

        # 根据时间步长调整领域随机性中各项间隔的时间。
        cfg.domain_rand.push_interval = np.ceil(cfg.domain_rand.push_interval_s / self.dt)
        cfg.domain_rand.rand_interval = np.ceil(cfg.domain_rand.rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_interval = np.ceil(cfg.domain_rand.gravity_rand_interval_s / self.dt)
        # 根据时间步长和随机性间隔调整重力随机性持续的时长。
        cfg.domain_rand.gravity_rand_duration = np.ceil(
            cfg.domain_rand.gravity_rand_interval * cfg.domain_rand.gravity_impulse_duration)

    def _draw_debug_vis(self):
        """
        绘制调试视觉效果，特别是地形的高度线

        注解:
        1. 如果配置中没有启用测量高度(measure_heights)，则直接返回，不进行绘制。
        2. 清除观察器中的所有线条。
        3. 刷新刚体状态张量，以确保绘制的是最新的状态。
        4. 创建一个黄色的线框球体几何体，用于表示高度点。
        5. 遍历所有环境，获取每个环境的基础位置和测量高度。
        6. 计算高度点的位置，并将其转换为世界坐标系中的点。
        7. 对于每个高度点，绘制一个球体，表示地面的高度。

        注意:
        - 这个方法假设已经有了一些类属性，如self.terrain、self.gym、self.viewer等。
        - 这个方法用于调试目的，可以帮助理解和可视化地形的高度信息。
        """
        # 如果没有启用测量高度的配置，直接返回
        if not self.terrain.cfg.measure_heights:
            return
        # 清除观察器中的所有线条
        self.gym.clear_lines(self.viewer)
        # 刷新刚体状态张量
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # 创建一个黄色的线框球体几何体
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        # 遍历所有环境
        for i in range(self.num_envs):
            # 获取基础位置
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            # 获取测量高度
            heights = self.measured_heights[i].cpu().numpy()
            # 计算高度点的位置
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                           self.height_points[i]).cpu().numpy()
            # 对于每个高度点，绘制一个球体
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                # 设置球体的位置
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                # 绘制线框球体
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """
        初始化用于测量地形高度的点

        返回值:
        - points: 包含所有环境中所有测量点位置的张量

        注解:
        1. 创建一个y坐标的张量，基于配置中指定的测量点的y坐标。
        2. 创建一个x坐标的张量，基于配置中指定的测量点的x坐标。
        3. 使用torch.meshgrid生成一个x和y坐标的网格。
        4. 计算网格中点的总数，并存储在num_height_points属性中。
        5. 初始化一个三维点的张量，用于存储所有环境中的测量点的位置。
        6. 将网格中的x坐标和y坐标分别赋值给点张量的对应位置。
        7. 返回初始化好的点张量。

        注意:
        - 这个方法假设类中已经有了self.cfg、self.device等属性。
        - 这个方法用于初始化地形高度测量点，这些点之后将用于调试和分析地形。
        """
        # 创建y坐标的张量
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        # 创建x坐标的张量
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        # 生成x和y坐标的网格
        grid_x, grid_y = torch.meshgrid(x, y)

        # 计算网格中点的总数
        self.num_height_points = grid_x.numel()
        # 初始化点张量
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        # 赋值x坐标
        points[:, :, 0] = grid_x.flatten()
        # 赋值y坐标
        points[:, :, 1] = grid_y.flatten()
        # 返回点张量
        return points

    def _get_heights(self, env_ids=None):
        """
        获取指定环境的地形高度

        参数:
        - env_ids: 可选参数，指定需要获取高度的环境ID列表

        返回值:
        - 高度数据的张量，形状为(num_envs, num_height_points)

        注解:
        1. 如果地形类型为平面，则返回零高度。
        2. 如果地形类型为'none'，则抛出异常。
        3. 根据环境ID计算高度点的世界坐标。
        4. 将高度点坐标转换为地形高度图的像素坐标。
        5. 从高度图中获取高度值，并返回调整比例后的高度。
        """

        # 如果地形类型为平面，返回零高度
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)

        # 如果地形类型为'none'，抛出异常
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        # 根据环境ID计算高度点的世界坐标
        if env_ids is None:
            for env_id in env_ids:
                points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                        self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = (quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) +
                      (self.root_states[:, :3]).unsqueeze(1))

        # 将高度点坐标转换为地形高度图的像素坐标
        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)

        # 限制像素坐标的范围，防止越界
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        # 从高度图中获取高度值
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]

        # 获取三个相邻点的最小高度值
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        # 返回调整比例后的高度
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        """
        计算机器人基座在z轴上的线速度惩罚

        返回值:
        - z轴线速度的平方，表示对z轴线速度的惩罚
        """
        # 对机器人基座在z轴上的线速度进行平方，以计算惩罚项
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """
        计算机器人基座在xy平面上的角速度惩罚

        返回值:
        - xy轴角速度的平方和，表示对xy平面上角速度的惩罚
        """
        # 对机器人基座在xy平面上的角速度进行平方，并求和，以计算惩罚项
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """
        计算机器人基座非平面朝向的惩罚

        返回值:
        - 重力投影到xy平面的分量的平方和，表示对非平面朝向的惩罚
        """
        # 对机器人基座相对于重力方向在xy平面上的分量进行平方，并求和，以计算惩罚项
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        """
        计算机器人基座高度偏离目标值的惩罚

        返回值:
        - 基座高度偏离目标值的平方，表示对高度偏离的惩罚
        """
        # 计算基座的平均高度与测量高度之差
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # 计算基座高度偏离目标值的平方，作为惩罚项
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        """
        计算施加的扭矩的惩罚

        返回值:
        - 扭矩的平方和，表示对扭矩大小的惩罚
        """
        # 对施加的扭矩进行平方，并求和，以计算惩罚项
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        计算关节速度的惩罚

        返回值:
        - 关节速度的平方和，表示对关节速度的惩罚
        """
        # 对关节速度进行平方，并求和，以计算惩罚项
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """
        计算关节加速度的惩罚

        返回值:
        - 关节加速度的平方和，表示对关节加速度的惩罚
        """
        # 计算关节速度的变化量除以时间步长，得到加速度，然后进行平方，并求和，以计算惩罚项
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        """
        计算动作变化率的惩罚

        返回值:
        - 动作变化的平方和，表示对动作变化率的惩罚
        """
        # 计算上一动作与当前动作之差的平方，并求和，以计算惩罚项
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        """
        计算选定部位发生碰撞的惩罚

        返回值:
        - 碰撞力超过阈值的部位数量，表示对碰撞的惩罚
        """
        # 计算接触力的范数，判断是否超过阈值，然后对超过阈值的接触力进行计数，以计算惩罚项
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        """
        计算终止时的奖励或惩罚

        返回值:
        - 如果发生终止，则返回1，否则返回0
        """
        # 使用重置缓冲区和超时缓冲区的逻辑非运算结果，来判断是否应该给予终止奖励或惩罚
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        """
        计算关节位置超出限制的惩罚

        返回值:
        - 一个张量，表示每个样本的关节位置超出限制的总惩罚值

        注解:
        1. 计算关节位置与其下限的差值，小于0的部分表示超出下限，对这部分进行惩罚。
        2. 计算关节位置与其上限的差值，大于0的部分表示超出上限，对这部分进行惩罚。
        3. 将上述两部分的惩罚值相加，得到总的惩罚值。
        4. 对每个样本的惩罚值进行求和，得到一个批次的总惩罚值。
        """
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # 计算低于下限的关节位置惩罚
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.) # 计算超过上限的关节位置惩罚
        return torch.sum(out_of_limits, dim=1) # 返回每个样本的关节位置超出限制的总惩罚值

    def _reward_dof_vel_limits(self):
        """
        计算关节速度超出限制的惩罚

        返回值:
        - 一个张量，表示每个样本的关节速度超出限制的总惩罚值

        注解:
        1. 计算关节速度与速度限制的差值，大于0的部分表示超出限制，对这部分进行惩罚。
        2. 限制最大的惩罚值为每个关节1弧度/秒，避免过大的惩罚。
        3. 对每个样本的惩罚值进行求和，得到一个批次的总惩罚值。
        """
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        """
        计算扭矩超出限制的惩罚

        返回值:
        - 一个张量，表示每个样本的扭矩超出限制的总惩罚值

        注解:
        1. 计算扭矩与扭矩限制的差值，大于0的部分表示超出限制，对这部分进行惩罚。
        2. 对每个样本的惩罚值进行求和，得到一个批次的总惩罚值。
        """
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        """
        计算线速度跟踪的奖励

        返回值:
        - 一个张量，表示每个样本的线速度跟踪奖励

        注解:
        1. 计算指令的线速度与基础线速度在xy轴上的误差。
        2. 对误差进行平方求和，得到每个样本的线速度误差。
        3. 根据配置中的tracking_sigma参数，计算指数衰减的奖励值。
        """
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1) # 计算线速度误差
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma) # 返回线速度跟踪奖励

    def _reward_tracking_ang_vel(self):
        """
        计算角速度跟踪的奖励

        返回值:
        - 一个张量，表示每个样本的角速度跟踪奖励

        注解:
        1. 计算指令的角速度与基础角速度在yaw轴上的误差。
        2. 对误差进行平方，得到每个样本的角速度误差。
        3. 根据配置中的tracking_sigma参数，计算指数衰减的奖励值。
        """
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2]) # 计算角速度误差
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma) # 返回角速度跟踪奖励

    def _reward_feet_air_time(self):
        """
        计算脚部空中停留时间的奖励

        返回值:
        - 一个张量，表示每个样本的脚部空中停留时间奖励

        注解:
        1. 由于PhysX在网格上的接触报告不可靠，需要过滤接触力来确定是否有接触。
        2. 如果接触力大于1，则认为脚部有接触。
        3. 使用逻辑或操作来更新过滤后的接触状态，考虑当前和上一次的接触情况。
        4. 更新上一次的接触状态为当前接触状态。
        5. 如果脚部空中停留时间大于0且当前有接触，则认为是第一次接触。
        6. 更新脚部空中停留时间，增加一个时间步长。
        7. 计算奖励，只在第一次接触地面时给予奖励，并且奖励值与空中停留时间相关。
        8. 如果指令的线速度大于0.1，则给予奖励；否则，不给予奖励。
        9. 如果当前没有接触，则将脚部空中停留时间重置为0。
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1. # 判断脚部是否有接触
        contact_filt = torch.logical_or(contact, self.last_contacts) # 更新过滤后的接触状态
        self.last_contacts = contact # 更新上一次的接触状态
        first_contact = (self.feet_air_time > 0.) * contact_filt # 判断是否为第一次接触
        self.feet_air_time += self.dt # 更新脚部空中停留时间
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # 计算第一次接触地面的奖励
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 # 如果指令的线速度大于0.1，则给予奖励
        self.feet_air_time *= ~contact_filt # 如果当前没有接触，则重置空中停留时间
        return rew_airTime # 返回脚部空中停留时间奖励

    def _reward_stumble(self):
        """
        计算脚部撞击垂直表面的惩罚

        返回值:
        - 一个布尔型张量，表示每个样本是否有脚部撞击垂直表面的情况

        注解:
        1. 计算脚部接触力在水平方向（x和y轴）的范数。
        2. 如果水平方向的接触力大于垂直方向接触力的5倍，则认为脚部撞击了垂直表面。
        3. 对每个样本检查是否有脚部撞击垂直表面的情况，并返回一个布尔型张量。
        """
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
                5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        """
        计算在零指令下的运动惩罚

        返回值:
        - 一个张量，表示每个样本在零指令下的运动惩罚

        注解:
        1. 计算关节位置与默认位置的差的绝对值之和，表示关节的运动量。
        2. 如果指令的线速度小于0.1，则认为是零指令。
        3. 在零指令的情况下，对关节的运动量进行惩罚。
        """
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        """
        计算高接触力的惩罚

        返回值:
        - 一个张量，表示每个样本的高接触力惩罚

        注解:
        1. 计算脚部接触力的范数。
        2. 如果接触力的范数超过了配置中设定的最大接触力，对超出部分进行惩罚。
        3. 对每个样本的超出部分进行求和，得到高接触力的总惩罚。
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
