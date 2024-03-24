import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from legged_gym.rsl_rl.env import VecEnv
from legged_gym.algorithms.sac import ReplayBuffer


class OffPolicyRunner:
    def __init__(self, env: VecEnv, log_dir=None, device='cpu'):
        from legged_gym.algorithms.sac import SAC

        self.device = device  # 设置计算设备
        self.env = env  # 设置环境

        # 根据环境是否提供特权观测来确定批评者的观测空间大小
        if self.env.num_privileged_obs is not None:
            self.num_critic_obs = self.env.num_privileged_obs  # 如果提供特权观测，则使用特权观测的数量
        else:
            self.num_critic_obs = self.env.num_obs  # 如果不提供特权观测，则使用普通观测的数量

        # 创建经验回放缓冲区
        replay_buffer = ReplayBuffer(max_size=int(5e6), obs_shape=self.env.num_obs,
                                     num_envs=self.env.num_envs,
                                     privileged_obs_shape=self.env.num_privileged_obs,
                                     actions_shape=self.env.num_actions, device=self.device)

        self.alg = SAC(state_dim=self.env.num_obs, replay_buffer=replay_buffer, device=self.device)

        # 设置日志记录器
        self.log_dir = log_dir  # 日志目录
        self.writer = None  # 初始化日志记录器为None，可能后续会被赋予实际的日志记录器对象
        self.tot_timesteps = 0  # 总时间步数初始化为0
        self.tot_time = 0  # 总时间初始化为0
        self.current_learning_iteration = 0  # 当前学习迭代初始化为0

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            # 如果log_dir已设置且writer尚未初始化，则创建一个SummaryWriter实例
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            # 如果初始化时需要随机的episode长度，则对环境的episode_length_buf进行随机赋值
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        _, _ = self.env.reset()

        # 获取环境的观测值
        obs = self.env.get_observations()
        # 获取环境的特权观测值，如果有的话
        privileged_obs = self.env.get_privileged_observations()
        # 如果有特权观测值则使用它，否则使用普通观测值
        critic_obs = privileged_obs if privileged_obs is not None else obs
        # 将观测值转移到指定的设备上（例如GPU）
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        # 初始化episode信息列表和奖励、长度缓冲
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        # 初始化当前奖励和episode长度的张量
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 计算总的学习迭代次数
        tot_iter = self.current_learning_iteration + num_learning_iterations  # 当前迭代加上新的迭代次数
        # for it in range(self.current_learning_iteration, tot_iter):  # 遍历每一次学习迭代
        for it in range(100000):  # 遍历每一次学习迭代
            start = time.time()  # 记录迭代开始的时间

            with (torch.inference_mode()):  # 开启推理模式，关闭梯度计算以提升性能
                # 根据观测值获取动作
                actions = self.alg.select_action(obs, deterministic=False)
                # 环境根据动作返回新的状态等信息
                obs_next, privileged_obs, rewards, dones, infos = self.env.step(actions)

                # 如果有特权观测值就更新，否则用普通观测值
                critic_obs_next = privileged_obs if privileged_obs is not None else obs_next
                # 将新的观测值、奖励和完成标志转移到指定设备
                obs, obs_next, critic_obs_next, rewards, dones = obs.to(self.device), obs_next.to(self.device), \
                    critic_obs_next.to(self.device), rewards.to(
                    self.device), dones.to(self.device)

                actions = actions.to(self.device)

                # 处理环境步骤的结果
                self.alg.process_env_step(actions=actions, obs=obs,
                                          obs_next=obs_next, rewards=rewards,
                                          dones=dones)

                obs = obs_next

                # 如果设置了日志目录，则进行日志记录
                if self.log_dir is not None:
                    # 记录episode信息
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])  # 记录每个episode的信息
                    # 更新当前奖励和episode长度
                    cur_reward_sum += rewards  # 累加奖励
                    cur_episode_length += 1  # 累加长度
                    # 对于完成的环境，记录奖励和长度，并重置计数器
                    new_ids = (dones > 0).nonzero(as_tuple=False)  # 找出完成的环境
                    r = rewards.max()
                    print(r)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())  # 记录奖励
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())  # 记录长度
                    cur_reward_sum[new_ids] = 0  # 重置累加的奖励
                    cur_episode_length[new_ids] = 0  # 重置累加的长度

            stop = time.time()  # 记录迭代结束的时间
            collection_time = stop - start  # 计算收集数据的时间

            # Learning step
            start = stop  # 更新学习步骤开始的时间

            for i in range(20):
                q_loss, a_loss = self.alg.update()  # 更新算法，获得平均值损失和替代损失
            stop = time.time()  # 记录学习步骤结束的时间
            # print("=======")
            # print("q loss: ", q_loss)
            # print("a loss: ", a_loss)
            # print("=======")
