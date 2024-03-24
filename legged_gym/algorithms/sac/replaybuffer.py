import torch
import numpy as np


class ReplayBuffer():

    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.observations_next = None

        def clear(self):
            self.__init__()

    def __init__(self, max_size, num_envs, obs_shape, privileged_obs_shape, actions_shape, device='cpu'):
        # 设备配置，用于确定张量存储在CPU还是GPU上
        self.device = device
        self.max_size = max_size

        # 观测空间的形状
        self.obs_shape = obs_shape
        # 特权观测空间的形状，用于增强学习中的特权信息
        self.privileged_obs_shape = privileged_obs_shape
        # 动作空间的形状
        self.actions_shape = actions_shape

        # 初始化观测张量，存储每个环境的观测值
        self.observations = torch.zeros([self.max_size, obs_shape], device=self.device)
        self.observations_next = torch.zeros([self.max_size, obs_shape], device=self.device)
        # 如果特权观测空间有定义，则初始化特权观测张量，否则设为None
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(self.max_size, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        # 初始化奖励张量，存储每个环境的奖励值
        self.rewards = torch.zeros([self.max_size, 1], device=self.device)
        # 初始化动作张量，存储每个环境的动作值
        self.actions = torch.zeros([self.max_size, actions_shape], device=self.device)
        # 初始化完成标志张量，存储每个环境是否完成（done状态）
        self.dones = torch.zeros([self.max_size, 1], device=self.device).byte()
        # 记录环境的数量
        self.num_envs = num_envs
        # 初始化步骤计数器
        self.step = 0

    def add(self, s, s_next, a, r, done):
        # 检查是否超出回放缓冲区的容量
        if self.step >= self.max_size:
            raise AssertionError("Rollout buffer overflow")
        # 复制观测数据到观测张量
        self.observations[self.step] = s
        self.observations_next[self.step]= s_next
        # 复制动作数据到动作张量
        self.actions[self.step] = a
        # 复制奖励数据到奖励张量，并调整形状
        self.rewards[self.step] = r
        # 复制完成标志到完成标志张量，并调整形状
        self.dones[self.step] = done
        # 步骤计数器自增
        self.step = (self.step + 1) % self.max_size

    def mini_batch_generator(self, batch_size):
        # 随机排列索引，不需要梯度
        indices = torch.randperm(batch_size, requires_grad=False, device=self.device)

        batch_idx = indices  # 获取当前小批次的索引

        # 通过索引获取当前小批次的数据
        obs_batch = self.observations[batch_idx]
        actions_batch = self.actions[batch_idx]
        rewards_batch = self.rewards[batch_idx]
        obs_next_batch = self.observations_next[batch_idx]
        done_batch = self.dones[batch_idx]

        return obs_batch, actions_batch, obs_next_batch, rewards_batch, done_batch
