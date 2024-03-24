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
        self.observations = torch.zeros(self.max_size, num_envs, *obs_shape, device=self.device)
        # 如果特权观测空间有定义，则初始化特权观测张量，否则设为None
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(self.max_size, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        # 初始化奖励张量，存储每个环境的奖励值
        self.rewards = torch.zeros(self.max_size, num_envs, 1, device=self.device)
        # 初始化动作张量，存储每个环境的动作值
        self.actions = torch.zeros(self.max_size, num_envs, *actions_shape, device=self.device)
        # 初始化完成标志张量，存储每个环境是否完成（done状态）
        self.dones = torch.zeros(self.max_size, num_envs, 1, device=self.device).byte()
        # 记录环境的数量
        self.num_envs = num_envs
        # 初始化步骤计数器
        self.step = 0

    def add_transitions(self, transition: Transition):
        # 检查是否超出回放缓冲区的容量
        if self.step >= self.max_size:
            raise AssertionError("Rollout buffer overflow")
        # 复制观测数据到观测张量
        self.observations[self.step].copy_(transition.observations)
        # 如果有特权观测数据，复制到特权观测张量
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        # 复制动作数据到动作张量
        self.actions[self.step].copy_(transition.actions)
        # 复制奖励数据到奖励张量，并调整形状
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        # 复制完成标志到完成标志张量，并调整形状
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        # 步骤计数器自增
        self.step = (self.step + 1) % self.max_size

    def get_statistics(self):
        done = self.dones  # 获取表示轨迹是否结束的标志
        done[-1] = 1  # 确保最后一个轨迹被标记为结束
        # 重新排列'done'标志并将其展平为一维数组，以便于处理
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        # 获取所有结束轨迹的索引位置
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        # 计算每个轨迹的长度
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        # 返回轨迹长度的平均值和奖励的平均值
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env  # 计算总批次大小
        mini_batch_size = batch_size // num_mini_batches  # 计算每个小批次的大小
        # 随机排列索引，不需要梯度
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        observations = self.observations.flatten(0, 1)  # 展平观测数据
        # 展平特权观测数据，如果存在的话
        critic_observations = self.privileged_observations.flatten(0, 1) \
            if self.privileged_observations is not None else observations

        actions = self.actions.flatten(0, 1)  # 展平动作数据

        for epoch in range(num_epochs):  # 对于每个epoch
            for i in range(num_mini_batches):  # 对于每个小批次
                start = i * mini_batch_size  # 小批次的起始索引
                end = (i + 1) * mini_batch_size  # 小批次的结束索引
                batch_idx = indices[start:end]  # 获取当前小批次的索引

                # 通过索引获取当前小批次的数据
                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]

                # 使用yield返回当前小批次的数据
                yield obs_batch, critic_observations_batch, actions_batch, (None, None), None