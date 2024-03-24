import numpy as np
import torch
import torch.nn.functional as F
import copy
import os

from legged_gym.algorithms.sac.actor_critic import Actor, Double_Q_Critic


class SAC:
    def __init__(self,
                 replay_buffer,
                 action_dim=12,
                 state_dim=12,
                 gamma=0.998,
                 l_r=3e-4,
                 adaptive_alpha=True,
                 device='cpu'):

        self.tau = 0.005  # 初始化目标网络的软更新系数
        self.batch_size = 512
        self.l_r = l_r
        self.alpha = 0.12
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.adaptive_alpha = adaptive_alpha
        self.device = device
        self.gamma = gamma

        self.replay_buffer = replay_buffer
        self.transition = self.replay_buffer.Transition()

        # 创建策略网络(actor)及其优化器
        self.actor = Actor(self.state_dim, self.action_dim, (128, 64, 32)).to(self.device).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.l_r)

        # 创建双Q网络(q_critic)及其优化器，并复制一个目标网络(q_critic_target)
        self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, (128, 64, 32)).to(self.device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.l_r)
        self.q_critic_target = copy.deepcopy(self.q_critic)

        # 冻结目标网络的参数
        for p in self.q_critic_target.parameters():
            p.requires_grad = False  # 不需要计算梯度

        # 如果启用自适应alpha，则初始化目标熵和alpha的对数表示以及其优化器
        if self.adaptive_alpha:
            self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True,
                                               device=self.device)  # 目标熵
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True,
                                          device=self.device)  # alpha的对数表示
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.l_r)  # alpha的优化器

    def select_action(self, state, deterministic):
        with torch.no_grad():  # 禁用梯度计算
            a, _ = self.actor(state, deterministic, with_logprob=False)  # 使用策略网络选择动作
        return a  # 将动作转换为NumPy数组并返回

    def process_env_step(self, actions, obs, obs_next, rewards, dones):
        """
        处理环境返回的步骤信息，并记录转换

        参数:
        - rewards: 从环境获得的奖励
        - dones: 表示每个环境是否完成的布尔值数组
        - infos: 包含额外信息的字典，可能包含时间限制信息

        主要步骤:
        1. 克隆奖励并记录到转换中。
        2. 记录完成标志。
        3. 如果infos中包含时间限制信息，则进行引导计算。
        4. 将记录的转换添加到存储器中。
        5. 清除当前转换的信息。
        6. 重置actor_critic网络的状态（如果有环境已完成）。
        """
        # 克隆奖励并记录到转换中
        for i in range(len(actions)):
            r = rewards[i]
            r = r * 10
            if i == 0:
                r = -1
            s = obs[i]
            s_next = obs_next[i]
            a = actions[i]
            done = dones[i]

            self.replay_buffer.add(s, s_next, a, r, done)  # 将当前转换（包含观测值、动作、奖励等）添加到经验回放存储器中

    def update(self):
        # 初始化平均价值损失和平均替代损失
        mean_value_loss = 0
        mean_surrogate_loss = 0

        # 从经验回放缓冲区中采样一批数据
        obs_batch, actions_batch, obs_next_batch, rewards_batch, done_batch = (self.replay_buffer.
                                                                               mini_batch_generator(self.batch_size))

        # ----------------------------- 更新Q网络 ------------------------------ #
        with torch.no_grad():
            # 计算下一个状态的动作和对数概率
            a_next, log_pi_a_next = self.actor(obs_next_batch, deterministic=False, with_logprob=True)
            # 计算目标Q网络对下一个状态和动作的两个Q值估计
            target_Q1, target_Q2 = self.q_critic_target(obs_next_batch, actions_batch)
            target_Q = torch.min(target_Q1, target_Q2)
            # 计算目标Q值，考虑折扣因子和是否结束
            target_Q = rewards_batch + (1 - done_batch) * self.gamma * (target_Q - self.alpha * log_pi_a_next)

        # 获取当前Q值估计
        current_Q1, current_Q2 = self.q_critic(obs_batch, actions_batch)

        # 计算Q值的均方误差损失
        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # 梯度下降更新Q网络
        self.q_critic_optimizer.zero_grad()  # 清除之前的梯度
        q_loss.backward()  # 反向传播计算梯度
        self.q_critic_optimizer.step()  # 执行一步梯度下降

        # ----------------------------- 更新策略网络(actor) ------------------------------ #
        # 冻结Q网络参数
        for params in self.q_critic.parameters(): params.requires_grad = False

        # 计算策略网络的损失
        a, log_pi_a = self.actor(obs_batch, deterministic=False, with_logprob=True)
        current_Q1, current_Q2 = self.q_critic(obs_batch, a)
        Q = torch.min(current_Q1, current_Q2)
        a_loss = (self.alpha * log_pi_a - Q).mean()
        # 梯度下降更新策略网络
        self.actor_optimizer.zero_grad()  # 清除之前的梯度
        a_loss.backward()  # 反向传播计算梯度
        self.actor_optimizer.step()  # 执行一步梯度下降

        # 解冻Q网络参数
        for params in self.q_critic.parameters(): params.requires_grad = True

        # ----------------------------- 更新alpha ------------------------------ #
        if self.adaptive_alpha:
            # 更新alpha以自适应调整策略的熵
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()  # 清除之前的梯度
            alpha_loss.backward()  # 反向传播计算梯度
            self.alpha_optim.step()  # 执行一步梯度下降
            self.alpha = self.log_alpha.exp()  # 更新alpha值

        # ----------------------------- 更新目标Q网络 ------------------------------ #
        # 使用Polyak平均软更新目标Q网络参数
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q_loss, a_loss

    def save(self, EnvName, timestep):
        # 保存策略网络(actor)的状态字典
        directory = "./model/{}/{}".format(EnvName, timestep)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), "./model/{}/{}/actor.pth".format(EnvName, timestep))
        # 保存Q网络(q_critic)的状态字典
        torch.save(self.q_critic.state_dict(), "./model/{}/{}/q_critic.pth".format(EnvName, timestep))

    def load(self, EnvName, timestep):
        # 从文件加载策略网络(actor)的状态字典并应用到模型上
        self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep)))
        # 从文件加载Q网络(q_critic)的状态字典并应用到模型上
        self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep)))
