import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def build_net(layer_shape, hidden_activation, output_activation):
    layers = []  # 初始化层列表
    # 遍历layer_shape中的每个元素，除了最后一个
    for j in range(len(layer_shape)-1):
        # 选择激活函数，隐藏层使用hidden_activation，输出层使用output_activation
        act = hidden_activation if j < len(layer_shape)-2 else output_activation
        # 向层列表中添加线性层和激活函数层
        layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
    # 使用nn.Sequential创建一个连续的模型
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
        super(Actor, self).__init__()  # 调用父类nn.Module的构造函数
        layers = [state_dim] + list(hid_shape)  # 构建隐藏层的形状列表

        self.a_net = build_net(layers, hidden_activation, output_activation)  # 创建策略网络的主体部分
        self.mu_layer = nn.Linear(layers[-1], action_dim)  # 创建输出动作均值的线性层
        self.log_std_layer = nn.Parameter(torch.zeros(1, action_dim))
        self.dropout = nn.Dropout(p=0.0)

        self.LOG_STD_MAX = 2  # 对数标准差的最大值
        self.LOG_STD_MIN = -20  # 对数标准差的最小值

    def forward(self, state, deterministic, with_logprob):
        net_out = self.a_net(state)  # 通过策略网络计算中间层输出
        net_out = self.dropout(net_out)
        mu = self.mu_layer(net_out)  # 计算动作的均值
        log_std = self.log_std_layer.expand_as(mu)# 计算动作对数标准差
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  # 将对数标准差限制在预定范围内
        std = torch.exp(log_std)  # 计算标准差
        dist = Normal(mu, std)  # 创建一个正态分布对象

        if deterministic:
            u = mu  # 如果是确定性策略，则输出均值作为动作
        else:
            u = dist.rsample()  # 否则从分布中随机采样一个动作

        a = torch.tanh(u)  # 将动作通过tanh函数转换，以确保其值在[-1, 1]内

        if with_logprob:
            # 计算动作概率的对数，使用数学技巧来提高数值稳定性
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True)  # 计算采样动作的对数概率
            logp_pi_a -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)  # 应用数学技巧调整对数概率
        else:
            logp_pi_a = None  # 如果不需要动作概率的对数，则设置为None

        a = torch.clamp(a, -100, 100)

        return a, logp_pi_a  # 返回动作和动作概率的对数（如果需要）


class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Critic, self).__init__()  # 调用父类nn.Module的构造函数
        layers = [state_dim + action_dim] + list(hid_shape) + [1]  # 构建网络层的形状
        self.dropout = nn.Dropout(p=0.0)

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)  # 创建第一个Q网络
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)  # 创建第二个Q网络

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)  # 在维度1上拼接状态和动作张量
        q1 = self.Q_1(sa)  # 通过第一个Q网络得到输出值q1
        q1 = self.dropout(q1)
        q2 = self.Q_2(sa)  # 通过第二个Q网络得到输出值q2
        q2 = self.dropout(q2)
        return q1, q2  # 返回两个Q网络的输出值


