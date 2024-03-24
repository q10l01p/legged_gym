import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def build_net(layer_shape, hidden_activation, output_activation):
    """
    构建一个神经网络模型

    参数:
    - layer_shape: 一个整数列表，定义了每层神经网络的节点数。
    - hidden_activation: 隐藏层使用的激活函数。
    - output_activation: 输出层使用的激活函数。

    返回值:
    - 一个nn.Sequential模型，包含了定义的线性层和激活函数。

    主要步骤:
    1. 初始化一个空列表来存放网络的层。
    2. 遍历layer_shape列表，除了最后一个元素，每个元素都会创建一个线性层。
    3. 对于每个线性层，根据它是否是最后一个线性层来决定使用隐藏层激活函数还是输出层激活函数。
    4. 将线性层和对应的激活函数层添加到层列表中。
    5. 使用nn.Sequential将所有层合并为一个连续的模型。

    注释:
    - 这个函数使用了一个for循环来构建一个多层的全连接神经网络。
    - nn.Linear(layer_shape[j], layer_shape[j+1]) 创建一个线性层，其中layer_shape[j]是输入特征的数量，
      layer_shape[j+1]是输出特征的数量。
    - act() 创建了一个激活函数层，这是通过调用hidden_activation或output_activation得到的。
    - nn.Sequential(*layers) 将所有层合并为一个模型，'*layers'是Python的解包操作符，它将列表中的元素作为独立的参数传递。
    """
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
    """
    定义一个Actor类，该类是nn.Module的子类，用于创建策略网络（即Actor网络）。

    属性:
    - a_net: 一个Sequential模型，用于从状态预测动作。
    - mu_layer: 线性层，用于输出动作的均值。
    - log_std_layer: 线性层，用于输出动作的对数标准差。
    - LOG_STD_MAX: 对数标准差的最大值，用于限制输出。
    - LOG_STD_MIN: 对数标准差的最小值，用于限制输出。

    参数:
    - state_dim: 状态空间的维度。
    - action_dim: 动作空间的维度。
    - hid_shape: 一个整数列表，定义隐藏层的形状。
    - hidden_activation: 隐藏层使用的激活函数，默认为nn.ReLU。
    - output_activation: 输出层使用的激活函数，默认为nn.ReLU。

    主要步骤:
    1. 调用父类nn.Module的构造函数。
    2. 构建隐藏层的形状列表，包括状态维度和隐藏层形状。
    3. 使用build_net函数创建策略网络的主体部分。
    4. 创建输出动作均值的线性层。
    5. 创建输出动作对数标准差的线性层。
    6. 设置对数标准差的最大和最小值，用于后续输出的限制。

    注释:
    - 这个类的实例化对象可以用于确定性策略或随机策略的强化学习算法中。
    - self.mu_layer和self.log_std_layer分别输出动作的均值和对数标准差，这对于实现随机策略尤其重要。
    - LOG_STD_MAX和LOG_STD_MIN是对数标准差的裁剪范围，避免过大或过小的值导致性能不稳定。
    """
    def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
        super(Actor, self).__init__()  # 调用父类nn.Module的构造函数
        layers = [state_dim] + list(hid_shape)  # 构建隐藏层的形状列表

        self.a_net = build_net(layers, hidden_activation, output_activation)  # 创建策略网络的主体部分
        self.mu_layer = nn.Linear(layers[-1], action_dim)  # 创建输出动作均值的线性层
        self.log_std_layer = nn.Linear(layers[-1], action_dim)  # 创建输出动作对数标准差的线性层

        self.LOG_STD_MAX = 2  # 对数标准差的最大值
        self.LOG_STD_MIN = -20  # 对数标准差的最小值

    def forward(self, state, deterministic, with_logprob):
        """
        定义前向传播过程，根据当前状态输出动作和动作概率的对数。

        参数:
        - state: 输入的状态张量。
        - deterministic: 布尔值，决定是否使用确定性策略。
        - with_logprob: 布尔值，决定是否计算动作概率的对数。

        返回值:
        - a: 输出的动作张量。
        - logp_pi_a: 动作概率的对数，如果with_logprob为False，则返回None。

        主要步骤:
        1. 通过a_net网络计算中间层输出。
        2. 计算动作的均值。
        3. 计算动作对数标准差，并将其限制在预定范围内。
        4. 计算标准差。
        5. 创建一个正态分布对象。
        6. 根据deterministic参数决定输出确定性动作还是随机采样动作。
        7. 对动作进行tanh转换来强制其在[-1, 1]的范围内。
        8. 如果需要，计算动作概率的对数。

        注释:
        - 使用clamp函数限制对数标准差的值可能会影响学习过程，因为它限制了梯度的自由流动。
        - 通过tanh函数转换动作来确保动作值在合理的范围内，这是一种常见的技术。
        - 计算动作概率的对数时使用的数学技巧是为了避免数值问题并保持计算的稳定性。
        """
        net_out = self.a_net(state)  # 通过策略网络计算中间层输出
        mu = self.mu_layer(net_out)  # 计算动作的均值
        log_std = self.log_std_layer(net_out)  # 计算动作对数标准差
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

        return a, logp_pi_a  # 返回动作和动作概率的对数（如果需要）


class Double_Q_Critic(nn.Module):
    """
    定义一个Double_Q_Critic类，该类是nn.Module的子类，用于创建两个独立的Q网络。

    属性:
    - Q_1: 第一个Q网络，用于估计状态-动作对的值。
    - Q_2: 第二个Q网络，用于估计状态-动作对的值。

    参数:
    - state_dim: 状态空间的维度。
    - action_dim: 动作空间的维度。
    - hid_shape: 一个整数列表，定义隐藏层的形状。

    主要步骤:
    1. 调用父类nn.Module的构造函数。
    2. 构建网络层的形状，包括状态和动作的维度以及隐藏层形状。
    3. 使用build_net函数创建第一个Q网络。
    4. 使用build_net函数创建第二个Q网络。

    注释:
    - Double Q-Learning的核心思想是减少Q-Learning算法中的过度估计(overestimation)偏差。
    - 使用两个独立的Q网络可以在估计状态-动作值时提供不同的视角，从而更稳健地进行学习。
    - nn.Identity激活函数表示输出层不应用任何激活函数，直接输出结果。
    - 这个类通常用于深度强化学习算法，如Twin Delayed DDPG (TD3) 或 Soft Actor-Critic (SAC)。
    """
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Critic, self).__init__()  # 调用父类nn.Module的构造函数
        layers = [state_dim + action_dim] + list(hid_shape) + [1]  # 构建网络层的形状

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)  # 创建第一个Q网络
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)  # 创建第二个Q网络

    def forward(self, state, action):
        """
        定义前向传播过程，根据当前状态和动作输出两个Q值。

        参数:
        - state: 输入的状态张量。
        - action: 输入的动作张量。

        返回值:
        - q1: 第一个Q网络的输出值。
        - q2: 第二个Q网络的输出值。

        主要步骤:
        1. 将状态和动作张量在第二维度上拼接。
        2. 将拼接后的张量输入到第一个Q网络并得到输出值q1。
        3. 将拼接后的张量输入到第二个Q网络并得到输出值q2。

        注释:
        - torch.cat 是一个PyTorch函数，用于将多个张量在指定维度上拼接。
        - 在这里，我们在维度1上拼接状态和动作，因为维度0通常是批次大小。
        - 这个方法使得可以同时得到两个独立Q网络对同一状态-动作对的估计值，这对于双Q学习算法来说是必需的。
        """
        sa = torch.cat([state, action], 1)  # 在维度1上拼接状态和动作张量
        q1 = self.Q_1(sa)  # 通过第一个Q网络得到输出值q1
        q2 = self.Q_2(sa)  # 通过第二个Q网络得到输出值q2
        return q1, q2  # 返回两个Q网络的输出值


def Reward_adapter(r, EnvIdex):
    """
    对原始奖励进行调整以适应不同的环境，这是奖励工程的一种形式，用于改善训练过程。

    参数:
    - r: 原始奖励值。
    - EnvIdex: 环境索引，用于标识不同的环境。

    返回值:
    - r: 调整后的奖励值。

    主要步骤:
    1. 根据环境索引对奖励进行不同的调整。
    2. 对于Pendulum-v0环境，将奖励重新缩放。
    3. 对于LunarLander环境，将较大的负奖励值缩小。
    4. 对于BipedalWalker环境，将较大的负奖励值缩小。

    注释:
    - 奖励工程是强化学习中常用的技术，可以帮助算法更快地学习。
    - 通过调整奖励，可以改变算法的学习焦点，使其更关注于重要的事件。
    - 不同的环境可能需要不同的奖励调整策略，因为它们的目标和挑战各不相同。
    - 这里的EnvIdex应该是一个预先定义好的环境索引，每个索引对应一种环境配置。
    """
    # For Pendulum-v0
    if EnvIdex == 0:
        r = (r + 8) / 8  # 将奖励值重新缩放，使其成为一个更加适合学习的范围

    # For LunarLander
    elif EnvIdex == 1:
        if r <= -100:
            r = -10  # 对于极端的负奖励值，进行缩小处理，以避免过大的负反馈

    # For BipedalWalker
    elif EnvIdex == 4 or EnvIdex == 5:
        if r == -100:
            r = -1  # 同样地，对于极端的负奖励值，进行缩小处理

    return r  # 返回调整后的奖励值


def Action_adapter(a, max_action):
    """
    将动作值从[-1, 1]的范围线性缩放到[-max_action, max_action]的范围。

    参数:
    - a: 原始动作值，假定其范围为[-1, 1]。
    - max_action: 动作的最大可能值。

    返回值:
    - 缩放后的动作值，其范围为[-max_action, max_action]。

    主要步骤:
    1. 将输入动作a与max_action相乘，以线性缩放动作的范围。

    注释:
    - 这个函数用于将标准化的动作值适配到实际环境中的动作范围。
    - 在实际的物理环境或仿真环境中，动作的范围通常不是标准化的[-1, 1]。
    - 这种线性变换是可逆的，即可以通过除以max_action恢复原始的动作值范围。
    """
    return a * max_action  # 线性缩放动作值到[-max_action, max_action]


def Action_adapter_reverse(act, max_action):
    """
    将动作值从[-max_action, max_action]的范围线性缩放回[-1, 1]的范围。

    参数:
    - act: 缩放后的动作值，假定其范围为[-max_action, max_action]。
    - max_action: 动作的最大可能值。

    返回值:
    - 缩放回[-1, 1]范围内的原始动作值。

    主要步骤:
    1. 将输入动作act除以max_action，以线性缩放动作的范围回[-1, 1]。

    注释:
    - 这个函数用于将实际环境中的动作值适配回标准化的动作范围。
    - 这是Action_adapter函数的逆操作。
    - 在强化学习中，这种适配和逆适配过程常用于在环境与学习算法之间转换动作值。
    """
    return act / max_action  # 线性缩放动作值回[-1, 1]


def evaluate_policy(env, agent, turns=3):
    """
    评估给定代理模型在特定环境下的策略表现。

    参数:
    - env: 用于评估代理的环境对象。
    - agent: 要评估的代理模型。
    - turns: 评估轮次，默认为3轮。

    返回值:
    - 平均得分: 在指定轮次的平均得分。

    主要步骤:
    1. 初始化总得分为0。
    2. 对于每一轮评估：
       a. 重置环境获取初始状态。
       b. 在环境未终止的情况下循环：
          i. 使用代理选择动作。
          ii. 执行动作并获取下一状态和奖励。
          iii. 更新总得分。
          iv. 如果达到终止条件，则退出循环。
    3. 计算所有轮次的平均得分。

    注释:
    - 在评估时，代理选择确定性的动作（deterministic=True），即不加入探索噪声。
    - 环境的step方法返回五个值：下一状态(s_next)、奖励(r)、done标志(dw)、是否达到时间限制(tr)和额外信息(info)。
    - done标志为真表示环境已经终止，可以是因为代理完成了任务，也可以是因为出错或达到了时间限制。
    - 评估的目的是为了不加入随机性地测试代理在环境中的表现。
    - 返回的平均得分是整数，这意味着得分被四舍五入。
    """
    total_scores = 0  # 初始化总得分
    for j in range(turns):  # 对于每一轮评估
        s, info = env.reset()  # 重置环境获取初始状态
        done = False  # 初始化环境终止标志
        while not done:  # 环境未终止时循环
            a = agent.select_action(s, deterministic=True)  # 代理选择确定性的动作
            s_next, r, dw, tr, info = env.step(a)  # 执行动作
            done = (dw or tr)  # 环境是否终止

            total_scores += r  # 更新总得分
            s = s_next  # 更新状态

    return int(total_scores / turns)  # 返回整数形式的平均得分


def str2bool(v):
    """
    将字符串转换为布尔值，主要用于argparse库中，以便从命令行参数中解析布尔值。

    参数:
    - v: 输入的字符串或布尔值。

    返回值:
    - 转换后的布尔值。

    主要步骤:
    1. 如果输入已经是布尔值，则直接返回。
    2. 将字符串转换为小写并检查是否匹配特定的布尔值表示。
    3. 根据匹配结果返回True或False。
    4. 如果输入的字符串不是预期的布尔值表示，则抛出异常。

    注释:
    - 这个函数能够处理字符串的多种布尔值表达方式，包括大小写的'yes', 'true', 't', 'y', '1'等表示True，
      以及'no', 'false', 'f', 'n', '0'等表示False。
    - 如果输入的字符串不在上述列表中，函数将抛出argparse.ArgumentTypeError异常，
      指出期望的是一个布尔值。
    """
    if isinstance(v, bool):
        return v  # 如果v已经是布尔值，则直接返回
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True  # 如果字符串表示True，则返回True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False  # 如果字符串表示False，则返回False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')  # 如果不是预期字符串，则抛出异常


def print_network_parameters(model):
    """
    打印出网络模型的参数。

    参数:
    - model: 要查看参数的网络模型。

    注释:
    - 这个函数会打印出模型中每一层的权重和偏置。
    - 只有那些具有可训练参数的层（例如线性层）的参数会被打印出来。
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"层: {name}")
            print(f"参数尺寸 (Size): {param.size()}")
            print(f"参数值 (Values): \n{param.data}")


def initialize_weights(m):
    """
    初始化网络参数。

    参数:
    - m: 网络模型或模型中的一个层。

    注释:
    - 这个函数会被应用到模型的每一个层上。
    - 可以根据需要选择不同的初始化策略，例如Xavier初始化或He初始化。
    """
    if isinstance(m, nn.Linear):
        # 使用Xavier初始化方法初始化线性层的权重
        torch.nn.init.xavier_uniform_(m.weight)
        # 如果有偏置项，将其初始化为0
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    actor = Double_Q_Critic(state_dim=1, action_dim=4, hid_shape=[2, 3])
    print_network_parameters(actor)
    print("\n初始化网络参数...\n")
    actor.apply(initialize_weights)
    print_network_parameters(actor)