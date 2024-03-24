import torch

from legged_gym.algorithms.utils import split_and_pad_trajectories


class RolloutStorage:
    """
    RolloutStorage 用于存储和管理在强化学习训练中的数据

    Attributes:
        device (str): 存储数据使用的设备类型（如 'cpu' 或 'cuda'）
        obs_shape (tuple): 观测数据的形状
        privileged_obs_shape (tuple): 特权观测数据的形状（用于特殊算法，如Hindsight Experience Replay）
        obs_history_shape (tuple): 历史观测数据的形状
        actions_shape (tuple): 动作数据的形状
        observations (Tensor): 观测数据的存储张量
        privileged_observations (Tensor): 特权观测数据的存储张量
        observation_histories (Tensor): 历史观测数据的存储张量
        rewards (Tensor): 奖励数据的存储张量
        actions (Tensor): 动作数据的存储张量
        dones (Tensor): 完成标记的存储张量（指示一个阶段或者环境是否完结）
        actions_log_prob (Tensor): 动作的对数概率的存储张量
        values (Tensor): 价值函数的估计值的存储张量
        returns (Tensor): 折扣回报的存储张量
        advantages (Tensor): 优势函数的估计值的存储张量
        mu (Tensor): 动作分布的平均值的存储张量
        sigma (Tensor): 动作分布的标准差的存储张量
        env_bins (Tensor): 环境区分的存储张量（可用于多任务环境）
        num_transitions_per_env (int): 每个环境中的转换次数
        num_envs (int): 环境的数量
        step (int): 当前的步骤，用于追踪存储的转换

    注解:
        初始化的时候，提供了环境和转换的尺寸信息，并初始化了各种数据的张量。
        ‘Transition’ 类定义了强化学习过程中的一个转换所需要的各种数据。
    """

    class Transition:
        """
        Transition类用于存储和表示在强化学习环境中的一个转换（transition）。

        属性：
        - observations: 存储当前状态的观测值。
        - privileged_observations: 存储额外的观测值，可能用于提供给特权代理或分析。
        - observation_histories: 存储观测值的历史记录，通常用于处理部分可观测环境。
        - critic_observations: 存储用于价值函数（critic）的观测值。
        - actions: 存储所执行动作的数据。
        - rewards: 存储从环境中得到的奖励。
        - dones: 标识交互是否结束的布尔值，例如是否到达了终止状态。
        - values: 存储价值函数的估计值。
        - actions_log_prob: 存储动作的对数概率，可能用于概率性策略。
        - action_mean: 存储动作平均值，可能用于连续动作空间。
        - action_sigma: 存储动作的标准差，可能用于连续动作空间。
        - env_bins: 可用于存储环境离散化后的各个区间或箱。

        方法：
        - clear: 重置所有属性，为下一个transition做准备。
        """

        def __init__(self):
            """
            初始化Transition实例，为所有属性设置初始值None。
            """
            self.observations = None  # 初始状态观测
            self.privileged_observations = None  # 特权状态观测
            self.observation_histories = None  # 观测的历史记录
            self.critic_observations = None  # 价值函数所需观测
            self.actions = None  # 执行的动作
            self.rewards = None  # 获得的奖励
            self.dones = None  # 交互是否结束标识
            self.values = None  # 价值函数估计值
            self.actions_log_prob = None  # 动作的对数概率
            self.action_mean = None  # 动作的均值（对于连续空间）
            self.action_sigma = None  # 动作的标准差（对于连续空间）
            self.env_bins = None  # 环境离散化的区间或箱
            self.action_histories = None

        def clear(self):
            """
            重置Transition对象的状态。该方法重新初始化所有属性，为新的transition准备。
            """
            self.__init__()  # 通过调用__init__方法来重置所有属性

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, obs_history_shape,
                 actions_history_shape, actions_shape, device='cpu'):
        """
        初始化轨迹缓存区RolloutBuffer类的构造函数

        Args:
        - num_envs: 同时运行的环境数量
        - num_transitions_per_env: 每个环境中转换(经历)的数量
        - obs_shape: 普通观测的形状
        - privileged_obs_shape: 特权观测(附加信息)的形状
        - obs_history_shape: 观测历史的形状
        - actions_shape: 动作的形状
        - device: 存储数据的设备，默认是'cpu'

        注解:
        1. 初始化存于特定设备内存上的变量，用以存储轨迹数据。
        2. 分别为观测、特权观测、观测历史、奖励、动作和结束标记创建对应形状的张量矩阵，并初始化为零。
        3. 为记录概率、值函数、回报以及优势函数也分别创建对应形状的张量矩阵，并初始化为零。
        4. 创建存储环境编号的张量矩阵。
        5. 初始化总转换次数以及当前步骤编号。
        """
        # 存储数据的设备，可以是 'cpu' 或者 'cuda' 如果可用
        self.device = device

        # 观测数据的维度形状
        self.obs_shape = obs_shape
        # 特权观测的维度，可能包含更多的环境信息
        self.privileged_obs_shape = privileged_obs_shape
        # 观测历史的维度，用于时间序列分析
        self.obs_history_shape = obs_history_shape
        # 动作历史
        self.actions_shape = actions_history_shape
        # 动作的维度形状
        self.actions_shape = actions_shape

        # 在指定设备上为每个环境的每个转换步骤初始化观测张量为零
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        # 在指定设备上为每个环境的每个转换步骤初始化特权观测张量为零
        self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape,
                                                   device=self.device)
        # 在指定设备上为每个环境的每个转换步骤初始化观测历史张量为零
        self.observation_histories = torch.zeros(num_transitions_per_env, num_envs, *obs_history_shape,
                                                 device=self.device)
        self.actions_histories = torch.zeros(num_transitions_per_env, num_envs, *actions_history_shape,
                                                 device=self.device)
        # 在指定设备上为每个环境的每个转换步骤初始化奖励张量为零
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # 在指定设备上为每个环境的每个转换步骤初始化动作张量为零
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # 在指定设备上为每个环境的每个转换步骤初始化结束标志张量为零，并将数据类型指定为byte
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # 在指定设备上为每个环境的每个转换步骤初始化动作的对数概率张量为零
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # 在指定设备上为每个环境的每个转换步骤初始化价值函数张量为零
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # 在指定设备上为每个环境的每个转换步骤初始化回报张量为零
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # 在指定设备上为每个环境的每个转换步骤初始化优势函数张量为零
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # 在指定设备上为每个环境的每个转换步骤初始化策略的均值张量为零
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # 在指定设备上为每个环境的每个转换步骤初始化策略的标准差张量为零
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # 在指定设备上为每个环境的每个转换步骤初始化环境编号张量为零
        self.env_bins = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # 记录共有多少个环境同时运行
        self.num_envs = num_envs
        # 记录每个环境中会有多少转换步骤
        self.num_transitions_per_env = num_transitions_per_env

        # 初始化当前步骤到零，用于追踪轨迹缓存的步进
        self.step = 0

    def add_transitions(self, transition: Transition):
        """
        向轨迹缓存添加单步转换数据

        Args:
        - transition: 一个包含所有必要字段的转换实例（Transition）

        Raises:
        - AssertionError: 如果步数超出了设定的每个环境的转换数量，将引发错误

        注解:
        1. 检查当前步骤是否已达到每个环境的转换上限，如果是，则触发断言错误。
        2. 将输入的transition中各项数据复制到轨迹缓存的相应位置。
        3. 对于非标量数据，使用view(-1, 1)确保数据维度与轨迹缓存一致。
        4. 每次添加数据后，步数self.step自增1，为下次添加做准备。
        """
        # 检查当前环境步数是否超过了限制，如超过则触发断言
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        # 复制普通观测到缓存对应步骤
        self.observations[self.step].copy_(transition.observations)
        # 复制特权观测到缓存对应步骤
        self.privileged_observations[self.step].copy_(transition.privileged_observations)
        # 复制观测历史到缓存对应步骤
        self.observation_histories[self.step].copy_(transition.observation_histories)
        # 复制动作历史到缓存对应步骤
        self.actions_histories[self.step].copy_(transition.action_histories)
        # 复制动作到缓存对应步骤
        self.actions[self.step].copy_(transition.actions)
        # 把奖励数据转换为列向量并复制到缓存对应步骤
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        # 把结束标志(dones)转换为列向量并复制到缓存对应步骤
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # 复制价值估计到缓存对应步骤
        self.values[self.step].copy_(transition.values)
        # 把动作的对数概率数据转换为列向量并复制到缓存对应步骤
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        # 复制动作的均值到缓存对应步骤
        self.mu[self.step].copy_(transition.action_mean)
        # 复制动作的标准差到缓存对应步骤
        self.sigma[self.step].copy_(transition.action_sigma)
        # 把环境编号数据转换为列向量并复制到缓存对应步骤
        self.env_bins[self.step].copy_(transition.env_bins.view(-1, 1))

        # 完成本步骤数据的添加后，准备进入下一个步骤
        self.step += 1

    def clear(self):
        """
        清除轨迹缓存的当前步骤计数

        注解:
        - 重新设置步骤self.step为0，为新的轨迹记录做准备。
        """
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        """
        计算每步的累积回报

        Args:
        - last_values: 最后一个观测值的价值估计
        - gamma: 折扣因子
        - lam: GAE(Generalized Advantage Estimation)的平滑参数

        注解:
        1. 从最后一个步骤开始向前计算，并重置advantage为0。
        2. 计算TD残差（Temporal Difference Residual），即回报与预期回报的差值。
        3. 使用TD残差更新advantage估计。
        4. 计算实际回报（returns）值。
        5. 计算并更新每个步骤上的优势函数估计值（advantages）。
        6. 对优势函数进行标准化处理。
        """
        # 初始化优势变量
        advantage = 0
        # 逆序遍历所有转换，计算累积回报和优势
        for step in reversed(range(self.num_transitions_per_env)):
            # 判断是否为最后一步，从而决定接下来的值
            if step == self.num_transitions_per_env - 1:
                next_values = last_values  # 如果是最后一步，使用传入的last_values
            else:
                next_values = self.values[step + 1]  # 如果不是，使用下一步的值

            # 下一步没有终止即为True (1.0)，终止了即为False (0.0)
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # 计算TD残差，即reward增加一步预测值和当前预测值的差
            delta = self.rewards[step] + gamma * next_is_not_terminal * next_values - self.values[step]
            # 根据GAE的方法更新advantage
            advantage = delta + gamma * lam * next_is_not_terminal * advantage
            # 计算returns，即当前步骤的价值和advantage和
            self.returns[step] = advantage + self.values[step]

        # 计算每个步骤的优势
        self.advantages = self.returns - self.values
        # 将优势进行归一化，即减去平均值，除以标准差
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        """
        获取轨迹的统计数据

        Returns:
        - 平均轨迹长度: 所有轨迹的平均步数
        - 平均奖励: 所有步中的平均奖励值

        注解:
        1. 计算每一轨迹的长度，首先设定最后一步必定结束。
        2. 通过交换维度和重塑张量获取所有的结束标志。
        3. 找到结束标志的索引，并将这些索引放入一个张量中。
        4. 通过比较相邻结束点之间的差异，计算每条轨迹的长度。
        5. 返回轨迹长度的平均值以及奖励的平均值。
        """
        # 将dones的最后一个元素设置为1，确保最后一轨迹能够结束
        done = self.dones
        done[-1] = 1
        # 重新排列dones的维度并将它们展平成一维数组
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        # 获取展平后数组中所有结束标志的索引，并在第一个位置手动添加-1作为起始索引
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        # 计算所有轨迹的长度（当前结束索引 - 前一结束索引）
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        # 返回轨迹长度的平均值以及所有奖励的平均值
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """
        生成mini-batch用于训练

        Args:
        - num_mini_batches: 指定要生成的mini-batches数量
        - num_epochs: 遍历数据集的轮数，默认为8

        Yields:
        - 一系列mini-batches，每个mini-batch包含用于训练模型的各种数据

        注解:
        1. 计算完整批大小以及每个mini-batch的大小。
        2. 随机生成索引，用于生成mini-batches。
        3. 将存储的观测、动作等数据展开为二维，以生成mini-batches。
        4. 遍历所有epoch，在每个epoch中遍历所有mini-batch。
        5. 为每个mini-batch抽取对应索引的数据。
        6. 将抽取的数据yield出来，以便用于训练。

        注意事项:
        - torch.randperm用于生成随机索引，需要指定device和是否需要计算梯度。
        - mini-batches用于梯度下降训练，在强化学习的Proximal Policy Optimization(PPO)等算法中是常见的。
        """
        batch_size = self.num_envs * self.num_transitions_per_env  # 完整批次的大小
        mini_batch_size = batch_size // num_mini_batches  # 每个mini-batch的大小

        # 生成随机索引，确定每个mini-batch的训练数据
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # 展平数据，用于生成mini-batches
        observations = self.observations.flatten(0, 1)  # 观测值
        privileged_obs = self.privileged_observations.flatten(0, 1)  # 特权观测值
        obs_history = self.observation_histories.flatten(0, 1)  # 观测历史
        actions_histories = self.actions_histories.flatten(0, 1)  # 动作历史
        critic_observations = observations  # 临界观测值

        actions = self.actions.flatten(0, 1)  # 动作
        values = self.values.flatten(0, 1)  # 价值估计
        returns = self.returns.flatten(0, 1)  # 回报
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)  # 旧动作的对数概率
        advantages = self.advantages.flatten(0, 1)  # 优势估计
        old_mu = self.mu.flatten(0, 1)  # 旧政策的mu参数
        old_sigma = self.sigma.flatten(0, 1)  # 旧政策的sigma参数
        old_env_bins = self.env_bins.flatten(0, 1)  # 环境状态的分箱

        # 定义了一个遍历训练数据的过程，用于创建mini-batch供模型训练使用。
        for epoch in range(num_epochs):  # 对于每一个epoch
            for i in range(num_mini_batches):  # 遍历每一个mini-batch
                start = i * mini_batch_size  # 计算当前mini-batch的起始索引
                end = (i + 1) * mini_batch_size  # 计算当前mini-batch的结束索引
                batch_idx = indices[start:end]  # 根据起始和结束索引确定当前mini-batch的索引范围

                # 使用当前mini-batch的索引来提取对应的数据集
                obs_batch = observations[batch_idx] # 观测数据
                critic_observations_batch = critic_observations[batch_idx]  # 评论员视角的观测数据
                privileged_obs_batch = privileged_obs[batch_idx]  # 特权观测数据
                obs_history_batch = obs_history[batch_idx]  # 观测历史数据
                actions_history_batch = actions_histories[batch_idx]
                actions_batch = actions[batch_idx]  # 动作数据
                target_values_batch = values[batch_idx]  # 目标值数据
                returns_batch = returns[batch_idx]  # 回报数据
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]  # 旧的动作对数概率数据
                advantages_batch = advantages[batch_idx]  # 优势数据
                old_mu_batch = old_mu[batch_idx]  # 旧的动作均值数据
                old_sigma_batch = old_sigma[batch_idx]  # 旧的动作标准差数据
                env_bins_batch = old_env_bins[batch_idx]  # 环境分箱数据

                # 使用yield语句提供当前mini-batch的数据，以便外部调用时可以迭代获取
                yield obs_batch, critic_observations_batch, privileged_obs_batch, obs_history_batch, \
                    actions_history_batch, actions_batch, target_values_batch, advantages_batch, \
                    returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, \
                    None, env_bins_batch

    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """
        创建一个递归的mini-batch生成器

        Args:
        - num_mini_batches: 指定mini-batch的数量。
        - num_epochs: 遍历全部数据的轮数，默认为8。

        Returns:
        以generator的形式逐个yield以下mini-batch的数据：
        - obs_batch: 观测数据的mini-batch。
        - critic_obs_batch: 评论员视角观测数据的mini-batch。
        - privileged_obs_batch: 特权观测数据的mini-batch。
        - obs_history_batch: 观测历史数据的mini-batch。
        - actions_batch: 动作数据的mini-batch。
        - values_batch: 价值数据的mini-batch。
        - advantages_batch: 优势数据的mini-batch。
        - returns_batch: 回报数据的mini-batch。
        - old_actions_log_prob_batch: 旧动作对数概率数据的mini-batch。
        - old_mu_batch: 旧均值数据的mini-batch。
        - old_sigma_batch: 旧标准差数据的mini-batch。
        - masks_batch: 轨迹掩码的mini-batch。

        注解:
        1. 使用split_and_pad_trajectories函数处理观测数据，并创建掩码。
        2. 计算mini-batch的大小。
        3. 遍历每个epoch和mini-batch，生成并返回训练所需的数据。
        4. 使用强化学习中的经验回放机制处理轨迹并分批训练。
        5. 通过生成器yield数据，可用于循环调用以得到每个mini-batch的数据集。
        """

        # 这一代码段的功能是对强化学习中的轨迹数据进行处理，并为它们创建相应的掩码。
        # 处理观测数据的轨迹，对不同长度的轨迹进行分割和填充，以便它们具有相同的形状，并创建轨迹的掩码
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        # 处理特权观测数据的轨迹，这是在某些强化学习设置中，一个额外的观测数据集，可能包含更多信息，但在实际应用中不可用
        padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        # 处理观测历史数据的轨迹，这包含了先前观测点的历史信息
        padded_obs_history_trajectories, _ = split_and_pad_trajectories(self.observation_histories, self.dones)
        # 评论员（Critic）视角的观测数据轨迹使用相同的处理过程，这里直接使用处理后的观测轨迹，因为它们对于评论员而言是相同的
        padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches  # 计算mini-batch的大小
        for ep in range(num_epochs):  # 遍历每个epoch
            first_traj = 0  # 初始化第一个轨迹
            for i in range(num_mini_batches):  # 遍历每个mini-batch
                start = i * mini_batch_size  # 开始位置
                stop = (i + 1) * mini_batch_size  # 结束位置

                # 对done信号进行处理，以便分割轨迹
                dones = self.dones.squeeze(-1)  # 从多余的维度中去除
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)  # 创建一个全False的tensor
                last_was_done[1:] = dones[:-1]  # 标记结束的轨迹
                last_was_done[0] = True  # 第一帧默认为轨迹开始
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])  # 计算每个mini-batch的轨迹数量
                last_traj = first_traj + trajectories_batch_size  # 储存最后一条轨迹的末尾

                # 根据轨迹分割数据
                masks_batch = trajectory_masks[:, first_traj:last_traj]  # 轨迹的掩码
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]  # 观测值
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]  # 评论员视角的观测值
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]  # 特权观测值
                obs_history_batch = padded_obs_history_trajectories[:, first_traj:last_traj]  # 观测历史

                # 获取当前mini-batch的其他相关数据以供后续训练使用。
                actions_batch = self.actions[:, start:stop]  # 从整体动作数据中提取当前的mini-batch动作数据
                old_mu_batch = self.mu[:, start:stop]  # 从mu数据中提取当前的mini-batch，代表之前策略的动作均值
                old_sigma_batch = self.sigma[:, start:stop]  # 从sigma数据中提取当前的mini-batch，代表之前策略的动作标准差
                returns_batch = self.returns[:, start:stop]  # 从回报数据中提取当前的mini-batch，用于模型的优化
                advantages_batch = self.advantages[:, start:stop]  # 从优势函数数据中提取当前的mini-batch，它衡量实际回报与预期回报的差异
                values_batch = self.values[:, start:stop]  # 从价值函数估计中提取当前的mini-batch，用于计算优势函数
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]  # 从动作的旧的对数概率中提取当前的mini-batch

                # 使用yield语句提供当前mini-batch的数据，以便外部调用时可以迭代获取
                yield obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, values_batch, \
                    advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, masks_batch

                first_traj = last_traj  # 更新新轨迹的起始位置
