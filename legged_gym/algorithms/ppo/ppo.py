import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from legged_gym.algorithms.ppo.actor_critic import ActorCritic
from legged_gym.algorithms.ppo.rollout_storage import RolloutStorage
from legged_gym.algorithms.ppo import caches


class PPO_Args(PrefixProto):
    """
    PPO_Args 是对 PPO 参数进行设置的类，集成自 PrefixProto

    Attributes:
    - value_loss_coef: 值函数损失的系数，默认为1.0
    - use_clipped_value_loss: 是否使用值函数损失的限幅技巧，定义为布尔值，默认为True
    - clip_param: PPO限幅参数，默认为0.2，作用于策略概率比率的裁剪
    - entropy_coef: 熵系数，用于鼓励探索，促进策略多样性，默认为0.01
    - num_learning_epochs: 学习的次数，默认为5
    - num_mini_batches: 每次学习使用的小批量数据组数量，默认为4
    - learning_rate: 学习率，默认为0.001
    - adaptation_module_learning_rate: 适应模块的学习率，默认为0.001
    - num_adaptation_module_substeps: 适应模块子步骤的数量，默认为1
    - schedule: 学习率调度策略，默认为'adaptive'表示自适应
    - gamma: 折扣因子，决定了未来回报的折现值，默认为0.99
    - lam: GAE(广义优势估算)中的平滑参数，决定了优势的计算方式，默认为0.95
    - desired_kl: 期望的KL散度变化值，用于控制策略更新步伐，默认为0.01
    - max_grad_norm: 梯度剪裁的最大范数，默认为1.0
    - selective_adaptation_module_loss: 是否采用选择性的适应模块损失函数，定义为布尔值，默认为False

    注解:
    1. 此类提供了针对PPO算法的配置参数集合
    2. 使用PPO_Args创建配置对象后，可以方便地传入PPO创建方法中调整算法行为
    3. 可以基于这些默认值进一步调优以改善算法性能和训练效率

    注意:
    - 类继承自PrefixProto，确保具备初始结构属性和方法
    - 对于不同任务和环境，这些参数需要进行调整以取得最佳效果
    """
    value_loss_coef = 1.0  # 设置值函数损失的系数
    use_clipped_value_loss = True  # 定义是否使用限幅的值函数损失
    clip_param = 0.2  # 设置PPO限幅参数
    entropy_coef = 0.01  # 设置熵系数，以鼓励探索
    num_learning_epochs = 5  # 设置学习的迭代次数
    num_mini_batches = 4  # 设置每次学习使用的小批量数据组数量
    learning_rate = 1.e-3  # 设置学习率
    adaptation_module_learning_rate = 1.e-3  # 设置适应模块的学习率
    num_adaptation_module_substeps = 1  # 设置适应模块子步骤的数量
    schedule = 'adaptive'  # 设置学习率调度策略为自适应
    gamma = 0.99  # 设置折扣因子
    lam = 0.95  # 设置GAE中的平滑参数
    desired_kl = 0.01  # 设置期望的KL散度变化值
    max_grad_norm = 1.  # 设置梯度剪裁的最大范数
    selective_adaptation_module_loss = False  # 定义是否使用选择性的适应模块损失函数


class PPO:
    """
    PPO类定义了Proximal Policy Optimization算法的基本功能

    Attributes:
    - actor_critic (ActorCritic): Actor-Critic 模型对象。

    Methods:
    - __init__: 构造函数，初始化PPO对象中的属性。
    - init_storage: 初始化存储系统，用于保持环境转换。
    - test_mode: 将模型设为测试模式。
    - train_mode: 将模型设为训练模式。
    - act: 根据观测值选择动作。
    - process_env_step: 处理环境的步骤反馈。
    - compute_returns: 计算回报。
    - update: 更新模型的参数。

    注解:
    - 初始化中使用Adam优化器来训练actor_critic网络。
    - 存储采用RolloutStorage来保持一系列转换。
    - 训练和测试模式控制actor_critic的行为。
    - 更新算法包括梯度剪裁以及动态调整学习率。

    注意:
    - 必须首先初始化storage才能进行训练。
    - 更新步骤中包含了计算loss和反向传播。
    """

    actor_critic: ActorCritic  # 在此处这一行声明了actor_critic为ActorCritic类型

    def __init__(self, actor_critic, device='cpu'):
        """
        PPO 类的初始化方法

        Args:
        - actor_critic: ActorCritic 类实例，包含策略和价值网络的模型
        - device: 字符串，指定模型应运行在哪个设备上，默认为'cpu'

        注解:
        1. 创建各种优化器并将模型移动到指定的设备（如CPU或GPU）
        2. 初始化转换存储结构，以在学习过程中保存数据

        注意:
        - 如果actor_critic有解码器(decode)部分，则为其单独创建一个优化器
        - 初始化中还引入了用于存储转换状态的RolloutStorage.Transition对象
        """
        self.device = device  # 设备名称，默认为 'cpu'
        self.actor_critic = actor_critic  # 赋值ActorCritic模型实例
        self.actor_critic.to(device)  # 将actor_critic模型移到指定的设备上
        self.storage = None  # 初始化时，没有存储器

        # 使用Adam优化器，为actor_critic的参数设置默认学习率
        self.teacher_actor_optimizer = optim.Adam(
            self.actor_critic.teacher_actor_body.parameters(),
            lr=PPO_Args.learning_rate
        )
        self.student_actor_optimizer = optim.Adam(
            self.actor_critic.student_actor_body.parameters(),
            lr=PPO_Args.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.actor_critic.critic_body.parameters(),
            lr=PPO_Args.learning_rate
        )

        # 使用Adam优化器，并为适应模块设置学习率
        self.adaptation_module_optimizer = optim.Adam(
            self.actor_critic.adaptation_module.parameters(),
            lr=PPO_Args.adaptation_module_learning_rate
        )

        # 如果actor_critic中包含解码器，则为解码器创建一个独立的Adam优化器
        if self.actor_critic.decoder:
            self.decoder_optimizer = optim.Adam(
                self.actor_critic.decoder.parameters(),
                lr=PPO_Args.decoder_learning_rate
            )

        self.transition = RolloutStorage.Transition()  # 创建转换储存结构的实例

        self.learning_rate = PPO_Args.learning_rate  # 保存PPO_Args中设定的学习率

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     actions_history_shape, action_shape):
        """
        初始化回合储存容器以保存经验数据。

        Args:
        - num_envs: 每个环境的数量
        - num_transitions_per_env: 每个环境中转换（transition）的数量
        - actor_obs_shape: 普通观察空间的形状
        - privileged_obs_shape: 特权观察空间的形状
        - obs_history_shape: 观察历史空间的形状
        - action_shape: 动作空间的形状

        注解:
        - RolloutStorage用于存储策略过程中的数据，这在训练过程中是必需的。
        - 这些形状被用来初始化正确定义存储结构的尺寸。

        注意:
        - 必须确保输入参数正确匹配实际情况，否则可能影响训练的有效性和稳定性。
        """
        # 创建并初始化一个RolloutStorage实例用于保存转换数据
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, actions_history_shape, action_shape, self.device)

    def test_mode(self):
        """
        将模型设置为测试模式。

        注解:
        - 测试模式下，模型不会更新梯度，只用于评价。

        注意:
        - 在进行模型评估之前，应该调用此方法。
        """
        self.actor_critic.test()  # 调用actor_critic的test方法，进入测试模式

    def train_mode(self):
        """
        将模型设置为训练模式。

        注解:
        - 训练模式下，模型将更新梯度。

        注意:
        - 在开始训练之前，调用此方法来确保模型处于正确的状态。
        """
        self.actor_critic.train()  # 调用actor_critic的train方法，进入训练模式

    def act(self, obs, privileged_obs, obs_history, acts_history, num_envs, action_max):
        """
        根据给定的观测来选择动作，并存储相关的信息。

        Args:
        - obs: 当前环境的观测数据
        - privileged_obs: 特权观测数据，可能包含额外的信息用于训练
        - obs_history: 包含之前观测的历史信息

        Returns:
        - 动作，从actor_critic模型中生成

        注解:
        1. 使用actor_critic模型基于观测历史提供的信息来选择动作。
        2. 评估提供的动作及其对应的价值。
        3. 从actor_critic获取动作的对数概率，均值和标准差。
        4. 将当前的观测，批评者观测，特权观测以及观测历史存储到transition结构中。

        注意:
        - 该函数不仅选择动作，还对返回的动作及其相关特性进行记录，这对于后续的学习步骤至关重要。
        - 所有获得的Tensor数据都会通过detach()方法来阻断梯度，这样在之后存储和使用这些数据时不会影响梯度传递。
        """
        # 调用actor_critic模型来选择动作并将梯度移除(detach)
        actions, c = self.actor_critic.act_teacher(obs_history, privileged_obs)
        self.transition.actions = actions.detach()
        num_command = int(num_envs / 2)
        c = c.squeeze()  # 去除多余的维度
        mask1 = c[:num_command] < 0.25
        mask2 = (c[:num_command] >= 0.25) & (c[:num_command] < 0.5)
        mask3 = (c[:num_command] >= 0.5) & (c[:num_command] < 0.75)
        mask4 = (c[:num_command] >= 0.75) & (c[:num_command] < 1.0)

        obs[:num_command, 8:11] = 0.0  # 先将所有值设为0

        obs[:num_command, 8]= torch.where(mask2, 0.5, obs[:num_command, 8:11][:, 0])
        obs[:num_command, 9] = torch.where(mask3, 0.5, obs[:num_command, 8:11][:, 1])
        obs[:num_command, 10] = torch.where(mask4, 0.5, obs[:num_command, 8:11][:, 2])

        # 评估当前状态的价值函数并将梯度移除(detach)
        self.transition.values = self.actor_critic.evaluate(obs_history, privileged_obs).detach()

        # 获取选择动作的对数概率并将梯度移除(detach)
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions).detach()

        # 获取动作的均值和标准差并分别将梯度移除(detach)
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # 存储当前的观测、批评者观测即普通观测、特权观测以及观测历史
        self.transition.observations = obs
        self.transition.critic_observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history

        self.transition.action_histories = acts_history

        # 返回选择的动作
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        """
        处理从环境中获得的步骤信息，并将其存储起来。

        Args:
        - rewards: 步骤的回报
        - dones: 表示每个环境是否完成了回合的布尔数组
        - infos: 包含额外信息字典，例如环境bins和时间终止

        注解:
        1. 存储回报、完成标志、环境的额外信息。
        2. 如果 'time_outs' 信息存在，则更新回报。
        3. 将这一步的transition加入到storage中。
        4. 清空transition以便下一步使用。
        5. 如果环境完成，则重置actor_critic状态。

        注意:
        - 这里的处理步骤对于正确存储经验和更新策略非常重要。
        - 回报(rewards)在存储前进行了克隆，以避免在原值上做改变。
        - 'time_outs' 处理的目的是为了给那些因为超时而终止而非因为策略导致终止的回合一个额外的值。
        """
        self.transition.rewards = rewards.clone()  # 复制从环境中得来的回报，避免在原变量上修改
        self.transition.dones = dones  # 存储环境是否结束的信息
        self.transition.env_bins = infos["env_bins"]  # 存储额外的环境信息如env_bins

        # 如果infos中包含 'time_outs' 键值，相应处理并更新回报
        if 'time_outs' in infos:
            self.transition.rewards += PPO_Args.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1
            )

        self.storage.add_transitions(self.transition)  # 将当前transition添加到存储中
        self.transition.clear()  # 清除transition以便新一步的数据存储
        self.actor_critic.reset(dones)  # 如果有环境已经结束，则重置actor_critic的状态

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        """
        计算累积回报（即返回值），以便后续训练。

        Args:
        - last_critic_obs: 最后一个时间步长的观察值，用于评估值函数
        - last_critic_privileged_obs: 最后一个时间步长的特权观察值，可能包含额外信息

        注解:
        1. 使用actor_critic模型评估最后一步的值函数并从中分离梯度。
        2. 用这个最终的值估计和指定的折扣因子、lambda值来计算累积回报。

        注意:
        - 后续训练步骤中使用的累积回报（如优势函数等）依赖于这里的计算结果。
        - 计算结束后，这些累积回报将被存储于之前初始化的storage对象中。
        - 传递给storage.compute_returns方法的gamma和lam参数定义了回报计算使用的折扣因子和泛化程度。
        """
        # 评估最后一个状态对应的值，并与计算梯度分离
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()

        # 计算存储中所有未完成状态的累积回报，并添加到存储对象中，使用PPO_Args中定义的gamma和lambda参数
        self.storage.compute_returns(last_values, PPO_Args.gamma, PPO_Args.lam)

    def update(self):
        """
        根据Proximal Policy Optimization (PPO)算法来更新模型参数

        返回值:
        各项损失的平均值，包括价值损失、替代损失、适配模块损失等

        注解:
        1. 初始化各种平均损失的累加器。
        2. 从存储器中获取批量的小批次数据生成器。
        3. 遍历生成器中的数据，执行PPO优化过程。
        4. 调整学习率以维持一个期望的KL散度，如果使用自适应调整计划。
        5. 计算价值和策略损失，应用梯度剪切并执行一步参数优化。
        6. 计算适应模块的损失。
        7. 清空存储器。
        8. 返回所有损失的平均值。

        注意:
        - 算法细节和超参数使用PPO_Args类中定义的静态值。
        - 整个更新过程中没有显示的输出，仅在最后返回损失的平均值。
        """
        mean_value_loss = 0  # 初始化平均价值损失累加器
        mean_surrogate_loss = 0  # 初始化平均替代损失累加器
        mean_adaptation_module_loss = 0  # 初始化平均适配模块损失累加器
        mean_student_module_loss = 0
        mean_decoder_loss = 0  # 初始化平均解码器损失累加器
        mean_decoder_loss_student = 0  # 初始化平均学生解码器损失累加器
        mean_adaptation_module_test_loss = 0  # 初始化平均适配模块测试损失累加器
        mean_student_module_test_loss = 0
        mean_decoder_test_loss = 0  # 初始化平均解码器测试损失累加器
        mean_decoder_test_loss_student = 0  # 初始化平均学生解码器测试损失累加器

        # 从存储器中获取用于训练的小批次数据生成器，参数定义了小批次数量和学习的周期
        generator = self.storage.mini_batch_generator(PPO_Args.num_mini_batches, PPO_Args.num_learning_epochs)

        # 迭代通过生成器产生的数据批次，执行PPO（近邻策略优化）的训练过程
        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_history_batch, \
                actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, masks_batch, env_bins_batch in generator:

            # 利用历史观察批次和mask执行actor-critic的行动决策函数
            # 使用历史观察和mask来做决策
            self.actor_critic.act_teacher(obs_history_batch, privileged_obs_batch, masks=masks_batch)
            # 获取当前动作的对数概率
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)  # 计算动作的对数概率
            # 评估给定观察批次对应的值
            value_batch = self.actor_critic.evaluate(obs_history_batch, privileged_obs_batch, masks=masks_batch)  # 评估状态价值
            # 获取行动均值和标准差以及熵
            mu_batch = self.actor_critic.action_mean  # 获取动作均值
            sigma_batch = self.actor_critic.action_std  # 获取动作标准差
            entropy_batch = self.actor_critic.entropy  # 计算熵，以评估策略随机性

            # 如果策略使用了自适应KL散度调整，则根据KL散度调整学习率
            if PPO_Args.desired_kl is not None and PPO_Args.schedule == 'adaptive':
                with torch.inference_mode():  # 在推断模式下减少计算开销
                    # 计算KL散度
                    kl = torch.sum(torch.log(sigma_batch / old_sigma_batch + 1.e-5) +
                                   (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) /
                                   (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    # 计算KL散度的均值
                    kl_mean = torch.mean(kl)

                    # 根据KL散度均值调整学习率
                    if kl_mean > PPO_Args.desired_kl * 2.0:  # 如果KL散度大于期望值的两倍
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)  # 降低学习率
                    elif kl_mean < PPO_Args.desired_kl / 2.0 and kl_mean > 0.0:  # 如果KL散度小于期望值的一半
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)  # 提高学习率

                    # 更新优化器的学习率参数
                    for param_group in self.teacher_actor_optimizer.param_groups:
                        param_group['lr'] = self.learning_rate  # 更新学习率

            # 计算比率、替代损失，执行梯度裁剪，并更新参数
            # 计算新旧概率的比率（比率=exp(新概率-旧概率)）
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            # 未裁剪的替代损失
            surrogate = -torch.squeeze(advantages_batch) * ratio  # 计算未裁剪的替代损失
            # 裁剪后的替代损失
            surrogate_clipped = (-torch.squeeze(advantages_batch) *
                                 torch.clamp(ratio, 1.0 - PPO_Args.clip_param, 1.0 + PPO_Args.clip_param))
            # 替代损失是未裁剪和裁剪损失中的最大值
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()  # 取两者中的较大值进行后续计算

            # 根据是否启用裁剪的价值损失来选择计算方式
            if PPO_Args.use_clipped_value_loss:
                # 计算裁剪后的价值
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-PPO_Args.clip_param, PPO_Args.clip_param)
                # 裁剪和未裁剪的价值损失
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                # 计算价值损失
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                # 如果不使用裁剪的价值损失，直接计算标准的价值损失
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 计算总损失并反向传播
            # 总损失包括替代损失、价值损失和熵的加权和
            loss = surrogate_loss + PPO_Args.value_loss_coef * value_loss - PPO_Args.entropy_coef * entropy_batch.mean()

            self.teacher_actor_optimizer.zero_grad()  # 清空之前的梯度
            self.critic_optimizer.zero_grad()  # 清空之前的梯度
            loss.backward()  # 反向传播计算梯度
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPO_Args.max_grad_norm)  # 梯度裁剪防止极端值
            self.teacher_actor_optimizer.step()  # 根据梯度更新参数
            self.critic_optimizer.step()

            # 累加价值损失和代理损失，用于后续分析或输出
            mean_value_loss += value_loss.item()  # 累加价值损失
            mean_surrogate_loss += surrogate_loss.item()  # 累加替代损失

            # 准备适配模块所需的数据，分割出训练集和验证集
            data_size = privileged_obs_batch.shape[0]  # 获取特权观察数据的大小
            num_train = int(data_size // 5 * 4)  # 把80%的数据用作训练数据

            # 训练适配模块并计算相关损失
            for epoch in range(PPO_Args.num_adaptation_module_substeps):
                # 通过适配模块预测特权观察数据
                adaptation_pred = self.actor_critic.adaptation_module(torch.cat((obs_history_batch,
                                                                                 actions_history_batch), dim=-1))
                with torch.no_grad():  # 不计算梯度，为求解适配目标
                    adaptation_target = privileged_obs_batch  # 设定特权观察数据为适配目标

                # 生成选择性适配损失索引
                selection_indices = torch.linspace(0, adaptation_pred.shape[1] - 1,
                                                   steps=adaptation_pred.shape[1],
                                                   dtype=torch.long)
                if PPO_Args.selective_adaptation_module_loss:
                    selection_indices = 0  # 如果采用选择性适配损失，则选择一个特征索引

                # 计算适配模块的损失
                adaptation_loss = F.mse_loss(adaptation_pred[:num_train, selection_indices],
                                             adaptation_target[:num_train, selection_indices])  # 训练集上的适配损失
                adaptation_test_loss = F.mse_loss(adaptation_pred[num_train:, selection_indices],
                                                  adaptation_target[num_train:, selection_indices])  # 测试集上的适配损失

                self.adaptation_module_optimizer.zero_grad()  # 清空适配模块优化器的梯度
                adaptation_loss.backward()  # 计算适配模块损失的梯度
                self.adaptation_module_optimizer.step()  # 根据梯度更新适配模块的参数

                mean_adaptation_module_loss += adaptation_loss.item()  # 将适配模块损失累加到平均损失
                mean_adaptation_module_test_loss += adaptation_test_loss.item()  # 将适配模块测试损失累加到平均测试损失

            # 训练适配模块并计算相关损失
            for epoch in range(PPO_Args.num_adaptation_module_substeps):
                # 通过适配模块预测特权观察数据
                adaptation_pred = self.actor_critic.adaptation_module(torch.cat((obs_history_batch,
                                                                                 actions_history_batch), dim=-1))
                action_pred = self.actor_critic.student_actor_body(torch.cat((obs_history_batch,
                                                                              adaptation_pred), dim=-1))
                with torch.no_grad():  # 不计算梯度，为求解适配目标
                    action_target = self.actor_critic.teacher_actor_body(torch.cat((obs_history_batch,
                                                                                    privileged_obs_batch), dim=-1))

                # 生成选择性适配损失索引
                selection_indices = torch.linspace(0, action_pred.shape[1] - 1,
                                                   steps=action_pred.shape[1], dtype=torch.long)
                if PPO_Args.selective_adaptation_module_loss:
                    selection_indices = 0  # 如果采用选择性适配损失，则选择一个特征索引

                # 计算适配模块的损失
                actions_loss = F.mse_loss(action_pred[:num_train, selection_indices],
                                          action_target[:num_train, selection_indices])  # 训练集上的适配损失
                actions_test_loss = F.mse_loss(action_pred[num_train:, selection_indices],
                                               action_target[num_train:, selection_indices])  # 测试集上的适配损失

                self.student_actor_optimizer.zero_grad()  # 清空适配模块优化器的梯度
                actions_loss.backward()  # 计算适配模块损失的梯度
                self.student_actor_optimizer.step()  # 根据梯度更新适配模块的参数

                mean_student_module_loss += actions_loss.item()  # 将适配模块损失累加到平均损失
                mean_student_module_test_loss += actions_test_loss.item()  # 将适配模块测试损失累加到平均测试损失

        # 在每个训练步骤后清除存储器的内容
        self.storage.clear()

        # 计算所有损失的均值并返回
        return mean_value_loss, \
            mean_surrogate_loss, \
            mean_adaptation_module_loss, \
            mean_decoder_loss, \
            mean_decoder_loss_student, \
            mean_adaptation_module_test_loss, \
            mean_decoder_test_loss, \
            mean_decoder_test_loss_student, \
            mean_student_module_loss, \
            mean_student_module_test_loss
