import torch
import torch.nn as nn
from torch.distributions import Normal
from params_proto import PrefixProto


class AC_Args(PrefixProto, cli=False):
    """
    定义了AC（Actor-Critic）算法相关参数的类

    继承自:
    - PrefixProto: 假定是一个进行参数前缀协议处理的基类
    - cli: 表示这个类是否也用于命令行接口设置，默认为False不用于命令行

    属性:
    - init_noise_std: 初始噪声标准差，浮点数，默认为1.0
    - actor_hidden_dims: actor网络的隐藏层维度列表，默认为[512, 256, 128]
    - critic_hidden_dims: critic网络的隐藏层维度列表，默认与actor相同
    - activation: 激活函数的类型，这里默认为'elu'（指数线性单元）
    - adaptation_module_branch_hidden_dims: 自适应模块分支的隐藏层维度列表，默认为[256, 128]
    - use_decoder: 表示是否使用解码器，布尔值，默认为False
    """
    init_noise_std = 1.0  # 初始化噪声的标准差
    actor_hidden_dims = [512, 256, 128]  # actor网络隐藏层的维度
    critic_hidden_dims = [512, 256, 128]  # critic网络隐藏层的维度
    activation = 'elu'  # 使用的激活函数类型
    adaptation_module_branch_hidden_dims = [256, 128]  # 自适应模块分支的隐藏层维度
    use_decoder = False  # 表明模型是否使用解码器


class ActorCritic(nn.Module):
    """
    定义了ActorCritic网络模型的类
    """
    is_recurrent = False  # 标记模型是否为循环网络

    def __init__(self, num_command, num_obs, num_privileged_obs, num_obs_history, num_actions_history, num_actions, **kwargs):
        """
        初始化ActorCritic网络模型实例的构造函数

        Args:
        - num_obs: 观测空间的维度
        - num_privileged_obs: 特权观测空间的维度
        - num_obs_history: 观测历史的维度
        - num_actions: 动作空间的维度
        - **kwargs: 接收任意额外的关键字参数，但在这里会被忽略
        """
        # 处理意料之外的关键字参数并忽略它们，打印它们的关键字
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))

        # 根据AC_Args类的设置决定是否使用decoder
        self.decoder = AC_Args.use_decoder

        # 调用nn.Module的初始化方法
        super().__init__()

        self.num_obs = num_obs  # 存储观测历史的维度
        self.num_obs_history = num_obs_history  # 存储观测历史的维度
        self.num_acts_history = num_actions_history
        self.num_privileged_obs = num_privileged_obs  # 存储特权观测的维度
        self.num_command = num_command

        activation = get_activation(AC_Args.activation)  # 获取激活函数实例

        # 构建适应性模块的网络层序列
        adaptation_module_layers = []
        # 添加第一个线性层，输入观测历史维度，输出到第一隐藏层
        adaptation_module_layers.append(nn.Linear(self.num_obs_history + self.num_acts_history, AC_Args.adaptation_module_branch_hidden_dims[0]))
        # 添加激活函数层
        adaptation_module_layers.append(activation)
        # 循环添加后续的线性层和激活函数层
        for l in range(len(AC_Args.adaptation_module_branch_hidden_dims)):
            # 若为适应性模块的最后一层，其输出维度应与特权观测维度一致
            if l == len(AC_Args.adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
            else:
                # 若非最后一层，则添加线性层和激活函数层，线性层的输入输出维度按照AC_Args中定义的维度列表来确定
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        # 将构建好的适应性模块层序列打包成一个Sequential模块
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)

        # 创建演员网络的层列表
        teacher_actor_layers = []
        # 添加演员网络的第一层，合并观测历史和特权观测的维度作为输入
        teacher_actor_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history, AC_Args.actor_hidden_dims[0]))
        # 添加激活函数层
        teacher_actor_layers.append(activation)
        # 循环创建接下来的演员网络层
        for l in range(len(AC_Args.actor_hidden_dims)):
            # 如果当前层是最后一层，则其输出维度应为动作空间的维度
            if l == len(AC_Args.actor_hidden_dims) - 1:
                teacher_actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions + 1))
            else:
                # 对于非最后一层，按照AC_Args中定义的隐藏层维度添加线性层和激活函数层
                teacher_actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))
                teacher_actor_layers.append(activation)
        # 将演员网络的层列表封装成Sequential模块
        self.teacher_actor_body = nn.Sequential(*teacher_actor_layers)

        # 创建演员网络的层列表
        student_actor_layers = []
        # 添加演员网络的第一层，合并观测历史和特权观测的维度作为输入
        student_actor_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history, AC_Args.actor_hidden_dims[0]))
        # 添加激活函数层
        student_actor_layers.append(activation)
        # 循环创建接下来的演员网络层
        for l in range(len(AC_Args.actor_hidden_dims)):
            # 如果当前层是最后一层，则其输出维度应为动作空间的维度
            if l == len(AC_Args.actor_hidden_dims) - 1:
                student_actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions + 1))
            else:
                # 对于非最后一层，按照AC_Args中定义的隐藏层维度添加线性层和激活函数层
                student_actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))
                student_actor_layers.append(activation)
        self.student_actor_body = nn.Sequential(*student_actor_layers)

        # 创建评论家网络的层列表
        critic_layers = []
        # 添加评论家网络的第一层，合并观测历史和特权观测的维度作为输入
        critic_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history, AC_Args.critic_hidden_dims[0]))
        # 添加激活函数层
        critic_layers.append(activation)
        # 循环创建接下来的评论家网络层
        for l in range(len(AC_Args.critic_hidden_dims)):
            # 如果当前层是最后一层，则其输出一个值，代表状态值（value function）
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
            else:
                # 对于非最后一层，按照AC_Args中定义的隐藏层维度添加线性层和激活函数层
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        # 将评论家网络的层列表封装成Sequential模块
        self.critic_body = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.teacher_actor_body}")  # 打印演员（actor）多层感知器的网络结构
        print(f"Critic MLP: {self.critic_body}")  # 打印评论家（critic）多层感知器的网络结构

        # 初始化并设置标准差参数 nn.Parameter，为网络输出动作的随机性提供基础
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        # 关闭PyTorch分布的默认参数检查，可以提高性能但需要确保参数合法性
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        """
        初始化Sequential模块中每个线性层的权重，使用正交初始化方法

        Args:
        - sequential: nn.Sequential对象，包含了多个神经网络层的序列
        - scales: 正交初始化的比例因子列表，每个线性层一个比例因子

        注解:
        1. 这个静态方法不需要实例化类，可以直接通过类调用。
        2. 遍历序列中的每个模块，检查它是否是线性层(nn.Linear)。
        3. 如果是线性层，使用torch.nn.init.orthogonal_方法对权重进行正交初始化。
        4. 初始化时使用对应线性层的scales列表中的比例因子。

        注意:
        - scales列表的长度应与sequential中线性层的数量相匹配。
        """
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
        enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]  # 对于sequential中的每个nn.Linear实例，使用正交初始化其权重

    def reset(self, dones=None):
        """
        重置环境状态

        Args:
        - dones: 一个指示哪些环境需要被重置的布尔数组或者None
        """
        pass  # 通常此处会包含重置环境状态的代码，但具体实现将依赖于具体的环境和用途

    def forward(self):
        """
        定义模型的向前传播逻辑
        """
        raise NotImplementedError  # 子类必须重写此方法来定义forward的具体行为

    @property
    def action_mean(self):
        """
        动作分布均值的属性

        Returns:
        - 动作分布的均值
        """
        return self.distribution.mean  # 提供访问动作分布均值的便捷方式

    @property
    def action_std(self):
        """
        动作分布标准差的属性

        Returns:
        - 动作分布的标准差
        """
        return self.distribution.stddev  # 提供访问动作分布标准差的便捷方式

    @property
    def entropy(self):
        """
        定义分布熵的属性

        Returns:
        - 分布的熵值
        """
        return self.distribution.entropy().sum(dim=-1)  # 获取整个分布熵的总和

    def get_actions_log_prob(self, actions):
        """
        计算行动的对数概率

        Args:
        - actions: 行动集合
        """
        return self.distribution.log_prob(actions).sum(dim=-1)  # 计算所有行动的对数概率之和

    def act_student(self, observation, privileged_info, **kwargs):
        """
        学生策略下执行行动
        """
        # 通过适应模块处理观测历史以得到潜在状态
        a = self.student_actor_body(torch.cat((observation, privileged_info), dim=-1))[:self.num_command]
        mean = a[:, self.num_command:]
        command = a[:, :self.num_command]
        self.distribution = Normal(mean, mean * 0. + self.std)  # 根据新的均值和现有的标准差创建新的动作分布
        return self.distribution.sample(), command  # 从动作分布中采样得到一个行动

    def act_teacher(self, observation, privileged_info, **kwargs):
        """
        教师策略下执行行动
        """
        u = self.teacher_actor_body(torch.cat((observation, privileged_info), dim=-1))
        mean = u[:, 1:]
        command = abs(torch.tanh(u[:, :1]))
        self.distribution = Normal(mean, mean * 0. + self.std)  # 根据新的均值和现有的标准差创建新的动作分布
        actions = self.distribution.sample()
        # actions = torch.tanh(actions)
        return actions, command  # 从动作分布中采样得到一个行动

    def evaluate(self, observation, privileged_observations, **kwargs):
        """
        评估函数
        """
        value = self.critic_body(torch.cat((observation, privileged_observations), dim=-1))  # 评估得到的值
        return value  # 返回评估值

    def get_student_latent(self, obs_history, acts_history):
        """
        获取学生策略的潜在状态
        """
        latent = self.adaptation_module(torch.cat((obs_history, acts_history), dim=-1))
        return latent  # 返回根据观测值历史记录得到的潜在状态


def get_activation(act_name):
    """
    根据激活函数的名称获取相应的激活函数对象

    Args:
    - act_name: 激活函数的名字，如"relu", "tanh"等
    """
    if act_name == "elu":
        return nn.ELU()  # 返回ELU激活函数对象
    elif act_name == "selu":
        return nn.SELU()  # 返回SELU激活函数对象
    elif act_name == "relu":
        return nn.ReLU()  # 返回ReLU激活函数对象
    elif act_name == "crelu":
        return nn.ReLU()  # 返回ReLU激活函数对象（crelu通常是concatenated ReLU，这里使用标准ReLU的行为）
    elif act_name == "lrelu":
        return nn.LeakyReLU()  # 返回LeakyReLU激活函数对象
    elif act_name == "tanh":
        return nn.Tanh()  # 返回Tanh激活函数对象
    elif act_name == "sigmoid":
        return nn.Sigmoid()  # 返回Sigmoid激活函数对象
    else:
        print("invalid activation function!")  # 打印无效激活函数信息
        return None  # 无效名称时返回None
