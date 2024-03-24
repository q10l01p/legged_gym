import numpy as np


class RunningMeanStd:
    """
    动态计算数据的均值和标准差

    Attributes:
    - n: 计数器，记录更新次数
    - mean: 当前的均值
    - S: 用于计算标准差的中间变量
    - std: 当前的标准差

    方法:
    - __init__: 类的初始化方法
    - update: 更新均值和标准差
    """

    def __init__(self, shape):  # 初始化方法
        """
        初始化RunningMeanStd对象

        Args:
        - shape: 输入数据的维度
        """
        self.n = 0  # 初始化计数器
        self.mean = np.zeros(shape)  # 初始化均值为0
        self.S = np.zeros(shape)  # 初始化S为0
        self.std = np.sqrt(self.S)  # 初始化标准差

    def update(self, x):  # 更新均值和标准差
        """
        根据新的输入数据x更新均值和标准差

        Args:
        - x: 新的输入数据
        """
        x = np.array(x)  # 确保x是numpy数组
        self.n += 1  # 更新次数加1
        if self.n == 1:  # 如果是第一次更新
            self.mean = x  # 直接将x设为均值
            self.std = x  # 直接将x设为标准差
        else:  # 如果不是第一次更新
            old_mean = self.mean.copy()  # 复制旧的均值
            self.mean = old_mean + (x - old_mean) / self.n  # 计算新的均值
            self.S = self.S + (x - old_mean) * (x - self.mean)  # 更新S
            self.std = np.sqrt(self.S / self.n)  # 计算新的标准差


class Normalization:
    """
    对观测数据进行标准化处理的类

    Attributes:
    - running_ms: RunningMeanStd实例，用于计算和更新均值和标准差

    方法:
    - __init__: 类的初始化方法
    - __call__: 使得Normalization实例可以像函数一样被调用，用于标准化数据
    """

    def __init__(self, shape):  # 初始化方法
        """
        初始化Normalization对象

        Args:
        - shape: 输入数据的维度
        """
        self.running_ms = RunningMeanStd(shape=shape)  # 创建RunningMeanStd实例

    def __call__(self, obs, update=True):  # 类的调用方法
        """
        对输入的观测数据obs进行标准化处理

        Args:
        - obs: 输入的观测数据，可以是字典类型，包含'observation'和'desired_goal'，或者是数组类型
        - update: 布尔值，表示是否更新均值和标准差，默认为True

        Returns:
        - 标准化后的观测数据
        """
        # 判断obs是否为字典类型，如果是，则将'observation'和'desired_goal'合并为一个数组
        if isinstance(obs, dict):
            x = np.concatenate([obs['observation'], obs['desired_goal']], axis=0)
        else:
            x = obs  # 如果不是字典类型，直接使用obs作为输入数据

        # 如果update为True，则更新均值和标准差
        if update:
            self.running_ms.update(x)

        # 使用当前的均值和标准差对数据进行标准化处理，1e-8是为了防止除以0
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        # 如果obs是字典类型，将标准化后的数据重新分配回'observation'和'desired_goal'
        if isinstance(obs, dict):
            obs['observation'] = x[:obs['observation'].shape[0]]
            obs['desired_goal'] = x[obs['observation'].shape[0]:]
            return obs  # 返回标准化后的字典类型数据
        return x  # 如果obs不是字典类型，直接返回标准化后的数组


class RewardScaling:
    """
    对奖励进行缩放处理的类

    Attributes:
    - shape: 奖励数据的维度
    - gamma: 折扣因子
    - running_ms: RunningMeanStd实例，用于计算和更新奖励的标准差
    - R: 累积奖励

    方法:
    - __init__: 类的初始化方法
    - __call__: 使得RewardScaling实例可以像函数一样被调用，用于缩放奖励
    - reset: 重置累积奖励
    """

    def __init__(self, shape, gamma):  # 初始化方法
        """
        初始化RewardScaling对象

        Args:
        - shape: 奖励数据的维度，通常为1
        - gamma: 折扣因子，用于计算累积奖励
        """
        self.shape = shape  # 设置奖励数据的维度
        self.gamma = gamma  # 设置折扣因子
        self.running_ms = RunningMeanStd(shape=self.shape)  # 创建RunningMeanStd实例
        self.R = np.zeros(self.shape)  # 初始化累积奖励为0

    def __call__(self, x):  # 类的调用方法
        """
        对输入的奖励x进行缩放处理

        Args:
        - x: 输入的奖励数据

        Returns:
        - 缩放后的奖励数据
        """
        self.R = self.gamma * self.R + x  # 更新累积奖励
        self.running_ms.update(self.R)  # 更新奖励的标准差
        x = x / (self.running_ms.std + 1e-8)  # 使用标准差对奖励进行缩放，1e-8是为了防止除以0
        return x  # 返回缩放后的奖励数据

    def reset(self):  # 重置方法
        """
        当一个episode结束时，重置累积奖励
        """
        self.R = np.zeros(self.shape)  # 将累积奖励重置为0
