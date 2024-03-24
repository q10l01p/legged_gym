import numpy as np
import torch
from matplotlib import pyplot as plt


def is_met(scale, l2_err, threshold):
    """
    判断函数

    Args:
    - scale: 规模化调整值
    - l2_err: l2误差值
    - threshold: 设定的阈值

    Returns:
    - 判断结果：如果 l2_err 和规模化调整值的比值小于设定阈值，则返回True，否则返回False

    注解:
    1. 如果 l2_err 等于规模化调整值时，返回值为1。
    2. 如果l2_err小于规模化调整值，返回值小于1，否则返回值大于1。
    """
    return (l2_err / scale) < threshold  # 判断 l2_err 和规模化调整值的比值是否小于设定阈值


def key_is_met(metric_cache, config, ep_len, target_key, env_id, threshold):
    """
    判断函数

    Args:
    - metric_cache: 嵌套字典
    - config: 配置参数
    - ep_len: 某个参数的长度
    - target_key: 需要查找字典的键值
    - env_id: 需要查找字典的键值
    - threshold: 设定的阈值

    Returns:
    - 判断结果：如果 l2_err 和规模化调整值的比值小于设定阈值，则返回True，否则返回False

    注解:
    1. 设置规模化调整值 scale 为 1。
    2. 设置l2误差值（l2_err）为0。
    3. 调用is_met函数比较l2_err和规模化调整值的比值是否小于设定阈值。
    """
    scale = 1  # 设置规模化调整值 scale 为 1
    l2_err = 0  # 设置l2误差值（l2_err）为0
    return is_met(scale, l2_err, threshold)  # 调用is_met函数比较l2_err和规模化调整值的比值是否小于设定阈值


class Curriculum:
    """
    Curriculum类

    注解:
    1. 该类用于处理一系列的键值对，每个键值对对应一个等距的向量和一个线性递增的向量。
    2. 提供了设置指定区域值的方法。
    3. 在初始化时，会生成配置网络和索引网络，并保存所有的键，方便后续操作。

    注意事项:
    - 每个键值对的值是一个三元组，表示一个范围和一个步长。
    - 权重初始化为0，索引初始化为从0到l-1的数组。
    """

    def set_to(self, low, high, value=1.0):
        """
        设置一个指定区域的值

        Args:
        - low: 区域的下限
        - high: 区域的上限
        - value: 设置的值，默认为1.0

        注解:
        1. inds是一个boolean型数组，其值由两个条件combined by logical_and决定。
        2. 在满足条件的索引处设置weights的值为value。
        """
        inds = np.logical_and(  # inds是一个boolean型数组，其值由两个条件combined by logical_and决定
            self.grid >= low[:, None],  # grid中大于等于low的项为True
            self.grid <= high[:, None]  # grid中小于等于high的项为True
        ).all(axis=0)  # 如果全部满足前面两个条件，则为True；否则为False

        assert len(inds) != 0, "You are intializing your distribution with an empty domain!"

        self.weights[inds] = value  # 在满足条件的索引处设置weights的值为value

    def __init__(self, seed, **key_ranges):  # 类初始化
        """
        类初始化

        Args:
        - seed: 随机种子
        - key_ranges: 一系列的键值对，每个键值对的值是一个三元组，表示一个范围和一个步长

        注解:
        1. 创建随机状态，以保证每次生成的随机数一致。
        2. 创建cfg字典，每个key对应等距的向量。
        3. 创建indices字典，每个key对应线性递增的向量。
        4. 处理边界情况。
        5. 计算每个键对应的bin大小。
        6. 生成配置网络和索引网络。
        7. 将所有key保存起来，方便后续操作。
        8. 重新构造配置网络和索引网络的形状。
        9. 获取grid的第一维长度。
        10. 获取每个key对应向量的长度。
        11. 初始化权重。
        12. 生成从0到l-1的数组。
        """
        self.rng = np.random.RandomState(seed)  # 创建随机状态，以保证每次生成的随机数一致

        self.cfg = cfg = {}  # 创建cfg字典，每个key对应等距的向量
        self.indices = indices = {}  # 创建indices字典，每个key对应线性递增的向量
        for key, v_range in key_ranges.items():  # 遍历传入的键值对
            bin_size = (v_range[1] - v_range[0]) / v_range[2]  # compute bin size for each key
            cfg[key] = np.linspace(v_range[0] + bin_size / 2, v_range[1] - bin_size / 2, v_range[2])  # 生成等距向量
            indices[key] = np.linspace(0, v_range[2] - 1, v_range[2])  # 生成线性递增向量

        self.lows = np.array([range[0] for range in key_ranges.values()])  # 处理边界情况
        self.highs = np.array([range[1] for range in key_ranges.values()])  # 处理边界情况

        # calcualte bin_sizes
        self.bin_sizes = {key: (v_range[1] - v_range[0]) / v_range[2] for key, v_range in key_ranges.items()}

        self._raw_grid = np.stack(np.meshgrid(*cfg.values(), indexing='ij'))  # 生成配置网络
        self._idx_grid = np.stack(np.meshgrid(*indices.values(), indexing='ij'))  # 生成索引网络
        self.keys = [*key_ranges.keys()]  # 将所有key保存起来，方便后续操作
        self.grid = self._raw_grid.reshape([len(self.keys), -1])  # 重新构造配置网络的形状
        self.idx_grid = self._idx_grid.reshape([len(self.keys), -1])  # 重新构造索引网络的形状

        self._l = l = len(self.grid[0])  # 获取grid的第一维长度
        self.ls = {key: len(self.cfg[key]) for key in self.cfg.keys()}  # 获取每个key对应向量的长度

        self.weights = np.zeros(l)  # 初始化权重
        self.indices = np.arange(l)  # 生成从0到l-1的数组

    def __len__(self):
        """
        返回该类（Curriculum）的长度

        Returns:
        - grid的第一维长度

        注解:
        1. 该方法用于获取Curriculum类的长度，即grid的第一维长度。
        """
        return self._l

    def __getitem__(self, *keys):
        """
        重写类（Curriculum）的getitem方法

        Args:
        - keys: 一个或多个键

        注解:
        1. 该方法用于返回一对或多对key的值。
        2. 该方法尚未实现。
        """
        pass

    def update(self, **kwargs):
        """
        更新方法

        Args:
        - kwargs: 一系列的键值对

        注解:
        1. 该方法用于在满足某些条件后更新细节。
        2. 该方法尚未实现。
        """
        pass

    def sample_bins(self, batch_size, low=None, high=None):
        """
        用于选择指定数量的样本

        Args:
        - batch_size: 要抽取样本的数量
        - low: 抽样的下界
        - high: 抽样的上界

        Returns:
        - 抽样结果以及对应的indices

        注解:
        1. 如果指定了抽样界限，根据参数low和high计算满足条件的索引，并在满足条件的索引处设置temp_weights的值为weights的对应项。
        2. 随机从indices抽取batch_size个样本，同时满足temp_weights/sum(temp_weights)的概率分布。
        3. 如果没有指定界限，随机从indices抽取batch_size个样本，同时满足weights / sum(weights)的概率分布。
        """
        if low is not None and high is not None:  # 如果指定了抽样界限
            valid_inds = np.logical_and(  # 根据参数low和high计算满足条件的索引
                self.grid >= low[:, None],
                self.grid <= high[:, None]
            ).all(axis=0)
            temp_weights = np.zeros_like(self.weights)  # 初始化和weights形状相同的零向量
            temp_weights[valid_inds] = self.weights[valid_inds]  # 在满足条件的索引处设置temp_weights的值为weights的对应项
            # 随机从indices抽取batch_size个样本，同时满足temp_weights/sum(temp_weights)的概率分布
            inds = self.rng.choice(self.indices, batch_size, p=temp_weights / temp_weights.sum())
        else:  # 如果没有指定界限
            # 随机从indices抽取batch_size个样本，同时满足weights / sum(weights)的概率分布
            inds = self.rng.choice(self.indices, batch_size, p=self.weights / self.weights.sum())

        return self.grid.T[inds], inds  # 返回抽样结果以及对应的indices

    def sample_uniform_from_cell(self, centroids):
        """
        从指定的区间内（cell）均匀采样一组数

        Args:
        - centroids: 中心值

        Returns:
        - 从low和high值形成的区间进行均匀采样的结果

        注解:
        1. 获取每个bin的大小。
        2. 对每个中心值进行调整获取low和high值。
        3. 从low和high值形成的区间进行均匀采样。
        4. 可能需要确保采样结果在self.lows 和 self.highs间。
        """
        bin_sizes = np.array([*self.bin_sizes.values()])  # 获取每个bin的大小
        low, high = centroids - bin_sizes / 2, centroids + bin_sizes / 2  # 对每个中心值进行调整获取low和high值
        return self.rng.uniform(low, high)  # 从low和high值形成的区间进行均匀采样

    def sample(self, batch_size, low=None, high=None):
        """
        采样函数

        Args:
        - batch_size: 批次大小
        - low: 采样的最小边界
        - high: 采样的最大边界

        Returns:
        - 采样结果和索引

        注解:
        1. 从bins中采样获取中心值和索引。
        2. 对每个中心值进行均匀采样，结果和索引一起返回。
        """
        cgf_centroid, inds = self.sample_bins(batch_size, low=low, high=high)  # 从bins中采样获取中心值和索引
        # 对每个中心值进行均匀采样，结果和索引一起返回
        return np.stack([self.sample_uniform_from_cell(v_range) for v_range in cgf_centroid]), inds


# SumCurriculum类继承于Curriculum类，并添加了一些额外的字段和方法
class SumCurriculum(Curriculum):
    # 类的初始化函数，除了父类Curriculum的初始化操作外，还初始化了自身的新成员变量
    def __init__(self, seed, **kwargs):
        super().__init__(seed, **kwargs)  # 调用父类的初始化函数

        self.success = np.zeros(len(self))  # 初始化一个记录成功次数的向量，长度等于self的长度
        self.trials = np.zeros(len(self))  # 初始化一个记录尝试次数的向量，长度等于self的长度

    # update函数，用于根据新的bin_inds和l1_error来更新self.success和self.trials
    def update(self, bin_inds, l1_error, threshold):
        is_success = l1_error < threshold  # 计算是否成功，即l1_error是否低于阈值
        self.success[bin_inds[is_success]] += 1  # 在成功的bin_inds处，self.success对应位置加1
        self.trials[bin_inds] += 1  # 在所有尝试的bin_inds处，self.trials对应位置加1

    # 成功率计算方法，根据self.success和self.trials来计算成功率，并根据*keys来计算marginal成功率
    def success_rates(self, *keys):
        s_rate = self.success / (self.trials + 1e-6)  # 计算成功率，避免除0错误
        s_rate = s_rate.reshape(list(self.ls.values()))  # 将成功率向量reshape为list(self.ls.values())的形状
        marginals = tuple(i for i, key in enumerate(self.keys) if key not in keys)  # 计算marginal的indices
        if marginals:
            return s_rate.mean(axis=marginals)  # 如果marginals不为空，则返回marginal成功率
        return s_rate  # 否则，返回全部的成功率


class RewardThresholdCurriculum(Curriculum):
    """
    RewardThresholdCurriculum类

    注解:
    1. 该类继承于Curriculum类，并添加了一些额外的字段和方法。
    2. 在初始化时，除了父类Curriculum的初始化操作外，还初始化了自身的新成员变量。
    3. 提供了get_local_bins函数，返回的是在给定的bin_inds周围的bins。周围的区域由ranges参数决定。
    4. 提供了update函数，用于更新权重，它在每一次学习任务执行完毕后被调用。
    5. 提供了log方法，用于记录每一次试验的线性速度、角速度和试验持续时间。

    注意事项:
    - self.episode_reward_lin, self.episode_reward_ang, self.episode_lin_vel_raw, self.episode_ang_vel_raw, self.episode_duration都是长度等于self的长度的向量。
    - 在update函数中，如果任务奖励大于成功阈值，就认为任务成功。如果没有任何成功阈值，那么就认为任务没有成功。
    - 在log方法中，记录的数据需要转换为numpy数组。
    """

    def __init__(self, seed, **kwargs):
        """
        类的初始化函数

        Args:
        - seed: 随机种子
        - kwargs: 一系列的键值对

        注解:
        1. 调用父类的初始化函数。
        2. 初始化一个记录线性奖励的向量，长度等于self的长度。
        3. 初始化一个记录角速度奖励的向量，长度等于self的长度。
        4. 初始化一个记录原始线性速度的向量，长度等于self的长度。
        5. 初始化一个记录原始角速度的向量，长度等于self的长度。
        6. 初始化一个记录每一次试验持续时间的向量，长度等于self的长度。
        """
        super().__init__(seed, **kwargs)  # 调用父类的初始化函数

        self.episode_reward_lin = np.zeros(len(self))  # 初始化一个记录线性奖励的向量，长度等于self的长度
        self.episode_reward_ang = np.zeros(len(self))  # 初始化一个记录角速度奖励的向量，长度等于self的长度
        self.episode_lin_vel_raw = np.zeros(len(self))  # 初始化一个记录原始线性速度的向量，长度等于self的长度
        self.episode_ang_vel_raw = np.zeros(len(self))  # 初始化一个记录原始角速度的向量，长度等于self的长度
        self.episode_duration = np.zeros(len(self))  # 初始化一个记录每一次试验持续时间的向量，长度等于self的长度

    def get_local_bins(self, bin_inds, ranges=0.1):
        """
        返回在给定的bin_inds周围的bins

        Args:
        - bin_inds: bin的索引
        - ranges: 周围的区域

        Returns:
        - 布尔数组，元素为True意味着对应的项目在bin_inds参数指定的bins附近。

        注解:
        1. 检查ranges参数的类型，如果是float，则变为长度等于self.grid形状的一个数组。
        2. 将bin_inds参数reshape为一维数组。
        3. 构造一个布尔数组，元素为True意味着对应的项目在bin_inds参数指定的bins附近。
        """
        if isinstance(ranges, float):  # 检查ranges参数的类型，如果是float，则变为长度等于self.grid形状的一个数组
            ranges = np.ones(self.grid.shape[0]) * ranges
        bin_inds = bin_inds.reshape(-1)  # 将bin_inds参数reshape为一维数组。

        # 下面这段代码构造一个布尔数组，元素为True意味着对应的项目在bin_inds参数指定的bins附近。
        adjacent_inds = np.logical_and(
            self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) >= self.grid[:, bin_inds, None] - ranges.reshape(-1, 1, 1),
            self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) <= self.grid[:, bin_inds, None] + ranges.reshape(-1, 1, 1)
        ).all(axis=0)

        return adjacent_inds

    def update(self, bin_inds, task_rewards, success_thresholds, local_range=0.5):
        """
        更新权重

        Args:
        - bin_inds: bin的索引
        - task_rewards: 任务奖励
        - success_thresholds: 成功阈值
        - local_range: 局部范围

        注解:
        1. 初始成功标志为1，即假设任务成功完成。
        2. 遍历每一个任务奖励和成功阈值，如果任务奖励大于成功阈值，就认为任务成功。
        3. 如果没有任何成功阈值，那么就认为任务没有成功。否则，将is_success转换为布尔数组。
        4. 如果有任务成功，打印"successes"。
        5. 找到成功的bins索引，增加它们的权重，并将权重限制在0和1之间。
        6. 对于每一个成功的bins，找到它们附近的bins。提升邻近bins的权重，且在0和1之间。
        """
        # 初始成功标志为1，即假设任务成功完成。
        is_success = 1.
        # 遍历每一个任务奖励和成功阈值，如果任务奖励大于成功阈值，就认为任务成功
        for task_reward, success_threshold in zip(task_rewards, success_thresholds):
            is_success = is_success * (task_reward > success_threshold).cpu()
        # 如果没有任何成功阈值，那么就认为任务没有成功。
        if len(success_thresholds) == 0:
            is_success = np.array([False] * len(bin_inds))
        # 否则，将is_success转换为布尔数组
        else:
            is_success = np.array(is_success.bool())

        # 如果有任务成功，打印"successes"
        # if len(is_success) > 0 and is_success.any():
        #     print("successes")

        # 找到成功的bins索引，增加它们的权重，并将权重限制在0和1之间。
        self.weights[bin_inds[is_success]] = np.clip(self.weights[bin_inds[is_success]] + 0.2, 0, 1)
        # 对于每一个成功的bins，找到它们附近的bins。
        adjacents = self.get_local_bins(bin_inds[is_success], ranges=local_range)
        for adjacent in adjacents:
            # print(adjacent)  # 打印邻近的bin索引
            # print(self.grid[:, adjacent])  # 打印邻近的bins的网格坐标
            adjacent_inds = np.array(adjacent.nonzero()[0])
            self.weights[adjacent_inds] = np.clip(self.weights[adjacent_inds] + 0.2, 0, 1)  # 提升邻近bins的权重，且在0和1之间

    def log(self, bin_inds, lin_vel_raw=None, ang_vel_raw=None, episode_duration=None):
        """
        记录每一次试验的线性速度、角速度和试验持续时间

        Args:
        - bin_inds: bin的索引
        - lin_vel_raw: 线性速度原始数据
        - ang_vel_raw: 角速度原始数据
        - episode_duration: 每一个试验的持续时间

        注解:
        1. 记录线性速度原始数据。
        2. 记录角速度原始数据。
        3. 记录每一个试验的持续时间。
        """
        self.episode_lin_vel_raw[bin_inds] = lin_vel_raw.cpu().numpy()  # 记录线性速度原始数据
        self.episode_ang_vel_raw[bin_inds] = ang_vel_raw.cpu().numpy()  # 记录角速度原始数据
        self.episode_duration[bin_inds] = episode_duration.cpu().numpy()  # 记录每一个试验的持续时间


if __name__ == '__main__':
    """
    当脚本作为主程序运行时，执行下列代码

    注解:
    1. 初始化RewardThresholdCurriculum类，并设置初始参数。
    2. 通过断言，保证r的_raw_grid形状是(3, 5, 2, 11)，否则打印错误信息。
    3. 设置low和high的值。
    4. 获取邻近的bins。
    5. 对于每一个邻近的bins，打印邻近bins的索引，然后更新对应的向量和奖励值。
    6. 获取样本和bins，并绘制散点图。

    注意事项:
    - 在update函数中，lin_vel_rewards和ang_vel_rewards都是全1的向量，lin_vel_threshold和ang_vel_threshold都是0，local_range是0.5。
    - 在绘制散点图时，只绘制了样本的前两个维度。
    """
    # 初始化RewardThresholdCurriculum类，并设置初始参数
    r = RewardThresholdCurriculum(100, x=(-1, 1, 5), y=(-1, 1, 2), z=(-1, 1, 11))

    # 通过断言，保证r的_raw_grid形状是(3, 5, 2, 11)，否则打印错误信息
    assert r._raw_grid.shape == (3, 5, 2, 11), "grid shape is wrong: {}".format(r.grid.shape)

    low, high = np.array([-1.0, -0.6, -1.0]), np.array([1.0, 0.6, 1.0])  # 设置low和high的值

    adjacents = r.get_local_bins(np.array([10, 1]), ranges=0.5)  # 获取邻近的bins

    for adjacent in adjacents:
        adjacent_inds = np.array(adjacent.nonzero()[0])  # 打印邻近bins的索引
        print(adjacent_inds)
        r.update(bin_inds=adjacent_inds, task_rewards=np.ones_like(adjacent_inds),
                 ang_vel_rewards=np.ones_like(adjacent_inds), lin_vel_threshold=0.0, ang_vel_threshold=0.0,
                 local_ranges=0.5)

    samples, bins = r.sample(10_000)  # 获取样本和bins

    plt.scatter(*samples.T[:2])  # 绘制散点图
    plt.show()
