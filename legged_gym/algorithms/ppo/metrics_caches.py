from collections import defaultdict

from ml_logger import logger
import numpy as np
import torch


class DistCache:
    def __init__(self):
        """
        分布式缓存的初始构造函数

        属性:
        - cache: 使用defaultdict创建的缓存，缺省值为0
        """
        self.cache = defaultdict(lambda: 0)

    def log(self, **key_vals):
        """
        记录键-值对到缓存，并更新相应的统计数据

        Args:
        - key_vals: 关键字参数，表示要记录的键-值对

        注解:
        1. 遍历传入的关键字参数中的每一对键值对。
        2. 对于每个键，计算新的计数，即该键之前记录的次数加一。
        3. 在缓存中更新该键的'@counts'后缀对应的值，即新的次数。
        4. 更新该键对应的值，使其变成总和，即新值加上之前的总和。
        5. 计算新的平均值，即总和除以计数。
        6. 存储计算后的新平均值替换旧值。

        注意事项:
        - 所有的键将根据新的数据点更新它们的平均值。
        - key_vals字典中的键名不应包含'@counts'，因为它用于内部计数。
        """
        for k, v in key_vals.items():
            count = self.cache[k + '@counts'] + 1  # 获取被记录次数，并加1
            self.cache[k + '@counts'] = count  # 更新被记录次数
            self.cache[k] = (v + (count - 1) * self.cache[k]) / count  # 更新平均值

    def get_summary(self):
        """
        获取缓存中所有键-值对的摘要，并清除缓存

        Returns:
        - ret: 不带有'@counts'后缀的键-值对的字典

        注释:
        1. 创建一个新字典，用于存储除去'@counts'后缀的键及其对应的值。
        2. 在这个新字典里，我们只包含那些没有'@counts'后缀的键和它们对应的平均值。
        3. 清空现有缓存。
        4. 返回创建的包含平均值的新字典。

        注意事项:
        - 调用此方法后，当前缓存内的所有数据将被清空。
        - 返回的字典中，键-值对的值代表的是各个键对应值的平均数。
        """
        ret = {k: v for k, v in self.cache.items() if not k.endswith("@counts")}  # 创建摘要字典
        self.cache.clear()  # 清除缓存
        return ret  # 返回摘要


if __name__ == '__main__':
    cl = DistCache()  # 创建DistCache对象

    lin_vel = np.ones((11, 11))  # 创建一个11x11的全1数组

    ang_vel = np.zeros((5, 5))  # 创建一个5x5的全0数组

    cl.log(lin_vel=lin_vel, ang_vel=ang_vel)  # 将lin_vel和ang_vel记录到缓存中

    lin_vel = np.zeros((11, 11))  # 创建一个11x11的全0数组

    ang_vel = np.zeros((5, 5))  # 创建一个5x5的全0数组

    cl.log(lin_vel=lin_vel, ang_vel=ang_vel)  # 将lin_vel和ang_vel记录到缓存中

    print(cl.get_summary())  # 打印缓存中的摘要信息


class SlotCache:
    def __init__(self, n):
        """
        初始化SlotCache类

        Args:
        - n: 缓存槽数量

        注解:
        1. 给每个缓存槽位分配一个默认值为0的Numpy数组。
        2. 使用defaultdict来创建一个缓存字典，当键不存在时将使用一个0值数组。

        注意事项:
        - 默认数组的长度与缓存槽位数量相同。
        """
        self.n = n  # 缓存槽数量
        self.cache = defaultdict(lambda: np.zeros([n]))  # 创建字典，未知键对应0值数组

    def log(self, slots=None, **key_vals):
        """
        记录或更新缓存中的值

        Args:
        - slots: 将要更新的槽位的索引列表，默认为None表示更新所有槽位
        - key_vals: 一个字典，包含要更新的键和对应的值

        注解:
        1. 如果没有指定slots，则默认为更新全部槽位（0 到 n-1）。
        2. 遍历所有要更新的键值对。
        3. 对每个键，更新相应槽位的计数器，并计算新的平均值。
        4. 根据更新的计数修改存储的值，计算新的平均值。

        注意事项:
        - 使用'@counts'后缀来区分用于计数的键和用于存储值的键。
        - 对于没有提供的键，缓存值不会被更改。
        """
        if slots is None:
            slots = range(self.n)  # 如果未指定槽位，则更新所有槽位

        for k, v in key_vals.items():
            # 更新计数器并计算新平均值
            counts = self.cache[k + '@counts'][slots] + 1  # 更新计数
            self.cache[k + '@counts'][slots] = counts  # 将新计数保存回缓存
            self.cache[k][slots] = v + (counts - 1) * self.cache[k][slots]  # 累加旧值和新值
            self.cache[k][slots] /= counts  # 计算新平均

    def get_summary(self):
        """
        获取缓存的摘要，并清除所有缓存值

        Returns:
        - ret: 不包含用于计数的键的摘要字典

        注解:
        1. 遍历当前的缓存字典。
        2. 构建一个新字典，仅包含不以'@counts'结尾的键。
        3. 清除所有缓存，以便重新开始。

        注意事项:
        - 返回的摘要包含了所有计算后的平均值。
        - 清除缓存后，所有之前的记录都将丢失。
        """
        # 构建包含平均值的摘要字典，排除计数器
        ret = {
            k: v
            for k, v in self.cache.items()
            if not k.endswith("@counts")
        }
        self.cache.clear()  # 清理缓存字典
        return ret  # 返回摘要


if __name__ == '__main__':
    # 创建一个有100个槽位的SlotCache实例
    cl = SlotCache(100)

    # 定义需要重置的环境ID
    reset_env_ids = [2, 5, 6]

    # 线速度的列表
    lin_vel = [0.1, 0.5, 0.8]

    # 角速度的列表
    ang_vel = [0.4, -0.4, 0.2]

    # 将reset_env_ids中指定的槽位更新为新的线速度和角速度
    cl.log(reset_env_ids, lin_vel=lin_vel, ang_vel=ang_vel)

    # 更新所有槽位的线速度为1, 由于没有指定slots，默认为全部更新
    cl.log(lin_vel=np.ones(100))
