import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, dt):
        """
        初始化日志记录器对象

        Args:
        - dt: 时间间隔，用于记录状态和奖励信息

        属性:
        - state_log: 默认字典，用来记录每个时间步的状态信息。键是时间步，值是状态值的列表。
        - rew_log: 默认字典，用来记录每个时间步的奖励信息。键是时间步，值是奖励值的列表。
        - dt: 记录时间间隔，每隔一定时间就记录一次信息。
        - num_episodes: 追踪已记录的总剧情（episodes）数。
        - plot_process: 用于绘图的进程，如果实现了绘图功能的话。

        注解:
        1. defaultdict(list): 初始化defaultdict，当访问一个不存在的键时，会自动创建一个新键并将其值设置为一个空列表。
        2. self.num_episodes = 0: 初始化时，已完成的剧情数设置为0。
        3. self.plot_process = None: 如果在类的其他部分有代码实现了绘图功能，plot_process可能会被用来存放绘图进程。

        注意事项:
        - 这个Logger类可以随着时间的推移记录状态和奖励，但不包含实际的状态和奖励记录方法。这可能需要在其他地方实现，例如在模拟环境中。
        """
        self.state_log = defaultdict(list)  # 默认字典，记录每个时间步的状态信息
        self.rew_log = defaultdict(list)  # 默认字典，记录每个时间步的奖励信息
        self.dt = dt  # 时间间隔
        self.num_episodes = 0  # 已记录的总剧情（episodes）数
        self.plot_process = None  # 用于绘图的进程（如果有）

    def log_state(self, key, value):
        """
        记录单个状态到日志

        Args:
        - key: 状态的键，通常是时间步或者其他标识符
        - value: 与键相对应的状态值

        注解:
        1. self.state_log[key].append(value): 在状态日志的特定键下添加一个新的状态值。
        
        注意事项:
        - 如果`key`还没有在`self.state_log`中，defaultdict会自动创建这个键并初始化为一个空列表。
        """
        self.state_log[key].append(value)  # 将状态值添加到对应键下的列表中

    def log_states(self, dict):
        """
        批量记录多个状态到日志

        Args:
        - dict: 包含多个要记录的状态的字典，键是状态的键，值是状态值

        注解:
        1. for key, value in dict.items(): 遍历字典中的所有键值对。
        2. self.log_state(key, value): 调用log_state方法记录每个键值对到状态日志中。

        注意事项:
        - 该方法是log_state的批量版本，方便同时记录多个状态。
        """
        for key, value in dict.items():  # 遍历字典中的所有键值对
            self.log_state(key, value)  # 调用log_state方法记录每个状态

    def log_rewards(self, dict, num_episodes):
        """
        批量记录多个奖励到日志，并更新剧情数

        Args:
        - dict: 包含需要记录的奖励的字典，键通常包含'rew'字样
        - num_episodes: 新完成的剧情（episodes）数量

        注解:
        1. for key, value in dict.items(): 遍历字典中的所有键值对。
        2. if 'rew' in key: 筛选出键名中含有'rew'的键值对。
        3. self.rew_log[key].append(value.item() * num_episodes): 将奖励值转换为浮点数并乘以完成的剧情数后记录到日志中。
        4. self.num_episodes += num_episodes: 更新记录的总剧情数。

        注意事项:
        - 保证传入的奖励值包含`item`方法，通常意味着奖励值应该是Tensor类型。
        - 记录的奖励值会被乘以剧情数，因此可能需要根据使用场景调整这一逻辑。
        """
        for key, value in dict.items():  # 遍历字典中的所有键值对
            if 'rew' in key:  # 筛选含有'rew'的键
                self.rew_log[key].append(value.item() * num_episodes)  # 记录乘以剧情数的奖励值
        self.num_episodes += num_episodes  # 更新总剧情数

    def reset(self):
        """
        重置状态日志和奖励日志

        注解:
        1. self.state_log.clear(): 清除状态日志中的所有记录。
        2. self.rew_log.clear(): 清除奖励日志中的所有记录。

        注意事项:
        - 该方法会移除所有的记录，不可恢复。在使用前确保不再需要日志中的信息。
        """
        self.state_log.clear()  # 清除状态日志
        self.rew_log.clear()  # 清除奖励日志

    def plot_states(self):
        """
        在独立进程中启动状态绘图

        注解:
        1. self.plot_process = Process(target=self._plot): 创建一个Process对象，并设置目标函数为类中定义的_plot方法。
        2. self.plot_process.start(): 启动进程，实际调用_plot方法进行绘图。

        注意事项:
        - 确保在类中定义了_plot方法，该方法包含绘制状态日志的逻辑。
        - 使用多进程可能会导致资源共享和同步的问题，确保进程间通信和数据共享得当。
        """
        self.plot_process = Process(target=self._plot)  # 创建绘图进程，并设置目标函数
        self.plot_process.start()  # 启动绘图进程

    def _plot(self):
        """
        绘制状态日志中记录的不同变量的时间序列图
        """

        # 设置子图的行数和列数
        nb_rows = 3
        nb_cols = 3
        
        # 创建一个3x3的子图网格
        fig, axs = plt.subplots(nb_rows, nb_cols)
        
        # 从状态日志中获取时间数据
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break  # 只需要从任意一个日志条目中提取时间序列
        
        # 将状态日志赋值给一个本地变量以简化代码
        log = self.state_log

        # 绘制关节的位置测量值和目标值
        a = axs[1, 0]
        if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
        if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()

        # 绘制关节的速度测量值和目标值
        a = axs[1, 1]
        if log["dof_vel"]: a.plot(time, log["dof_vel"], label='measured')
        if log["dof_vel_target"]: a.plot(time, log["dof_vel_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()

        # 绘制基座沿x轴的线速度的测量值和指令值
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()

        # 绘制基座沿y轴的线速度的测量值和指令值
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()

        # 绘制基座的偏航角速度的测量值和指令值
        a = axs[0, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()

        # 绘制基座沿z轴的线速度测量值
        a = axs[1, 2]
        if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        a.legend()

        # 绘制垂直接触力测量值
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()

        # 绘制关节速度和关节扭矩的测量值
        a = axs[2, 1]
        if log["dof_vel"] != [] and log["dof_torque"] != []: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        a.legend()

        # 绘制关节扭矩的测量值
        a = axs[2, 2]
        if log["dof_torque"] != []: a.plot(time, log["dof_torque"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()

        # 展示绘图
        plt.show()

    def print_rewards(self):
        """
        打印每秒平均奖励的信息

        注解:
        - 此方法主要是用来打印出每个回合的平均奖励以及总回合数。
        - 它依次处理奖励日志中的每个关键值，并计算相应的平均奖励。
        - 最后打印出所记录的总回合数。

        注意:
        - 假设self.rew_log是一个字典，它存储了每个关键值下的奖励数组。
        - 假设self.num_episodes是一个整数，代表了总的回合数。
        - 这里使用了numpy的sum和array方法来处理数值运算。
        """

        # 打印平均每秒奖励
        print("Average rewards per second:")
        for key, values in self.rew_log.items():  # 遍历奖励日志的每个项
            mean = np.sum(np.array(values)) / self.num_episodes  # 计算平均奖励
            print(f" - {key}: {mean}")  # 打印每个关键值的平均奖励
        print(f"Total number of episodes: {self.num_episodes}")  # 打印总回合数
    
    def __del__(self):
        """
        析构函数，在对象被销毁前调用

        注解:
        - 此方法用于确保关联的绘图进程在对象被销毁前被终止。
        - 当Python解释器要销毁对象并回收内存时，会自动调用此方法。

        注意:
        - 此方法中的self.plot_process应该是一个multiprocessing.Process对象或类似的容器进程对象。
        - kill()方法将会安全地终止绘图进程，避免产生僵尸进程或未完成的作业。
        """

        if self.plot_process is not None:  # 检查绘图进程是否存在
            self.plot_process.kill()  # 如果存在，则终止这个进程
