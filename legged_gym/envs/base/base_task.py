import sys

import gym
import torch
from isaacgym import gymapi, gymutil


# RL任务的基础类
class BaseTask(gym.Env):  # 继承了gym库的Env类

    # 构造函数
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, eval_cfg=None):
        """
        模拟环境类

        Args:
        - cfg: 配置信息
        - sim_params: 模拟参数
        - physics_engine: 物理引擎
        - sim_device: 模拟设备
        - headless: 是否为无头模式
        - eval_cfg: 评价配置

        注解:
        1. 获取gym的api。
        2. 判断physics_engine是否为"SIM_PHYSX"字符串，如果是，则将其替换为gymapi库中的对应值。
        3. 初始化模拟参数，物理引擎，模拟设备，无头模式等实例变量。
        4. 解析设备字符串，获取设备类型和设备id。
        5. 判断环境设备，如果模拟设备是GPU且sim_params.use_gpu_pipeline为True，则设备是sim_device，否则是CPU。
        6. 将模拟设备ID初始化为图形设备ID，用于渲染，如果为-1则不进行渲染。
        7. 在无头模式下，也将模拟设备ID设为图形设备ID。
        8. 从配置文件中获取环境的观察数量，特权观察数量，以及行动数量，并分别初始化。
        9. 如果存在评价设置(eval_cfg)，那么初始化训练环境，评价环境及总环境数量。
        10. 设置PyTorch JIT的优化标志，关闭性能分析模式和执行器。
        11. 分配缓冲区，用于存储环境的观察结果，奖励，重设标志，偏正奖励，偏负奖励，每个环境的时间和特权访问。
        12. 为额外数据分配空字典。
        13. 创建环境，模拟和视图。
        14. 初始设置视图同步为启用状态。
        15. 如果以视图方式运行，设置键盘快捷方式和相机。

        注意事项:
        - 模拟环境的创建和初始化依赖于传入的配置信息和模拟参数。
        - 观察结果，奖励，重设标志，偏正奖励，偏负奖励，每个环境的时间和特权访问都存储在缓冲区中。
        - 视图同步的启用和禁用可以通过键盘快捷方式进行切换。
        """
        self.gym = gymapi.acquire_gym()  # 获取gym的api

        if isinstance(physics_engine, str) and physics_engine == "SIM_PHYSX":
            physics_engine = gymapi.SIM_PHYSX  # 将物理引擎字符串替换为对应的gymapi值

        self.sim_params = sim_params  # 初始化模拟参数
        self.physics_engine = physics_engine  # 初始化物理引擎
        self.sim_device = sim_device  # 初始化模拟设备
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)  # 解析设备字符串
        self.headless = headless  # 初始化无头模式标志

        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device  # 使用GPU设备
        else:
            self.device = 'cpu'  # 使用CPU设备

        self.graphics_device_id = self.sim_device_id  # 初始化图形设备ID
        if self.headless == True:
            self.graphics_device_id = self.sim_device_id  # 在无头模式下，图形设备ID设为模拟设备ID

        self.num_obs = cfg.env.num_observations  # 初始化观察数量
        self.num_privileged_obs = cfg.env.num_privileged_obs  # 初始化特权观察数量
        self.num_actions = cfg.env.num_actions  # 初始化行动数量

        if eval_cfg is not None:
            self.num_eval_envs = eval_cfg.env.num_envs  # 初始化评价环境数量
            self.num_train_envs = cfg.env.num_envs  # 初始化训练环境数量
            self.num_envs = self.num_eval_envs + self.num_train_envs  # 初始化总环境数量
        else:
            self.num_eval_envs = 0  # 初始化评价环境数量为0
            self.num_train_envs = cfg.env.num_envs  # 初始化训练环境数量
            self.num_envs = cfg.env.num_envs  # 初始化总环境数量

        torch._C._jit_set_profiling_mode(False)  # 关闭PyTorch JIT的性能分析模式
        torch._C._jit_set_profiling_executor(False)  # 关闭PyTorch JIT的执行器

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)  # 分配观察结果缓冲区
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)  # 分配奖励缓冲区
        self.rew_buf_pos = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)  # 分配偏正奖励缓冲区
        self.rew_buf_neg = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)  # 分配偏负奖励缓冲区
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)  # 分配重设标志缓冲区
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # 分配每个环境的时间缓冲区
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)  # 分配超时缓冲区
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                              dtype=torch.float)  # 分配特权访问缓冲区

        self.extras = {}  # 初始化额外数据字典

        self.create_sim()  # 创建模拟
        self.gym.prepare_sim(self.sim)  # 准备模拟

        self.enable_viewer_sync = True  # 初始设置视图同步为启用状态
        self.viewer = None  # 初始化视图为None

        if self.headless == False:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())  # 创建视图
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")  # 订阅"退出"键盘事件
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")  # 订阅"切换视图同步"键盘事件

    def get_observations(self):
        """
        获取观察值缓冲区内容

        Returns:
        - obs_buf: 观察值缓冲区内容

        注解:
        1. 返回观察值缓冲区的内容。

        注意事项:
        - 观察值缓冲区存储了环境的观察结果。
        """
        return self.obs_buf  # 返回观察值缓冲区内容

    def get_privileged_observations(self):
        """
        获取特权观察值缓冲区内容

        Returns:
        - privileged_obs_buf: 特权观察值缓冲区内容

        注解:
        1. 返回特权观察值缓冲区的内容。

        注意事项:
        - 特权观察值缓冲区存储了环境的特权观察结果。
        """
        return self.privileged_obs_buf  # 返回特权观察值缓冲区内容

    def reset_idx(self, env_ids):
        """
        重置指定的环境

        Args:
        - env_ids: 需要重置的环境的id列表

        注解:
        1. 重置指定的环境。

        注意事项:
        - 这是一个抽象方法，需要在子类中实现。
        - 重置环境通常包括重置环境的状态，清空缓冲区等操作。
        """
        raise NotImplementedError  # 抛出未实现错误

    def reset(self):
        """
        重置所有环境

        Returns:
        - obs: 重置后的观察值
        - privileged_obs: 重置后的特权观察值

        注解:
        1. 通过调用reset_idx方法并传入所有环境的ID来完成环境的重置。
        2. 对每一个环境进行步进操作。

        注意事项:
        - 重置环境通常包括重置环境的状态，清空缓冲区等操作。
        - 步进操作通常包括根据动作更新环境状态，计算奖励等操作。
        """
        self.reset_idx(torch.arange(self.num_envs, device=self.device))  # 重置所有环境
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))  # 对每一个环境进行步进操作
        return obs, privileged_obs  # 返回重置后的观察值和特权观察值

    def step(self, actions):
        """
        环境步进函数

        Args:
        - actions: 动作列表

        注解:
        1. 定义环境在收到一个动作后的变化情况。

        注意事项:
        - 这是一个抽象方法，需要在子类中实现。
        - 步进操作通常包括根据动作更新环境状态，计算奖励等操作。
        """
        raise NotImplementedError  # 抛出未实现错误

    def render_gui(self, sync_frame_time=True):
        """
        渲染图形用户界面

        Args:
        - sync_frame_time: 是否同步帧时间，默认为True

        注解:
        1. 这段代码只在有视图的情况下运行。
        2. 检查视图窗口是否已关闭，如果关闭了，那么结束程序。
        3. 检查键盘事件，如果收到了"退出"事件并且其值大于0，那么结束程序；如果收到了"切换视图同步"事件并且其值大于0，那么切换视图同步的启用状态。
        4. 如果设备不是CPU，那么获取执行的结果。
        5. 刷新视图，如果视图同步被启用，那么刷新图形，更新视图，并同步帧时间；如果视图同步被禁用，那么仅获取视图事件。

        注意事项:
        - 视图同步的启用和禁用可以通过键盘快捷方式进行切换。
        - 视图窗口的关闭会导致程序结束。
        """
        if self.viewer:  # 只在有视图的情况下运行
            if self.gym.query_viewer_has_closed(self.viewer):  # 检查视图窗口是否已关闭
                sys.exit()  # 如果关闭了，那么结束程序

            for evt in self.gym.query_viewer_action_events(self.viewer):  # 检查键盘事件
                if evt.action == "QUIT" and evt.value > 0:  # 如果收到了"退出"事件并且其值大于0，那么结束程序
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:  # 如果收到了"切换视图同步"事件并且其值大于0，那么切换视图同步的启用状态
                    self.enable_viewer_sync = not self.enable_viewer_sync

            if self.device != 'cpu':  # 如果设备不是CPU，那么获取执行的结果
                self.gym.fetch_results(self.sim, True)

            if self.enable_viewer_sync:  # 如果视图同步被启用
                self.gym.step_graphics(self.sim)  # 刷新图形
                self.gym.draw_viewer(self.viewer, self.sim, True)  # 更新视图
                if sync_frame_time:  # 如果需要同步帧时间
                    self.gym.sync_frame_time(self.sim)  # 同步帧时间
            else:  # 如果视图同步被禁用
                self.gym.poll_viewer_events(self.viewer)  # 仅获取视图事件

    def close(self):
        """
        关闭函数

        注解:
        1. 用来结束程序并清理资源。
        2. 如果有视图，那么销毁视图。
        3. 销毁模拟环境。

        注意事项:
        - 销毁视图和模拟环境是为了释放占用的资源，避免内存泄漏。
        """
        if self.headless == False:  # 如果有视图
            self.gym.destroy_viewer(self.viewer)  # 销毁视图
        self.gym.destroy_sim(self.sim)  # 销毁模拟环境
