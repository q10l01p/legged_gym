from isaacgym.torch_utils import *
from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg


class Anymal(LeggedRobot):
    """
    Anymal类继承自LeggedRobot，用于创建Anymal型机器人的实例，并管理其仿真和控制。

    属性:
    - cfg: AnymalCRoughCfg类的实例，包含机器人的配置信息。

    方法:
    - __init__: 构造函数，初始化机器人实例。
    - reset_idx: 重置特定环境ID的状态。
    - _init_buffers: 初始化缓冲区。
    - _compute_torques: 计算机器人关节的扭矩。
    """

    cfg: AnymalCRoughCfg  # AnymalCRoughCfg类的实例，包含机器人的配置信息

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """
        初始化Anymal机器人的构造函数

        参数:
        - cfg: 包含机器人配置信息的字典。
        - sim_params: 包含仿真参数的字典。
        - physics_engine: 使用的物理引擎。
        - sim_device: 仿真运行的设备名称（如'cpu'或'cuda:0'）。
        - headless: 布尔值，指明是否在无头模式下运行仿真（无图形界面）。

        主要步骤:
        1. 调用父类构造函数，传递所有初始化参数。
        2. 检查配置信息，如果需要使用执行器网络，则加载执行器网络并将其分配到指定运行设备上。

        注释:
        - `cfg.control.use_actuator_network` 如果为True，表示需要使用执行器网络。
        - `LEGGED_GYM_ROOT_DIR` 是一个环境变量，用于指定加载执行器网络的根目录路径。
        - `actuator_network_path` 是执行器网络文件的完整路径。
        - 执行器网络使用`torch.jit.load`进行加载，这意味着网络已被追踪或脚本化为TorchScript模型。
        - 加载后的执行器网络使用`.to(self.device)`移动到指定的仿真设备上。
        """
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)  # 调用父类构造函数来初始化

        # 检查配置信息中是否指定使用执行器网络
        if self.cfg.control.use_actuator_network:
            # 格式化路径使用LEGGED_GYM_ROOT_DIR环境变量并加载执行器网络
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            # 加载执行器网络并将其移至指定的计算设备
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)

    def reset_idx(self, env_ids):
        """
        为特定的环境ID重置执行器网络的状态

        参数:
        - env_ids: 一个包含需要重置状态的环境ID的列表

        主要步骤:
        1. 调用父类的reset_idx方法来重置基础状态。
        2. 对指定的环境ID，将执行器网络的隐藏状态和细胞状态重置为零。

        注释:
        - 这个函数通常在环境需要在新的回合开始时重置时调用。
        - `env_ids` 中的ID对应于需要重置的环境。
        - 将隐藏状态和细胞状态设置为0可以清除之前的状态信息，为新回合初始化状态。
        """
        super().reset_idx(env_ids)  # 调用父类的reset_idx方法来重置基础状态

        # 将指定环境ID的执行器网络隐藏状态重置为0
        self.sea_hidden_state_per_env[:, env_ids] = 0.

        # 将指定环境ID的执行器网络细胞状态重置为0
        self.sea_cell_state_per_env[:, env_ids] = 0.

    def _init_buffers(self):
        """
        初始化执行器网络的缓冲区

        主要步骤:
        1. 调用父类方法来初始化基础缓冲区。
        2. 初始化执行器网络的输入张量。
        3. 初始化执行器网络的隐藏状态和细胞状态张量。
        4. 将隐藏状态和细胞状态张量重新组织为每个环境的视图，以方便后续操作。

        注释:
        - `num_envs` 表示环境的数量。
        - `num_actions` 表示每个环境中可能的动作数量。
        - `device` 是计算将在其上执行的设备（如CPU或GPU）。
        - `requires_grad` 设置为False表示这些张量在计算图中不需要梯度，这通常用于只需通过前向传播的张量。
        """
        super()._init_buffers()  # 调用父类的_init_buffers方法来初始化基础缓冲区

        # 初始化执行器网络的输入张量，大小为(num_envs*num_actions, 1, 2)，并指定计算设备和梯度需求
        self.sea_input = torch.zeros(self.num_envs * self.num_actions, 1, 2, device=self.device, requires_grad=False)

        # 初始化执行器网络的隐藏状态张量，大小为(2, num_envs*num_actions, 8)，并指定计算设备和梯度需求
        self.sea_hidden_state = torch.zeros(2, self.num_envs * self.num_actions, 8, device=self.device,
                                            requires_grad=False)

        # 初始化执行器网络的细胞状态张量，大小为(2, num_envs*num_actions, 8)，并指定计算设备和梯度需求
        self.sea_cell_state = torch.zeros(2, self.num_envs * self.num_actions, 8, device=self.device,
                                          requires_grad=False)

        # 将隐藏状态张量重新组织为每个环境的视图，形状为(2, num_envs, num_actions, 8)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)

        # 将细胞状态张量重新组织为每个环境的视图，形状为(2, num_envs, num_actions, 8)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

    def _compute_torques(self, actions):
        """
        计算机器人关节的扭矩。

        参数:
        - actions: 动作值数组，表示希望达到的目标位置或目标动作。

        返回值:
        - torques: 计算得到的扭矩值数组，用于驱动机器人关节。

        主要步骤:
        1. 判断配置是否指定使用执行器网络。
        2. 如果使用执行器网络，则构造网络输入并计算扭矩。
        3. 如果不使用执行器网络，则调用基类的方法使用PD控制器计算扭矩。
        4. 返回计算得到的扭矩。

        注释:
        - 执行器网络用于更精细的动作执行，而PD控制器则是一种更传统的控制方式。
        """
        # 根据配置选择使用执行器网络还是PD控制器
        if self.cfg.control.use_actuator_network:
            with torch.inference_mode():  # 使用推理模式以提高性能
                # 准备执行器网络的输入数据
                # 输入为当前动作的目标位置，经过缩放和默认位置的调整后减去当前关节位置
                self.sea_input[:, 0, 0] = (
                            actions * self.cfg.control.action_scale + self.default_dof_pos - self.dof_pos).flatten()
                # 输入还包括当前关节的速度
                self.sea_input[:, 0, 1] = self.dof_vel.flatten()
                # 通过执行器网络计算扭矩，并更新隐藏状态和细胞状态
                torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(self.sea_input, (
                self.sea_hidden_state, self.sea_cell_state))
            return torques  # 返回计算得到的扭矩值
        else:
            # 使用PD控制器计算扭矩
            return super()._compute_torques(actions)  # 调用父类方法计算扭矩
