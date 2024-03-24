from legged_gym.envs import AnymalCRoughCfg, AnymalCRoughCfgPPO

class AnymalCFlatCfg(AnymalCRoughCfg):
    """
    AnymalC机器人在平坦地形配置的类，继承自AnymalC机器人的粗糙地形配置类

    属性:
    - env: 环境相关配置
    - terrain: 地形相关配置
    - asset: 资源相关配置
    - rewards: 奖励相关配置
    - commands: 指令相关配置
    - domain_rand: 域随机化相关配置
    """

    class env(AnymalCRoughCfg.env):
        """
        环境配置

        属性:
        - num_observations: 观测的数量，设为48
        """
        num_observations = 48   # 观测的数量
        num_privileged_obs = 2  # 特权观察数量

        priv_observe_friction = True
        priv_observe_restitution = True  # 观察恢复系数

    class terrain(AnymalCRoughCfg.terrain):
        """
        地形配置

        属性:
        - mesh_type: 地形的网格类型，设为'plane'表示平面
        - measure_heights: 是否测量高度，设为False表示不测量
        """
        mesh_type = 'plane'  # 地形的网格类型
        measure_heights = False  # 是否测量高度
  
    class asset(AnymalCRoughCfg.asset):
        """
        资源配置

        属性:
        - self_collisions: 自身碰撞的设置，0表示启用，1表示禁用
        """
        self_collisions = 0  # 自身碰撞的设置

    class rewards(AnymalCRoughCfg.rewards):
        """
        奖励配置

        属性:
        - max_contact_force: 最大接触力，设为350.0
        - scales: 奖励比例的配置
        """
        max_contact_force = 350.  # 最大接触力
        class scales(AnymalCRoughCfg.rewards.scales):
            """
            奖励比例配置

            属性:
            - orientation: 方向奖励比例，设为-5.0
            - torques: 扭矩奖励比例，设为-0.000025
            - feet_air_time: 脚部空中时间奖励比例，设为2.0
            """
            orientation = -5.0  # 方向奖励比例
            torques = -0.000025  # 扭矩奖励比例
            feet_air_time = 2.0  # 脚部空中时间奖励比例
            # feet_contact_forces = -0.01  # 脚部接触力奖励比例（已注释）
    
    class commands(AnymalCRoughCfg.commands):
        """
        指令配置

        属性:
        - heading_command: 是否有朝向指令，设为False表示没有
        - resampling_time: 重新采样时间，设为4.0秒
        - ranges: 指令范围的配置
        """
        heading_command = False  # 是否有朝向指令
        resampling_time = 4.0  # 重新采样时间
        class ranges(AnymalCRoughCfg.commands):
            """
            指令范围配置

            属性:
            - ang_vel_yaw: 偏航角速度的范围，设为[-1.5, 1.5]
            """
            ang_vel_yaw = [-1.5, 1.5]  # 偏航角速度的范围

    class domain_rand(AnymalCRoughCfg.domain_rand):
        """
        域随机化配置

        属性:
        - friction_range: 摩擦系数范围，设为[0., 1.5]
        """
        friction_range = [0., 1.5]  # 摩擦系数范围

class AnymalCFlatCfgPPO(AnymalCRoughCfgPPO):
    """
    AnymalC机器人使用PPO算法在平坦地形配置的类，继承自AnymalC机器人使用PPO算法的粗糙地形配置类

    属性:
    - policy: 策略网络相关配置
    - algorithm: 算法相关配置
    - runner: 运行相关配置
    """

    class policy(AnymalCRoughCfgPPO.policy):
        """
        策略网络配置

        属性:
        - actor_hidden_dims: 演员网络隐藏层维度，设为[128, 64, 32]
        - critic_hidden_dims: 评论家网络隐藏层维度，设为[128, 64, 32]
        - activation: 激活函数类型，设为'elu'
        """
        actor_hidden_dims = [128, 64, 32]  # 演员网络隐藏层维度
        critic_hidden_dims = [128, 64, 32]  # 评论家网络隐藏层维度
        activation = 'elu'  # 激活函数类型

    class algorithm(AnymalCRoughCfgPPO.algorithm):
        """
        算法配置

        属性:
        - entropy_coef: 熵系数，设为0.01
        """
        entropy_coef = 0.01  # 熵系数

    class runner(AnymalCRoughCfgPPO.runner):
        """
        运行配置

        属性:
        - run_name: 运行的名称，可以根据需要设置
        - experiment_name: 实验的名称，设为'flat_anymal_c'
        - load_run: 加载运行的编号，-1表示不加载之前的运行
        - max_iterations: 最大迭代次数，设为300
        """
        run_name = ''  # 运行的名称
        experiment_name = 'flat_anymal_c'  # 实验的名称
        load_run = -1  # 加载运行的编号
        max_iterations = 10000  # 最大迭代次数
