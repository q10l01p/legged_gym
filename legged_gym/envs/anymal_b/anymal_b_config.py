from legged_gym.envs import AnymalCRoughCfg, AnymalCRoughCfgPPO

class AnymalBRoughCfg(AnymalCRoughCfg):
    """
    AnymalB机器人在粗糙地形配置的类，继承自AnymalC机器人的粗糙地形配置类

    属性:
    - asset: 包含机器人URDF文件路径、名称和脚部名称的配置
    - rewards: 包含奖励相关配置的类
    """

    class asset(AnymalCRoughCfg.asset):
        """
        机器人资源配置

        属性:
        - file: URDF文件的路径，指向AnymalB机器人的URDF文件
        - name: 机器人的名称，设为"anymal_b"
        - foot_name: 机器人脚部的名称，设为'FOOT'
        """
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_b/urdf/anymal_b.urdf'  # AnymalB机器人URDF文件的路径
        name = "anymal_b"  # 机器人的名称
        foot_name = 'FOOT'  # 机器人脚部的名称

    class rewards(AnymalCRoughCfg.rewards):
        """
        奖励配置

        属性:
        - scales: 继承自AnymalCRoughCfg.rewards.scales的奖励比例配置
        """
        class scales(AnymalCRoughCfg.rewards.scales):
            # 这里可以添加或覆盖AnymalC的奖励比例配置
            pass

class AnymalBRoughCfgPPO(AnymalCRoughCfgPPO):
    """
    AnymalB机器人使用PPO算法在粗糙地形配置的类，继承自AnymalC机器人使用PPO算法的粗糙地形配置类

    属性:
    - runner: 包含运行名称、实验名称和加载运行编号的配置
    """

    class runner(AnymalCRoughCfgPPO.runner):
        """
        运行配置

        属性:
        - run_name: 运行的名称，可以根据需要设置
        - experiment_name: 实验的名称，设为'rough_anymal_b'
        - load_run: 加载运行的编号，-1表示不加载之前的运行
        """
        run_name = ''  # 运行的名称
        experiment_name = 'rough_anymal_b'  # 实验的名称
        load_run = -1  # 加载运行的编号
