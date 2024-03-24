from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class AnymalCRoughCfg(LeggedRobotCfg):
    """
    AnymalCRoughCfg类继承自LeggedRobotCfg，用于配置Anymal C型机器人在粗糙地形上的仿真环境。

    类属性:
    - env: 设置仿真环境的参数。
    - terrain: 设置地形的参数。
    - init_state: 设置机器人初始状态的参数。
    - control: 设置机器人控制参数。
    - asset: 设置机器人资源文件的参数。
    - domain_rand: 设置领域随机化参数。
    - rewards: 设置训练过程中的奖励参数。
    """

    class env(LeggedRobotCfg.env):
        """
        env子类，设置仿真环境相关参数。

        类属性:
        - num_envs: 同时运行的环境数量。
        - num_actions: 每个环境中的动作数量。
        """
        num_envs = 8  # 同时运行的环境数量
        num_actions = 12  # 每个环境中的动作数量

    class terrain(LeggedRobotCfg.terrain):
        """
        terrain子类，设置地形相关参数。

        类属性:
        - mesh_type: 地形的网格类型。
        """
        mesh_type = 'trimesh'  # 地形的网格类型

    class init_state(LeggedRobotCfg.init_state):
        """
        机器人初始状态配置的子类

        类属性说明:
        - pos: 为机器人在仿真环境中的初始位置坐标，格式为三维列表[x, y, z]，单位为米。
        - default_joint_angles: 用于设置当动作指令为0时各关节的目标角度，这是一个字典，包含每个关节的名称及其对应的角度值，单位为弧度。

        注意:
        - "HAA"表示髋关节水平轴扭矩（Hip Abduction/Adduction）。
        - "HFE"表示髋关节前后弯曲扭矩（Hip Flexion/Extension）。
        - "KFE"表示膝关节弯曲扭矩（Knee Flexion/Extension）。
        - "LF", "LH", "RF", "RH"分别代表左前（Left Front）、左后（Left Hind）、右前（Right Front）、右后（Right Hind）的腿部。
        - 关节角度设置是基于特定的机器人模型和仿真环境，旨在提供一个稳定的起始姿态。
        """
        pos = [0.0, 0.0, 0.6]  # 设置机器人在仿真环境中的初始位置坐标，单位为米

        # 设置动作为0时的目标关节角度，这些值是提供稳定起始姿态的关键参数
        default_joint_angles = {
            "LF_HAA": 0.0,  # 左前髋关节水平轴扭矩的目标角度
            "LH_HAA": 0.0,  # 左后髋关节水平轴扭矩的目标角度
            "RF_HAA": -0.0,  # 右前髋关节水平轴扭矩的目标角度（-0.0为了表示这是一个经过考虑的值，即使它数值上等同于0.0）
            "RH_HAA": -0.0,  # 右后髋关节水平轴扭矩的目标角度
            "LF_HFE": 0.4,  # 左前髋关节前后弯曲扭矩的目标角度
            "LH_HFE": -0.4,  # 左后髋关节前后弯曲扭矩的目标角度
            "RF_HFE": 0.4,  # 右前髋关节前后弯曲扭矩的目标角度
            "RH_HFE": -0.4,  # 右后髋关节前后弯曲扭矩的目标角度
            "LF_KFE": -0.8,  # 左前膝关节弯曲扭矩的目标角度
            "LH_KFE": 0.8,  # 左后膝关节弯曲扭矩的目标角度
            "RF_KFE": -0.8,  # 右前膝关节弯曲扭矩的目标角度
            "RH_KFE": 0.8,  # 右后膝关节弯曲扭矩的目标角度
        }

    class control(LeggedRobotCfg.control):
        """
        control子类，设置机器人控制相关参数。

        类属性:
        - stiffness: PD控制器的刚度参数。
        - damping: PD控制器的阻尼参数。
        - action_scale: 动作缩放比例。
        - decimation: 控制动作更新的频率。
        - use_actuator_network: 是否使用执行器网络。
        - actuator_net_file: 执行器网络文件的路径。
        """
        # PD控制器的刚度参数，单位为牛顿米每弧度
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}
        # PD控制器的阻尼参数，单位为牛顿米秒每弧度
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}
        action_scale = 0.5  # 动作缩放比例
        decimation = 4  # 控制动作更新的频率
        use_actuator_network = True  # 是否使用执行器网络
        # 执行器网络文件的路径，LEGGED_GYM_ROOT_DIR为环境变量
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class asset(LeggedRobotCfg.asset):
        """
        asset子类，设置机器人资源文件相关参数。

        类属性:
        - file: 机器人URDF文件的路径。
        - name: 机器人的名称。
        - foot_name: 机器人脚部的名称。
        - penalize_contacts_on: 接触惩罚适用的部件名称列表。
        - terminate_after_contacts_on: 接触终止适用的部件名称列表。
        - self_collisions: 自碰撞设置，1为禁用，0为启用。
        """
        # 机器人URDF文件的路径，LEGGED_GYM_ROOT_DIR为环境变量
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c.urdf"
        name = "anymal_c"  # 机器人的名称
        foot_name = "FOOT"  # 机器人脚部的名称
        # 接触惩罚适用的部件名称列表
        penalize_contacts_on = ["SHANK", "THIGH"]
        # 接触终止适用的部件名称列表
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 自碰撞设置，1为禁用，0为启用

    class domain_rand(LeggedRobotCfg.domain_rand):
        """
        domain_rand子类，设置领域随机化相关参数。

        类属性:
        - randomize_base_mass: 是否随机化基础质量。
        - added_mass_range: 添加质量的范围。
        """
        randomize_base_mass = True  # 是否随机化基础质量
        added_mass_range = [-5., 5.]  # 添加质量的范围，单位为千克

    class rewards(LeggedRobotCfg.rewards):
        """
        rewards子类，设置训练过程中的奖励相关参数。

        类属性:
        - base_height_target: 基础高度目标。
        - max_contact_force: 最大接触力。
        - only_positive_rewards: 是否只有正奖励。
        """
        base_height_target = 0.5  # 基础高度目标，单位为米
        max_contact_force = 500.  # 最大接触力，单位为牛顿
        only_positive_rewards = True  # 是否只有正奖励

        class scales(LeggedRobotCfg.reward_scales):
            """
            scales子类，用于设置奖励的缩放比例。
            """
            pass  # 当前类为空，用于未来可能的扩展


class AnymalCRoughCfgPPO(LeggedRobotCfgPPO):
    """
    AnymalCRoughCfgPPO类继承自LeggedRobotCfgPPO，用于配置Anymal C型机器人使用PPO算法在粗糙地形上的训练运行参数。

    类属性:
    - runner: 设置训练运行的参数。
    """

    class runner(LeggedRobotCfgPPO.runner):
        """
        runner子类，设置训练运行相关参数。

        类属性:
        - run_name: 训练运行的名称。
        - experiment_name: 实验名称。
        - load_run: 加载先前运行的ID。
        """
        run_name = ''  # 训练运行的名称，通常用于区分不同的训练实验
        experiment_name = 'rough_anymal_c'  # 实验名称，用于标识当前训练的环境或任务
        load_run = -1  # 加载先前运行的ID，-1通常表示不加载任何先前的运行
