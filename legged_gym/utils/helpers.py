import os
import copy
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

import torch

from legged_gym import MINI_GYM_ROOT_DIR, MINI_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    """
    将对象转换为字典形式

    Args:
    - obj: 任意的Python对象

    Returns:
    - result: 包含对象属性的字典

    注解:
    1. 检查对象是否有__dict__属性，没有则直接返回对象本身。
    2. 创建一个空字典用于存储结果。
    3. 遍历对象的所有属性，dir()函数获取对象所有属性和方法的列表。
    4. 跳过所有以"_"开头的私有属性。
    5. 初始化element空列表用于收集属性值。
    6. 获取对象属性的值。
    7. 如果属性值是列表，对列表中的每一项递归调用class_to_dict函数。
    8. 如果属性值不是列表，直接递归调用class_to_dict函数处理。
    9. 在结果字典中添加处理后的属性。
    10. 返回包含对象属性的字典。
    """
    # 1. 检查对象是否有可转换的__dict__属性
    if not hasattr(obj, "__dict__"):
        return obj
    
    # 2. 创建一个空字典用来存储属性和其对应的值
    result = {}
    
    # 3. 遍历对象的所有属性
    for key in dir(obj):
        # 4. 跳过私有属性
        if key.startswith("_"):
            continue
        
        # 5. 初始化element，用于收集属性值
        element = []
        # 6. 获取属性值
        val = getattr(obj, key)
        # 7. 如果属性值是列表，则递归调用class_to_dict
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        # 8. 否则，直接对属性值递归调用class_to_dict
        else:
            element = class_to_dict(val)
        
        # 9. 存储处理后的属性和值
        result[key] = element
    
    # 10. 返回结果字典
    return result


def update_class_from_dict(obj, dict):
    """
    使用字典更新对象的属性

    Args:
    - obj: 需要更新的对象
    - dict: 包含属性更新信息的字典

    注解:
    1. 遍历字典中的键值对。
    2. 尝试从对象中获取名称与键相同的属性，不存在则默认为None。
    3. 如果属性类型为class (type)，则对该属性递归调用update_class_from_dict。
    4. 如果不是class，则直接将字典中的值设置给对象的属性。
    5. 函数没有返回值。
    """
    # 1. 遍历字典中的键值对。
    for key, val in dict.items():
        # 2. 尝试获取对象的属性，若不存在，则返回None。
        attr = getattr(obj, key, None)
        
        # 3. 判断attr是否为一个类类型（type），是则递归更新。
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        # 4. 若attr不是类类型，则直接更新对象的属性值。
        else:
            setattr(obj, key, val)

    # 5. 函数没有返回值，所以直接结束。
    return


def set_seed(seed):
    """
    设置随机数生成器的种子，以确保结果的可重现性

    Args:
    - seed: 一个整数表示的种子值，如果为-1，则随机生成一个种子值

    注解:
    1. 检查传入的种子值是否为-1，若为-1则随机生成一个0到9999之间的种子值。
    2. 打印出设置的种子值以方便追踪和调试。
    3. 设置Python内置随机生成器的种子。
    4. 设置NumPy随机生成器的种子。
    5. 设置PyTorch随机生成器的种子。
    6. 设置OS的环境变量PYTHONHASHSEED，这会影响Python哈希算法的随机性，用于控制hash()的结果。
    7. 如果使用CUDA，则还需要设置CUDA随机生成器的种子。
    8. 设置所有CUDA设备的随机种子，保证多GPU环境下的一致性。
    """
    # 1. 检查种子值，若为-1则随机生成新的种子
    if seed == -1:
        seed = np.random.randint(0, 10000)
    
    # 2. 打印设置的种子值
    print("Setting seed: {}".format(seed))
    
    # 3. 设置Python内置随机生成器的种子
    random.seed(seed)
    # 4. 设置NumPy随机生成器的种子
    np.random.seed(seed)
    # 5. 设置PyTorch随机生成器的种子
    torch.manual_seed(seed)
    # 6. 设置环境变量PYTHONHASHSEED
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 7. 设置PyTorch在CUDA上的随机种子
    torch.cuda.manual_seed(seed)
    # 8. 设置所有CUDA设备的随机种子
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    """
    解析仿真参数，并根据命令行参数和配置文件更新仿真设置

    Args:
    - args: 命令行参数对象，包含物理引擎、计算设备和其他仿真选项
    - cfg: 配置字典，通常从一个JSON或YAML配置文件加载

    Returns:
    - sim_params: 配置好的仿真参数对象

    注解:
    1. 初始化仿真参数对象sim_params。
    2. 根据args中提供的物理引擎类型设置特定的仿真参数：
       - 如果使用的是FLEX物理引擎，并且计算设备不是CPU，输出警告信息。
       - 如果使用的是PHYSX物理引擎，则设置相应的GPU参数和subscenes参数。
    3. 设置是否使用GPU pipeline。
    4. 如果在cfg中提供了'仿真(sim)'相关的配置项，解析这些配置并更新sim_params对象。
    5. 如果指定了物理引擎为PHYSX且命令行提供了线程数，则覆盖之前的设置。
    6. 返回配置好的sim_params对象。
    """
    # 1. 初始化仿真参数对象
    sim_params = gymapi.SimParams()

    # 2. 从args设置一些仿真参数
    if args.physics_engine == gymapi.SIM_FLEX:
        # 当使用FLEX物理引擎并且计算设备不是CPU时，输出一个警告
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        # 使用PHYSX并且指定了GPU使用和subscenes的数量
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    
    # 3. 设置是否使用GPU pipeline
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # 4. 如果cfg提供了仿真相关的配置项，则更新sim_params
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # 5. 如果使用PHYSX并且指定了线程数，则更新线程设置
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    # 6. 返回配置好的sim_params对象
    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    """
    从指定根目录中获取模型的加载路径

    Args:
    - root: 模型保存的根目录路径
    - load_run: 指定加载的运行目录名，-1表示加载最新运行目录
    - checkpoint: 指定加载的模型检查点，-1表示加载最新检查点

    Returns:
    - load_path: 选定模型的完整文件路径

    注解:
    1. 尝试列出root目录下的所有子目录。
    2. 按日期排序运行目录，注意要处理月份改变的情况（未实现）。
    3. 移除runs列表中的'exported'目录（如果存在）。
    4. 获取最新的运行目录。
    5. 如果没有提供load_run参数，或者给定的值为-1，则使用最新的运行目录。
    6. 如果提供了load_run参数，则使用指定的运行目录。
    7. 对于checkpoint，如果为-1，则从指定运行目录中找到最新的模型检查点。
    8. 如果给定了checkpoint，则直接使用该检查点名构造模型文件名。
    9. 构造最终的模型文件加载路径，并返回。
    """
    try:
        # 1. 列出根目录下所有子目录
        runs = os.listdir(root)
        # 2. 待实现：按日期排序runs
        runs.sort()
        # 3. 如果存在'exported'目录，则移除它
        if 'exported' in runs: runs.remove('exported')
        # 4. 获取最新的运行目录
        last_run = os.path.join(root, runs[-1])
    except:
        # 如果列出目录失败，抛出异常
        raise ValueError("No runs in this directory: " + root)
    
    # 5. 确定加载哪个运行目录
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    # 7. 确定加载哪个检查点
    if checkpoint == -1:
        # 列出所有含模型名的文件
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        # 获取最新的模型文件名
        model = models[-1]
    else:
        # 格式化给定的检查点为模型文件名
        model = "model_{}.pt".format(checkpoint)

    # 9. 构造模型的完整文件路径
    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    """
    根据命令行参数更新环境配置和训练配置

    Args:
    - env_cfg: 环境的配置对象
    - cfg_train: 训练的配置对象
    - args: 命令行参数对象

    Returns:
    - env_cfg: 更新后的环境配置对象
    - cfg_train: 更新后的训练配置对象

    注解:
    1. 如果env_cfg对象不为空，那么根据args中的参数更新环境的配置信息。
    2. 如果cfg_train对象不为空，那么根据args中的参数更新训练的配置信息。
    3. 根据args中不同的参数，可能会更新环境数量、随机种子、最大迭代次数、实验名称等。

    注意事项:
    - 此函数依赖于命令行参数对象args提供的参数，以及预先存在的env_cfg和cfg_train配置对象。
    """

    # 如果环境配置对象不为空
    if env_cfg is not None:
        # 如果命令行参数指定了环境数量
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs  # 更新环境数量

    # 如果训练配置对象不为空
    if cfg_train is not None:
        # 如果命令行参数指定了随机种子
        if args.seed is not None:
            cfg_train.seed = args.seed  # 更新随机种子
        # 如果命令行参数指定了最大迭代次数
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations  # 更新最大迭代次数
        # 如果命令行参数指示要继续之前的训练
        if args.resume:
            cfg_train.runner.resume = args.resume  # 设置训练继续运行
        # 如果命令行参数指定了实验名称
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name  # 更新实验名称
        # 如果命令行参数指定了运行名称
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name  # 更新运行名称
        # 如果命令行参数指定了要加载的运行
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run  # 更新要加载的运行
        # 如果命令行参数指定了检查点
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint  # 更新检查点

    # 返回更新后的环境和训练配置对象
    return env_cfg, cfg_train


def get_args():
    """
    解析命令行参数并返回配置参数对象

    Returns:
    - args: 命令行参数构成的命名空间对象

    注解:
    1. 定义了一系列自定义的命令行参数。
    2. 使用gymutil的parse_arguments函数解析命令行参数。
    3. 对解析后的参数做了额外处理，以便对设备进行命名和联网。
    """
    # 自定义的命令行参数列表
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat", "help": "从检查点开始训练或测试。如提供，将覆盖配置文件中的设定。"},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "从检查点恢复训练"},
        {"name": "--experiment_name", "type": str,  "help": "运行或加载实验的名称。如提供，将覆盖配置文件中的设定。"},
        {"name": "--run_name", "type": str,  "help": "运行的名称。如提供，将覆盖配置文件中的设定。"},
        {"name": "--load_run", "type": str,  "help": "当resume=True时加载的运行名称。如果是-1：将加载最后一次运行。如提供，将覆盖配置文件中的设定。"},
        {"name": "--checkpoint", "type": int,  "help": "保存的模型检查点编号。如果是-1：将加载最后一个检查点。如提供，将覆盖配置文件中的设定。"},
        {"name": "--headless", "action": "store_true", "default": True, "help": "始终关闭显示器"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "使用horovod进行多GPU训练"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": '用于RL算法的设备（cpu, gpu, cuda:0, cuda:1等）'},
        {"name": "--num_envs", "type": int, "help": "要创建的环境数量。如提供，将覆盖配置文件中的设定。"},
        {"name": "--seed", "type": int, "help": "随机种子。如提供，将覆盖配置文件中的设定。"},
        {"name": "--max_iterations", "type": int, "help": "最大训练迭代次数。如提供，将覆盖配置文件中的设定。"},
    ]
    # 解析命令行参数
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # 参数名称对齐
    args.sim_device_id = args.compute_device_id  # 将设备ID和计算设备ID保持一致
    args.sim_device = args.sim_device_type  # 将仿真设备名称设置为计算设备类型
    if args.sim_device=='cuda':  # 如果设备类型是cuda，则添加设备ID用以指定
        args.sim_device += f":{args.sim_device_id}"
    return args  # 返回解析好的参数


def export_policy_as_jit(actor_critic, path):
    """
    将策略模型导出为Torch JIT模块

    Args:
    - actor_critic: 包含actor和critic的策略模型
    - path: 导出模型的目标文件夹路径

    注解:
    1. 检查模型是否有'memory_a'属性，如果有，假设模型使用了LSTM。目前代码注释提到了TODO项，可能未来会考虑GRU。
    2. 如果模型使用LSTM，创建PolicyExporterLSTM的实例并导出策略。
    3. 如果模型没有使用LSTM，先将path路径下的文件夹创建（如果不存在），然后导出模型。
    4. 复制actor_critic中的actor部分，转移到CPU上，并使用torch.jit.script进行转换以便能够将模型存储为一个Torch Script模块。
    5. 保存转换后的模型到指定的路径。
    """
    if hasattr(actor_critic, 'memory_a'):
        # 检查actor_critic是否有'memory_a'属性，如果有，则假设使用了LSTM
        exporter = PolicyExporterLSTM(actor_critic)  # 创建模型导出器实例针对LSTM
        exporter.export(path)  # 导出模型
    else:
        # 如果模型不包含LSTM，执行以下操作
        os.makedirs(path, exist_ok=True)  # 创建目标文件夹（如果该路径不存在）
        path = os.path.join(path, 'policy_1.pt')  # 指定导出文件的完整路径
        model = copy.deepcopy(actor_critic.actor).to('cpu')  # 复制actor到CPU
        traced_script_module = torch.jit.script(model)  # 将模型转换为Torch Script模块
        traced_script_module.save(path)  # 保存模块至指定路径


class PolicyExporterLSTM(torch.nn.Module):

    def __init__(self, actor_critic):
        """
        初始化代理算法

        Args:
        - actor_critic: 集成了演员（actor）和评论家（critic）功能的一个对象实例

        注解:
        1. super().__init__(): 在子类中调用父类的初始化方法。
        2. self.actor: 通过深度复制(actor_critic对象中的演员部分来创建一个新的演员对象并赋值给self.actor。
        3. self.is_recurrent: 将传入的actor_critic中的is_recurrent属性值保存下来，判断策略是否是循环的。
        4. self.memory: 通过深度复制来创建一个新的记忆体，并赋值给self.memory。
        5. self.memory.cpu(): 将记忆体移至CPU，这在训练中使用GPU时是必要的。
        6. self.register_buffer: 注册hidden_state和cell_state，使它们变为模型的持久状态，
        无论何时加载该模型状态都能获取到这两个变量的状态。
        
        注意事项:
        - 为了确保模型可以在不同的硬件上运行，重要的张量需要正确地移到CPU或GPU上。
        - 持久状态是指那些即便没有参数也应该与模型一同保存和加载的状态。
        """
        super().__init__()  # 调用父类的初始化方法
        # 深复制actor_critic中的actor部分，并将复制后的actor移至CPU
        self.actor = copy.deepcopy(actor_critic.actor)
        # 获取策略是否是循环的标志
        self.is_recurrent = actor_critic.is_recurrent  
        # 深复制actor_critic中的记忆体（假设是LSTM结构），并将复制后的记忆体移至CPU
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()  # 将记忆体转移到CPU
        # 注册hidden_state和cell_state为模块的持久状态
        self.register_buffer('hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer('cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        """
        定义了该模型的前馈（forward pass）计算过程

        Args:
        - x: 输入到模型的特征数据

        Returns:
        - 返回执行actor网络后的结果

        注解:
        1. self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state)): 输入x通过LSTM记忆单元，并获取输出以及新的隐藏状态和细胞状态。
        x数据需要通过unsqueeze(0)来增加一个批次维度，因为LSTM期望三维输入：批次，序列，特征。
        2. self.hidden_state[:] = h: 使用新的隐藏状态h来更新当前的隐藏状态。
        3. self.cell_state[:] = c: 使用新的细胞状态c来更新当前的细胞状态。
        4. self.actor(out.squeeze(0)): 将输出out的批次维度去掉后，通过actor网络进行处理，并返回最后的结果。

        注意事项:
        - 在循环神经网络中，维持和更新隐藏状态和细胞状态是非常重要的，它们负责记住序列中的信息。
        - out.squeeze(0) 是因为LSTM的输出是三维的，而actor网络期望的输入是二维的，因此需要移除多余的批次维度。
        """
        # 将输入通过LSTM记忆单元，并更新隐藏状态和细胞状态
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        # 更新隐藏状态和细胞状态
        self.hidden_state[:] = h  
        self.cell_state[:] = c  
        # 将LSTM的输出提取出来，移除批次维度，然后通过actor网络，并返回最终结果
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        """
        重置LSTM记忆体的隐藏状态和细胞状态
        
        注解:
        1. @torch.jit.export: 这个装饰器告诉Torch JIT编译器这个方法可以被导出，即使它不是类的forward方法。
        2. self.hidden_state[:] = 0.: 将隐藏状态的所有值重置为0，意味着记忆体忘记了之前学到的信息。
        3. self.cell_state[:] = 0.: 将细胞状态的所有值重置为0，与隐藏状态的重置类似，重置LSTM的长期记忆。

        注意事项:
        - 这个方法通常用在序列开始时或者在一个新的序列要被处理之前，确保神经网络不会受到上一个序列的状态的影响。
        """
        # 将隐藏状态和细胞状态重置为0
        self.hidden_state[:] = 0.  
        self.cell_state[:] = 0.  
 
    def export(self, path):
        """
        导出模型为Torch Script格式并保存到磁盘

        Args:
        - path: 模型导出文件的目标文件夹路径

        注解:
        1. os.makedirs(path, exist_ok=True): 创建文件夹路径，如果路径已经存在，则不抛出错误。
        2. path = os.path.join(path, 'policy_lstm_1.pt'): 设置模型文件的完整路径名。
        3. self.to('cpu'): 将模型的所有参数和缓冲区移动到CPU，保证在不同环境中的兼容性。
        4. torch.jit.script(self): 将当前模块转换成Torch Script的形式。这一步是将模型和其方法转换为一种静态可序列化的形式，便于在未来不同的执行环境中使用。
        5. traced_script_module.save(path): 将转换后的模型保存到指定路径的文件中。

        注意事项:
        - 在转换过程中，torch.jit.script会检查模型的代码来确定所有的路径和变量都被包含在内，因此需要模型中所有的代码是可脚本化的。
        - 使用此方法导出的模型具有更好的跨平台兼容性和更高效的执行速度。
        """
        # 确保目标文件夹存在
        os.makedirs(path, exist_ok=True)
        # 设置文件保存路径
        path = os.path.join(path, 'policy_lstm_1.pt') 
        # 确保模型在CPU上
        self.to('cpu') 
        # 转换模块为Torch Script
        traced_script_module = torch.jit.script(self) 
        # 保存模型到文件
        traced_script_module.save(path)  
