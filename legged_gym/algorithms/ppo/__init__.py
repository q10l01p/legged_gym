import torch
import time
from collections import deque
import copy
import os

from ml_logger import logger
from params_proto import PrefixProto

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage


def class_to_dict(obj) -> dict:
    """
    将类实例转换为字典。

    参数:
    - obj: 要转换的类实例。

    返回:
    - result: 类实例的字典表示。
    """
    # 检查对象是否具有 __dict__ 属性，如果没有则直接返回该对象。
    if not hasattr(obj, "__dict__"):
        return obj
    # 初始化结果字典。
    result = {}
    # 遍历对象的所有属性。
    for key in dir(obj):
        # 跳过私有属性和特定的 'terrain' 属性。
        if key.startswith("_") or key == "terrain":
            continue
        # 获取属性值。
        val = getattr(obj, key)
        # 如果属性值是列表，递归转换列表中的每个元素。
        if isinstance(val, list):
            element = [class_to_dict(item) for item in val]
        else:
            # 对于非列表属性，直接递归转换。
            element = class_to_dict(val)
        # 添加转换后的属性到结果字典。
        result[key] = element
    # 返回结果字典。
    return result


class DataCaches:
    """
    数据缓存类，用于管理数据缓存。

    属性:
    - slot_cache: SlotCache 实例，用于管理特定类型的数据缓存。
    - dist_cache: DistCache 实例，用于管理另一种类型的数据缓存。
    """

    def __init__(self, curriculum_bins):
        # 导入 SlotCache 和 DistCache 类。
        from go1_gym_learn.ppo.metrics_caches import SlotCache, DistCache

        # 初始化 slot_cache 和 dist_cache。
        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


# 创建 DataCaches 实例。
caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    """
    RunnerArgs 类用于存储算法运行的参数配置。
    """
    algorithm_class_name = 'RMA'  # 默认算法类名
    num_steps_per_env = 24  # 每迭代环境的步数
    max_iterations = 1500  # 策略更新的最大迭代次数
    save_interval = 400  # 检查保存间隔
    save_video_interval = 100  # 视频保存间隔
    log_freq = 10  # 日志记录频率
    resume = False  # 是否从上一次运行恢复
    load_run = -1  # 要加载的运行编号，默认为最后一次运行
    checkpoint = -1  # 要加载的模型检查点，默认为最后保存的模型
    resume_path = None  # 恢复路径，根据 load_run 和 checkpoint 更新
    resume_curriculum = True  # 是否继续之前的课程计划


class Runner:
    def __init__(self, env, device='cpu'):
        """
        初始化 Runner 类的实例。

        参数:
        - env: 强化学习环境。
        - device: 运算设备，默认为 'cpu'。
        """
        from .ppo import PPO  # 1. 从当前模块相对路径下的 ppo 模块中导入 PPO 类

        self.device = device  # 2. 设置要使用的计算设备，例如 'cpu' 或 'cuda' (即 GPU)
        self.env = env  # 2. 引用传入的环境对象

        # 3. 创建 ActorCritic 模型，并将其配置和移动到指定的计算设备
        actor_critic = ActorCritic(
            3,
            self.env.num_obs,
            self.env.num_privileged_obs,
            self.env.num_obs_history,
            self.env.num_acts_history,
            self.env.num_actions
        ).to(self.device)

        # 4. 根据 RunnerArgs.resume 判断是否需要从预训练权重恢复模型
        if RunnerArgs.resume:
            from ml_logger import ML_Logger  # 导入日志工具类
            # 5. 设置 ML_Logger 实例用于下载权重
            loader = ML_Logger(root="http://escher.csail.mit.edu:8080",
                               prefix=RunnerArgs.resume_path)
            weights = loader.load_torch("checkpoints/ac_weights_last.pt")  # 加载最后保存的模型权重
            actor_critic.load_state_dict(state_dict=weights)  # 将加载的权重应用到模型中

            # 5. 如果环境具备课程学习能力并在 RunnerArgs 中设置恢复课程学习状态，则加载相关状态
            if hasattr(self.env, "curricula") and RunnerArgs.resume_curriculum:
                distributions = loader.load_pkl("curriculum/distribution.pkl")  # 加载课程分布数据
                distribution_last = distributions[-1]["distribution"]  # 选取最后一个分布记录
                # 提取分布名，并为环境中的每一种步态赋权重
                gait_names = [key[8:] if key.startswith("weights_") else None for key in distribution_last.keys()]
                for gait_id, gait_name in enumerate(self.env.category_names):
                    self.env.curricula[gait_id].weights = distribution_last[f"weights_{gait_name}"]
                    print(gait_name)

        self.alg = PPO(actor_critic, device=self.device)  # 6. 初化化 PPO 算法实例并指定设备
        self.num_steps_per_env = RunnerArgs.num_steps_per_env  # 设置每个环境要进行的步骤数

        # 7. 初始化存储，用于保存不同环境的状态、行动等数据
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history],
                              [self.env.num_acts_history], [self.env.num_actions])

        # 8. 初始化计数相关的变量
        self.tot_timesteps = 0  # 总时间步数
        self.tot_time = 0  # 总时间
        self.current_learning_iteration = 0  # 当前学习迭代次数
        self.last_recording_it = 0  # 上一次记录的迭代次数

        self.env.reset()  # 9. 重置环境状态至初始状态

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500,
              eval_expert=False, cfg=None):
        """
        执行学习算法的迭代过程。

        参数:
        - num_learning_iterations: 学习迭代的次数。
        - init_at_random_ep_len: 是否在随机的episode长度上初始化环境，默认为False。
        - eval_freq: 评估的频率。
        - curriculum_dump_freq: 课程信息转储的频率。
        - eval_expert: 是否使用专家进行评估，默认为False。

        功能:
        - 初始化训练过程，并进行迭代学习。
        - 随机初始化环境的episode长度（如果设置为True）。
        - 在训练过程中定期进行评估，并转储课程信息。
        - 更新模型并记录学习过程中的数据。

        注解:
        1. 从ml_logger中导入logger工具，并确保其已正确配置prefix。
        2. 启动logger来跟踪学习过程中的关键信息。
        3. 初始化环境的episode长度（如果设置了该选项）。
        4. 获取环境中的所有观察值，并将其移动到指定计算设备上。
        5. 配置actor_critic模型为训练模式。
        6. 初始化用于追踪平均回报和平均长度的缓冲区。
        7. 计算总迭代次数并开始主循环。
        8. 迭代过程中，收集训练数据并按指定频率进行评估和信息转储。
        9. 在每次迭代末尾，更新模型并记录各项指标。
        10. 根据指定间隔，保存模型状态字典和日志信息。

        注意事项:
        - 学习过程中，追踪的数据会定期记录和评估，确保数据的准确性和完整性。
        - 检查点保存及其他日志记录操作应确保同步执行，以免造成数据的丢失或不一致。
        """
        from ml_logger import logger  # 导入ml_logger模块的logger

        # 检查logger的prefix是否已经设定，以确保不会覆盖instrument server上的数据
        assert logger.prefix, "you will overwrite the entire instrument server"

        # 启动logger，记录诸如'epoch', 'episode', 'run', 'step'等指标
        logger.start('start', 'epoch', 'episode', 'run', 'step')

        # 如果开启了在随机episode长度初始化，则将环境的episode长度设为随机值
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # 获取训练环境的数量
        num_train_envs = self.env.num_train_envs

        # 获取初始观察值，包括普通观察值、特权观察值和历史观察值
        obs_dict = self.env.get_observations()  # 获取环境的观察值字典
        obs, privileged_obs, obs_history, acts_history = (obs_dict["obs"], obs_dict["privileged_obs"],
                                                          obs_dict["obs_history"], obs_dict["acts_history"])
        # 将这些观察值移到当前设备上
        obs, privileged_obs, obs_history, acts_history = (obs.to(self.device), privileged_obs.to(self.device),
                                                          obs_history.to(self.device), acts_history.to(self.device))

        # 将actor_critic模型设置为训练模式
        self.alg.actor_critic.train()

        # 初始化用于记录回报和episode长度的缓冲区
        rewbuffer = deque(maxlen=100)  # 记录最近的100次回报
        lenbuffer = deque(maxlen=100)  # 记录最近的100次episode长度
        rewbuffer_eval = deque(maxlen=100)  # 用于评估的回报缓冲区
        lenbuffer_eval = deque(maxlen=100)  # 用于评估的长度缓冲区

        # 初始化记录当前回报总和和当前episode长度的张量
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 计算总的迭代次数
        tot_iter = self.current_learning_iteration + num_learning_iterations  # 当前学习迭代次数 + 指定的学习迭代次数

        # 主循环：通过迭代学习过程进行训练
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()  # 记录循环开始时间

            # 开始不计算梯度的数据收集模式
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):  # 在每个环境中执行指定数量的步骤
                    # 生成训练用和评估用动作
                    actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs], acts_history[:num_train_envs],
                                                 num_train_envs, cfg.normalization.clip_actions)
                    # if eval_expert:
                    #     actions_eval = self.alg.actor_critic.act_teacher(obs_history[num_train_envs:],
                    #                                                      privileged_obs[num_train_envs:])
                    # else:
                    #     actions_eval = self.alg.actor_critic.act_student(obs_history[num_train_envs:],
                    #                                                      acts_history[num_train_envs:])
                    #
                    # # 执行动作并收集环境的响应
                    # ret = self.env.step(torch.cat((actions_train, actions_eval), dim=0))
                    ret = self.env.step(actions_train)
                    obs_dict, rewards, dones, infos = ret  # 解包环境返回的数据

                    # 更新观察值、奖励和完成状态
                    obs, privileged_obs, obs_history, acts_history = (obs_dict["obs"], obs_dict["privileged_obs"],
                                                                      obs_dict["obs_history"], obs_dict["acts_history"])
                    obs, privileged_obs, obs_history, acts_history, rewards, dones = \
                        (obs.to(self.device), privileged_obs.to(self.device), obs_history.to(self.device),
                         acts_history.to(self.device), rewards.to(self.device), dones.to(self.device))

                    # 处理每一步在环境中采取的动作，以及它们的后果
                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)

                    # 如果有训练相关的信息，则记录它们
                    if 'train/episode' in infos:
                        with logger.Prefix(metrics="train/episode"):  # 设置日志前缀，方便分类
                            logger.store_metrics(**infos['train/episode'])

                    # 如果有评估相关的信息，也进行记录
                    if 'eval/episode' in infos:
                        with logger.Prefix(metrics="eval/episode"):  # 设置日志前缀，方便分类
                            logger.store_metrics(**infos['eval/episode'])

                    # 检查课程相关的信息，并进行相应的处理
                    if 'curriculum' in infos:  # 如果课程信息存在
                        cur_reward_sum += rewards  # 更新当前回合的奖励总和
                        cur_episode_length += 1  # 更新当前回合的长度

                        new_ids = (dones > 0).nonzero(as_tuple=False)  # 识别那些环境已完成的索引

                        # 对于完成的训练环境，记录它们的奖励和长度，然后重置
                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0  # 重置已完成实例的累计奖励
                        cur_episode_length[new_ids_train] = 0  # 重置已完成实例的累计长度

                        # 对于完成的评估环境，记录奖励和长度，然后重置
                        new_ids_eval = new_ids[new_ids >= num_train_envs]
                        rewbuffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())
                        lenbuffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_eval] = 0  # 重置已完成实例的累计奖励
                        cur_episode_length[new_ids_eval] = 0  # 重置已完成实例的累计长度

                    # 如果有课程分布信息，在这里处理
                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']

                # 记录数据收集所消耗的时间
                stop = time.time()
                collection_time = stop - start  # 计算数据收集过程的时间

                # 开始算法的学习步骤
                start = stop  # 重置开始时间，准备下一个学习步骤
                # 计算回报，这是学习算法优化的关键部分
                self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])

                # 如果达到课程信息保存的间隔，将当前信息保存到文件
                if it % curriculum_dump_freq == 0:
                    logger.save_pkl({  # 保存课程信息到.pkl文件
                        "iteration": it,
                        **caches.slot_cache.get_summary(),
                        **caches.dist_cache.get_summary()
                    }, path=f"curriculum/info.pkl", append=True)

                    # 如果有分布信息，也保存
                    if 'curriculum/distribution' in infos:
                        logger.save_pkl({"iteration": it,
                                         "distribution": distribution},
                                        path=f"curriculum/distribution.pkl", append=True)

            # 更新算法并获取损失函数的值
            mean_value_loss,  mean_surrogate_loss, mean_adaptation_module_loss, \
                mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, \
                mean_decoder_test_loss, mean_decoder_test_loss_student, mean_student_module_loss, \
                mean_student_module_test_loss = self.alg.update()

            # 记录学习步骤的结束时间
            stop = time.time()
            learn_time = stop - start  # 计算学习步骤耗费的时间

            # 存储和记录各种学习过程中的度量指标
            logger.store_metrics(
                total_time=learn_time - collection_time,  # 可记录学习阶段和数据收集阶段的总耗时
                time_elapsed=logger.since('start'),  # 计算从算法开始到当前的总耗时
                time_iter=logger.split('epoch'),  # 当前迭代所花费的时间
                adaptation_loss=mean_adaptation_module_loss,  # 记录适应模块的平均损失
                student_loss=mean_student_module_loss,
                mean_value_loss=mean_value_loss,  # 记录价值的平均损失
                mean_surrogate_loss=mean_surrogate_loss,  # 记录替代模型的平均损失
                mean_decoder_loss=mean_decoder_loss,  # 记录解码器的平均损失
                mean_decoder_loss_student=mean_decoder_loss_student,  # 记录学生解码器的平均损失
                mean_decoder_test_loss=mean_decoder_test_loss,  # 记录测试解码器的平均损失
                mean_decoder_test_loss_student=mean_decoder_test_loss_student,  # 记录学生测试解码器的平均损失
                mean_adaptation_module_test_loss=mean_adaptation_module_test_loss,  # 记录测试时适应模块的平均损失
                mean_student_module_loss=mean_student_module_loss,
                mean_student_module_test_loss=mean_student_module_test_loss
            )

            # 如果设置了保存视频的间隔，则调用视频记录函数
            if RunnerArgs.save_video_interval:
                self.log_video(it)

            # 更新总步骤数
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs  # 将当前迭代的所有环境步骤加到总步骤数中

            # 按照给定的频率记录日志
            if logger.every(RunnerArgs.log_freq, "iteration", start_on=1):
                # 记录关键性能指标，如总步数和迭代次数
                logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
                logger.job_running()  # 更新任务的运行状态

            # 检查是否达到了模型保存的间隔
            if it % RunnerArgs.save_interval == 0:
                with logger.Sync():  # 同步操作，确保操作的原子性
                    # 保存actor_critic的权重到检查点
                    logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
                    # 保存当前权重为最新权重
                    logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

                    # 确保保存路径存在
                    path = './tmp/legged_data'
                    os.makedirs(path, exist_ok=True)

                    # 保存适应模块到本地，并上传至日志记录系统
                    adaptation_module_path = f'{path}/adaptation_module_latest.jit'
                    adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                    traced_script_adaptation_module = torch.jit.script(adaptation_module)
                    traced_script_adaptation_module.save(adaptation_module_path)
                    logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)

                    # 保存主体模型到本地，并上传至日志记录系统
                    teacher_body_path = f'{path}/teacher_body_latest.jit'
                    body_model = copy.deepcopy(self.alg.actor_critic.teacher_actor_body).to('cpu')
                    traced_script_body_module = torch.jit.script(body_model)
                    traced_script_body_module.save(teacher_body_path)
                    logger.upload_file(file_path=teacher_body_path, target_path=f"checkpoints/", once=False)

                    student_body_path = f'{path}/student_body_latest.jit'
                    body_model = copy.deepcopy(self.alg.actor_critic.student_actor_body).to('cpu')
                    traced_script_body_module = torch.jit.script(body_model)
                    traced_script_body_module.save(student_body_path)
                    logger.upload_file(file_path=student_body_path, target_path=f"checkpoints/", once=False)

            # 更新当前学习迭代的计数器
            self.current_learning_iteration += num_learning_iterations

        # 确保所有日志操作是同步进行的，以保证数据的一致性
        with logger.Sync():
            # 使用PyTorch的功能来保存算法中的actor_critic部分的权重
            logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
            # 创建当前权重的副本，作为最新权重的参照
            logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

            # 确保用于存储模型数据的文件夹存在，如果不存在则创建
            path = './tmp/legged_data'
            os.makedirs(path, exist_ok=True)  # 创建路径，如果路径已存在则忽略

            # 为适应模块设置保存路径，并将适应模块的状态保存到该路径
            adaptation_module_path = f'{path}/adaptation_module_latest.jit'
            adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
            # 将适应模块通过Torch JIT静态编译，增加其执行速度
            traced_script_adaptation_module = torch.jit.script(adaptation_module)
            # 保存编译后的适应模块
            traced_script_adaptation_module.save(adaptation_module_path)

            # 为主体模型设置保存路径，并保存其状态
            teacher_body_path = f'{path}/body_latest.jit'
            body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
            # 使用Torch JIT对主体模型进行静态编译
            traced_script_body_module = torch.jit.script(body_model)
            # 保存编译后的主体模型
            traced_script_body_module.save(teacher_body_path)

            # 为主体模型设置保存路径，并保存其状态
            student_body_path = f'{path}/body_latest.jit'
            body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
            # 使用Torch JIT对主体模型进行静态编译
            traced_script_body_module = torch.jit.script(body_model)
            # 保存编译后的主体模型
            traced_script_body_module.save(student_body_path)

            # 上传已保存的适应模块和主体模型至日志记录系统的指定位置
            logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
            logger.upload_file(file_path=teacher_body_path, target_path=f"checkpoints/", once=False)
            logger.upload_file(file_path=student_body_path, target_path=f"checkpoints/", once=False)

    def log_video(self, it):
        """
        记录和保存环境的视频

        Args:
        - it: 当前的迭代次数

        注解:
        1. 检查当前迭代次数与上次记录视频的迭代次数的差值是否达到了设定的保存视频的间隔。
        2. 如果达到间隔，开始记录主环境的视频，并且如果有评估环境，也开始记录评估环境的视频。
        3. 更新最后记录视频的迭代次数。
        4. 获取并检查主环境的完整帧是否存在，如果存在，则暂停录制并保存视频。
        5. 如果有评估环境，也执行相同的帧获取、暂停和保存操作。
        6. 使用fps相等于1除以环境的步长(`env.dt`)来保存视频以确保视频速率和实际环境时间相匹配。
        """
        # 检查是否达到了记录视频的间隔
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()  # 开始记录主环境的视频
            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()  # 如果有评估环境，也开始记录评估环境的视频
            print("START RECORDING")
            self.last_recording_it = it  # 更新最后记录视频的迭代次数

        # 获取主环境的完整帧
        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()  # 暂停记录视频
            print("LOGGING VIDEO")
            # 保存视频文件
            logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)

        # 对于评估环境执行类似操作
        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()
            if len(frames) > 0:
                self.env.pause_recording_eval()  # 暂停记录评估环境的视频
                print("LOGGING EVAL VIDEO")
                # 保存评估环境的视频文件
                logger.save_video(frames, f"videos/{it:05d}_eval.mp4", fps=1 / self.env.dt)
