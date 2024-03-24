import os
import gymnasium as gym
import numpy as np
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置环境变量，指定使用的GPU设备编号为0
print('CUDA version:', torch.version.cuda)  # 打印当前使用的CUDA版本
print('CUDA available:', torch.cuda.is_available())  # 检查并打印CUDA是否可用

local_path = os.path.dirname(__file__)  # 获取当前文件的路径
log_path = os.path.join(local_path, 'log')  # 拼接得到日志文件的路径


class ModelBase(object):

    def __init__(self, env: gym.Env, args: argparse.Namespace):
        """
        ModelBase类是所有模型的基类，包含了模型的一些基本属性和方法。

        Args:
        - env: gym环境对象，用于模型的训练和评估。
        - args: argparse.Namespace对象，包含了模型的配置参数。

        Attributes:
        - args: argparse.Namespace对象，包含了模型的配置参数。
        - env: gym环境对象，用于模型的训练。
        - env_evaluate: gym环境对象，用于模型的评估。
        - agent: 代理对象，用于执行在环境中的动作。在子类中具体实现。
        - model_name: 字符串，模型的名称。在子类中具体定义。
        - state_norm: 状态归一化对象，用于将环境状态归一化。在子类中具体实现。
        """
        self.args = args  # 存储模型的配置参数
        self.env = env  # 存储用于模型训练的环境
        self.env_evaluate = env  # 存储用于模型评估的环境
        self.set_seed(args.seed)  # 设置随机种子

        self.agent = None  # 初始化代理对象，具体实现在子类中
        self.model_name = None  # 初始化模型名称，具体定义在子类中
        self.state_norm = None  # 初始化状态归一化对象，具体实现在子类中

    def evaluate_policy(self):
        """
        评估当前策略的性能。

        Returns:
        - avg_reward: float，评估过程中的平均奖励。

        注解:
        1. 设定评估次数为3次。
        2. 对每次评估，首先重置环境并获取初始状态。
        3. 如果使用了状态归一化，则对初始状态进行归一化处理。
        4. 在环境中执行策略，直到环境终止或达到最大步数。
        5. 在执行策略的过程中，使用确定性策略（即不带噪声的策略）进行动作选择。
        6. 执行动作并获取新的状态、奖励以及环境是否终止的信息。
        7. 如果使用了状态归一化，则对新的状态进行归一化处理。
        8. 累计每次评估的奖励。
        9. 返回平均奖励。

        注意事项:
        - 在评估过程中，状态归一化的更新选项应设为False。
        - 在评估过程中，应使用确定性策略进行动作选择。
        """
        times = 3  # 设定评估次数
        evaluate_reward = 0  # 初始化评估奖励
        for _ in range(times):  # 对每次评估
            s, _ = self.env_evaluate.reset(seed=self.args.seed)  # 重置环境并获取初始状态
            if self.args.use_state_norm:  # 如果使用了状态归一化
                s = self.state_norm(s, update=False)  # 对初始状态进行归一化处理，注意更新选项设为False
            episode_reward = 0  # 初始化每次评估的奖励
            while True:  # 在环境中执行策略，直到环境终止或达到最大步数
                action = self.agent.sample_action(s, deterministic=True)  # 使用确定性策略进行动作选择
                s_, r, terminated, truncated, _ = self.env_evaluate.step(action)  # 执行动作并获取新的状态、奖励以及环境是否终止的信息
                if self.args.use_state_norm:  # 如果使用了状态归一化
                    s_ = self.state_norm(s_, update=False)  # 对新的状态进行归一化处理，注意更新选项设为False
                episode_reward += r  # 累计奖励
                s = s_  # 更新状态
                if terminated or truncated:  # 如果环境终止或达到最大步数
                    break  # 结束当前评估
            evaluate_reward += episode_reward  # 累计每次评估的奖励

        return evaluate_reward / times  # 返回平均奖励

    def set_seed(self, seed=10):
        """
        设置随机种子。

        Args:
        - seed: int，默认为10，用于设置随机种子。

        注解:
        1. 如果seed为0，则不进行任何操作。
        2. 否则，将seed设置到环境的动作空间、numpy随机数生成器、PyTorch的CPU和GPU随机数生成器以及Python的哈希函数中。

        注意事项:
        - 设置随机种子可以保证实验的可复现性。
        """
        if seed == 0:  # 如果seed为0
            return  # 不进行任何操作
        self.env.action_space.seed(seed)  # 将seed设置到环境的动作空间
        self.env_evaluate.action_space.seed(seed)  # 将seed设置到评估环境的动作空间
        np.random.seed(seed)  # 将seed设置到numpy随机数生成器
        torch.manual_seed(seed)  # 将seed设置到PyTorch的CPU随机数生成器
        torch.cuda.manual_seed(seed)  # 将seed设置到PyTorch的GPU随机数生成器
        os.environ['PYTHONHASHSEED'] = str(seed)  # 将seed设置到Python的哈希函数

    def train(self):
        """
        训练模型。

        注解:
        1. 初始化训练步数、评估次数、采样计数以及评估奖励列表。
        2. 配置Tensorboard的日志路径。
        3. 在达到最大训练步数之前，进行以下操作：
            - 重置环境并获取初始状态。
            - 在每个回合中，进行以下操作：
                - 选择动作并执行，获取新的状态、奖励以及环境是否终止的信息。
                - 如果达到最大回合步数，则设定环境为终止状态。
                - 将状态、动作、奖励以及环境是否终止的信息存储到代理的记忆中。
                - 更新状态和总训练步数。
                - 每隔一定步数，更新代理的策略。
                - 每隔一定步数，进行一次策略评估，并将评估奖励存储到评估奖励列表中。
                - 每隔一定次数，保存评估奖励列表到文件中。
            - 如果环境终止或达到最大回合步数，则结束当前回合并开始新的回合。
        4. 训练结束后，关闭环境。

        注意事项:
        - 在训练过程中，应定期更新代理的策略，并进行策略评估。
        - 在策略评估过程中，应将评估奖励存储到评估奖励列表中，并定期保存到文件中。
        """
        print("开始训练！")

        total_steps = 0  # 初始化训练步数
        evaluate_num = 0  # 初始化评估次数
        sample_count = 0  # 初始化采样计数
        evaluate_rewards = []  # 初始化评估奖励列表

        # 配置Tensorboard的日志路径
        log_dir = os.path.join(log_path, f'./runs/{self.model_name}')
        writer = SummaryWriter(log_dir=log_dir)

        while total_steps < self.args.max_train_steps:  # 在达到最大训练步数之前
            s, _ = self.env.reset(seed=self.args.seed)  # 重置环境并获取初始状态
            ep_step = 0  # 初始化回合步数
            while True:  # 在每个回合中
                ep_step += 1  # 更新回合步数
                sample_count += 1  # 更新采样计数
                a, a_logprob = self.agent.sample_action(s)  # 选择动作
                s_, r, terminated, truncated, _ = self.env.step(a)  # 执行动作并获取新的状态、奖励以及环境是否终止的信息

                if ep_step == self.args.max_episode_steps:  # 如果达到最大回合步数
                    truncated = True  # 设定环境为终止状态

                # 将状态、动作、奖励以及环境是否终止的信息存储到代理的记忆中
                self.agent.memory.push((s, a, a_logprob, s_, r, terminated, truncated))
                s = s_  # 更新状态
                total_steps += 1  # 更新总训练步数

                # 每隔一定步数，更新代理的策略
                if sample_count % self.args.buffer_size == 0:
                    self.agent.update(total_steps)

                if total_steps % self.args.evaluate_freq == 0:  # 每隔一定步数
                    evaluate_num += 1  # 更新评估次数
                    evaluate_reward = self.evaluate_policy()  # 进行一次策略评估
                    evaluate_rewards.append(evaluate_reward)  # 将评估奖励存储到评估奖励列表中
                    print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                    writer.add_scalar('step_rewards_{}'.format(self.args.env_name), evaluate_rewards[-1],
                                      global_step=total_steps)  # 将评估奖励写入Tensorboard日志
                    # 每隔一定次数，保存评估奖励列表到文件中
                    if evaluate_num % self.args.save_freq == 0:
                        model_dir = os.path.join(log_path, f'./data_train/{self.model_name}.npy')
                        np.save(model_dir, np.array(evaluate_rewards))

                if terminated or truncated:  # 如果环境终止或达到最大回合步数
                    break  # 结束当前回合

        print("完成训练！")
        self.env.close()  # 训练结束后，关闭环境
