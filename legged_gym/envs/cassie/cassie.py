from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

class Cassie(LeggedRobot):
    def _reward_no_fly(self):
        """
        计算并返回Cassie机器人的“不飞行”奖励

        返回值:
        - single_contact: 一个表示机器人是否只有一个脚与地面接触的奖励值

        注解:
        1. 通过检查机器人的脚部接触力是否大于0.1来确定哪些脚与地面接触。
        2. 计算接触地面的脚的数量，如果只有一个脚接触地面，则返回1作为奖励，否则返回0。

        注释:
        - 这个函数是用来鼓励机器人在行走时始终保持至少一个脚与地面接触，以避免“飞行”状态。
        """
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1  # 检查每个脚部的Z轴接触力是否大于0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1  # 计算每一行（每个时间步）中接触地面的脚的数量，如果正好为1，则single_contact为True
        return 1.*single_contact  # 如果只有一个脚接触地面，返回1作为奖励；否则返回0
