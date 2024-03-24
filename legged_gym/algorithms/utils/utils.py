# License: see [LICENSE, LICENSES/rsl_rl/LICENSE]

import torch


def split_and_pad_trajectories(tensor, dones):
    """
    分割并填充轨迹

    参数:
    - tensor: 包含多个环境多个时间步的数据的张量
    - dones: 指示每个时间步是否完成的布尔张量

    返回:
    - padded_trajectories: 填充后的轨迹张量
    - trajectory_masks: 表示有效轨迹部分的掩码张量

    说明:
    - 此函数处理强化学习中的轨迹数据，以便用于RNN。
    - 首先在完成的时间步处分割轨迹，然后连接并用零填充到最长轨迹的长度。
    - 返回与轨迹的有效部分相对应的掩码，以帮助RNN区分不同轨迹。

    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                f]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]

    """
    dones = dones.clone()
    dones[-1] = 1  # 确保最后一个时间步被标记为完成

    # 调整数据形状以便正确处理
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # 计算每条轨迹的长度
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()

    # 提取并填充各个轨迹
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)

    # 创建掩码张量
    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """
    将轨迹数据的填充和分割进行逆操作

    参数:
    - trajectories: 填充和分割后的轨迹数据
    - masks: 用于指示有效数据位置的掩码

    返回:
    - 逆操作后的轨迹数据

    说明:
    - 该函数首先对轨迹进行转置，然后应用掩码，最后再次转置以还原原始形状。
    """

    # 对轨迹进行转置以使其维度对应掩码的维度
    transposed_trajectories = trajectories.transpose(1, 0)

    # 应用掩码来筛选出有效数据
    masked_trajectories = transposed_trajectories[masks.transpose(1, 0)]

    # 重新调整形状并再次转置以还原原始形状
    reshaped_trajectories = masked_trajectories.view(-1, trajectories.shape[0], trajectories.shape[-1])
    return reshaped_trajectories.transpose(1, 0)
