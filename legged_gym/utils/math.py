import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple


#@torch.jit.script
def quat_apply_yaw(quat, vec):
    """
    对四元数应用偏航（yaw）角，并将变换应用于向量

    Args:
    - quat: 输入的四元数张量，代表旋转
    - vec: 输入的向量张量，可能代表一个三维点的坐标

    Returns:
    - 应用了偏航角变换的向量结果

    注解:
    - 此函数首先复制输入的四元数，并确保它是正确的形状。
    - 然后只保留四元数的偏航分量，设置其他分量为零。
    - 对这个修改后的四元数进行归一化，以确保它仍然代表一个有效的旋转。
    - 最后，使用这个只包含偏航分量的四元数对输入向量进行变换。

    注意:
    - `torch.jit.script`装饰器用于将这个函数编译为TorchScript，一个中间表示，可以优化运行速度并允许在没有Python解释器的环境中运行。
    - 所有插入此函数的输入应该是PyTorch的张量（tensor）。
    - 函数中的`normalize`和`quat_apply`函数需要已被定义和适用于张量操作。
    """

    quat_yaw = quat.clone().view(-1, 4)  # 复制输入四元数，并保证它的形状为(N, 4)
    quat_yaw[:, :2] = 0.  # 将xi和yj分量设置为0，保留偏航分量zk和实部w
    quat_yaw = normalize(quat_yaw)  # 归一化修改后的四元数，保证其为单位四元数
    return quat_apply(quat_yaw, vec)  # 应用包含偏航的四元数到输入向量，并返回结果向量


#@torch.jit.script
def wrap_to_pi(angles):
    """
    将角度值限制在-pi到pi的范围内

    Args:
    - angles: 一个或多个角度值的张量

    Returns:
    - 修正后的角度值张量，其值位于[-π, π]的区间

    注解:
    - 此函数首先通过取模运算限制输入角度的范围在[0, 2π]。
    - 然后减去2π从而把超过π的角度值映射到[-π, π]区间内。
    - 这种处理方式适用于周期性边界场景，如角度值的归一化。

    注意:
    - torch.jit.script装饰器使得这个函数被视为TorchScript代码，允许进行进一步优化。
    - 输入的张量类型是浮点数，这是为了确保取模的计算准确性。
    """

    angles %= 2*np.pi  # 将所有角度通过取模限制在0到2π之间
    angles -= 2*np.pi * (angles > np.pi)  # 将角度值大于π的部分转换到-π到π的范围
    return angles  # 返回处理后的角度张量


#@torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    """
    生成一个均匀分布后经过平方根转换的随机浮点数张量

    Args:
    - lower: 生成的浮点数的下界
    - upper: 生成的浮点数的上界
    - shape: 期望生成的张量形状
    - device: 生成张量所在的设备

    Returns:
    - 经过平方根转换后并且限制在指定范围内的随机浮点数张量

    注解:
    - 此函数先生成一个均匀分布在[-1, 1]区间的随机浮点数张量。
    - 然后对该张量应用条件平方根变换，正数取平方根，负数先取相反数再开平方根。
    - 将结果的取值范围再次变换到[0, 1]区间。
    - 最后对结果张量进行线性缩放，使其取值限制在用户定义的[lower, upper]范围内。

    注意:
    - 此函数通过torch.jit.script装饰器被转换为TorchScript，能在不同的环境下运行，包括没有Python环境的地方。
    - 传入的`shape`参数为二元组(int, int)，定义了输出张量的维度。
    - 返回的张量类型为Tensor，并确保其按照设备参数分配至正确的计算设备。
    """

    r = 2*torch.rand(*shape, device=device) - 1  # 生成-1到1之间的均匀分布的随机数张量
    r = torch.where(r < 0., -torch.sqrt(-r), torch.sqrt(r))  # 对张量中的每个元素应用条件平方根变换
    r = (r + 1.) / 2.  # 调整张量的数值范围至0到1之间
    return (upper - lower) * r + lower  # 将张量线性缩放至lower至upper的范围并返回结果张量


def get_scale_shift(range):
    """
    计算归一化处理所需的缩放比例和平移量。

    参数:
    - range: 一个包含两个元素的列表或元组，表示待归一化数据的最小值和最大值。

    返回值:
    - scale: 缩放比例，用于将数据归一化到一个特定的范围。
    - shift: 平移量，与缩放比例配合使用，用于调整数据的中心位置。

    主要步骤:
    1. 计算缩放比例（scale），使得数据的范围被归一化到 [-1, 1]。
    2. 计算平移量（shift），使得数据的中心点对齐到原点（0点）。
    """

    # 计算缩放比例。范围的宽度被映射到2，因此每一单位变化等价于原始数据范围宽度的1/2。
    scale = 2. / (range[1] - range[0])

    # 计算平移量。首先计算原始数据范围的中点，然后通过平移实现中点对齐到0。
    shift = (range[1] + range[0]) / 2.

    # 返回计算出的缩放比例和平移量。
    return scale, shift
