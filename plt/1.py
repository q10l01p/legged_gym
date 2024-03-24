import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 定义正态分布的累积密度函数Phi
def Phi(x, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma) ** 2)

# 定义计算C_foot_cmd的函数
def C_foot_cmd(t_foot, sigma):
    return Phi(t_foot, sigma) * (1 - Phi(t_foot - 0.5, sigma)) + Phi(t_foot - 1, sigma) * (1 - Phi(t_foot - 1.5, sigma))

# 设置标准差sigma的值
sigma = 0.1  # 标准差可以调整以查看其对函数形状的影响

# 生成t_foot值
t_foot_values = np.linspace(-0.5, 1.5, 400)

# 计算C_foot_cmd值
C_foot_cmd_values = [C_foot_cmd(t_foot, sigma) for t_foot in t_foot_values]

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(t_foot_values, C_foot_cmd_values, label='C_foot_cmd(t_foot)')
plt.title('Cumulative Density Function $C_{foot_{cmd}}(t_{foot})$')
plt.xlabel('$t_{foot}$')
plt.ylabel('$C_{foot_{cmd}}(t_{foot})$')
plt.grid(True)
plt.legend()
plt.show()
