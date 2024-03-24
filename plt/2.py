import numpy as np
import matplotlib.pyplot as plt

# 定义正态分布的累积密度函数Phi
def phi(x, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma) ** 2)

# 定义x的值范围
x = np.linspace(-3, 3, 400)
sigma = 1  # 标准差设为1

# 计算Phi的值
phi_values = phi(x, sigma)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(x, phi_values, label=f'$\sigma$ = {sigma}')
plt.title('Normal Distribution PDF ($\Phi(x; \sigma)$)')
plt.xlabel('x')
plt.ylabel('$\Phi(x; \sigma)$')
plt.grid(True)
plt.legend()
plt.show()
