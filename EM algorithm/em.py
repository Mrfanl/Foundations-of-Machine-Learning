import numpy as np
import matplotlib.pyplot as plt

mu1, std1 = 1, 2
mu2, std2 = 10, 4
N = 1000
sim1 = np.random.normal(mu1, std1, 400)
sim2 = np.random.normal(mu2, std2, 600)
sim = np.append(sim1, sim2)
np.random.permutation(sim)  # 将两个数组合并并对数据打乱
print(sim.shape)

# a1 = 0.4 u1 = 1,std1 = 2
# a2 = 0.6 u2 =10 std2 = 4

# 设置初值
a1, a2 = 0.5, 0.5
u1, u2 = 4, 6
std1, std2 = 4, 6


def gaussian(u, std, x):
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-np.square(x - u) / (2 * np.square(std)))

max_iter = 200
A1 = []
A2 = []
U1 = []
U2 = []
STD1 = []
STD2 = []
for j in range(max_iter):
    gau_sum = a1 * gaussian(u1, std1, sim) + a2 * gaussian(u2, std2, sim)
    gamma_1 = a1 * gaussian(u1, std1, sim) / gau_sum
    gamma_2 = a2 * gaussian(u2, std2, sim) / gau_sum

    # 更新参数a
    a1 = np.sum(gamma_1) / N
    a2 = np.sum(gamma_2) / N
    u_tmp1 = u1
    u_tmp2 = u2
    # 更新参数u
    u1 = np.dot(gamma_1, sim.transpose()) / np.sum(gamma_1)
    u2 = np.dot(gamma_2, sim.transpose()) / np.sum(gamma_2)
    # 更新参数stdd
    std1 = np.sqrt(np.dot(gamma_1, np.square(sim - u_tmp1)) / np.sum(gamma_1))
    std2 = np.sqrt(np.dot(gamma_2, np.square(sim - u_tmp2)) / np.sum(gamma_2))
    # 记录变化的参数值
    A1.append(a1)
    A2.append(a2)
    U1.append(u1)
    U2.append(u2)
    STD1.append(std1)
    STD2.append(std2)

print(a1, a2, u1, u2, std1, std2)
fig = plt.figure()
plt.subplot(1, 3, 1)
plt.plot(A1, label='a1')
plt.plot(A2, label='a2')
plt.subplot(1, 3, 2)
plt.plot(U1, label='u1')
plt.plot(U2, label='u2')
plt.subplot(1, 3, 3)
plt.plot(STD1, label='std1')
plt.plot(STD2, label='std2')
plt.legend()
plt.show()
