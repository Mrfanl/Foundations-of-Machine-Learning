import numpy as np
import matplotlib.pyplot as plt

n = 1000  # 训练的数据点个数

X = np.random.randn(2, n) * 10
# Q = 2.5 * X[0] + 2 * X[1] + 2
X[1] = (2 - 2.5 * X[0]) / 2
pr = np.random.uniform(-5, 5, (1, n)) * 10
pr[pr > 0] = pr[pr > 0] + np.random.uniform(2, 5);
pr[pr < 0] = pr[pr < 0] - np.random.uniform(2, 5);
X = X - pr
Y = np.array(pr > 0, dtype=np.int)
plt.scatter(x=X[0][Y[0] > 0], y=X[1][Y[0] > 0], color='r')
plt.scatter(x=X[0][Y[0] <= 0], y=X[1][Y[0] <= 0], color='b')

lr = 0.05
w0 = np.random.randn()
w1 = np.random.randn()
b = 0

for i in range(n):
    if Y[0][i] * (w0 * X[0][i] + w1 * X[1][i] + 2) <= 0:
        w0 = w0 + lr * Y[0][i] * X[0][i]
        w1 = w1 + lr * Y[0][i] * X[1][i]
        b = b + lr * Y[0][i]
print("W:[%f,%f]" % (w0, w1))
print("b:[%f]" % b)
X_ = (b - w0 * X[0]) / w1
plt.plot(X[0],X_,color='y')
plt.show()