import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

A = np.eye(5)

### 单变量的线性回归

# Plotting the Data
path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
# plt.show()

# 梯度下降


def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:, :-1]
y = data.iloc[:, cols - 1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))

# 计算代价函数（theta初始值为0），答案为32.07
J = computeCost(X, y, theta)

# 学习速率α和要执行的迭代次数
alpha = 0.01
iters = 1500

g, cost = gradientDescent(X, y, theta, alpha, iters)

predict1 = [1, 3.5] * g.T
predict2 = [1, 7] * g.T

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

# 原始数据以及拟合的直线
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
# plt.show()


### 多变量的线性回归

path = 'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

# 特征归一化
data2 = (data2 - data2.mean()) / data2.std()

# 梯度下降
data2.insert(0, 'Ones', 1)

cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols-1]
y2 = data2.iloc[:, cols-1:cols]

X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0, 0, 0]))

g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)



