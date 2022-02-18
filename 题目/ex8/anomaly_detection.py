import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from scipy import stats

data = loadmat('ex8data1.mat')
X = data['X']

# 计算高斯分布参数
def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)

    return mu, sigma

mu, sigma = estimate_gaussian(X)

# 数据可视化
# xplot = np.linspace(0, 25, 100)
# yplot = np.linspace(0, 25, 100)
# Xplot, Yplot = np.meshgrid(xplot, yplot)
# Z = np.exp((-0.5) * ((Xplot - mu[0]) ** 2 / sigma[0] + (Yplot - mu[1]) ** 2 / sigma[1]))

# fig, ax = plt.subplots(figsize=(12, 8))
# contour = plt.contour(Xplot, Yplot, Z, [10 ** -11, 10 ** -7, 10 ** -5, 10 ** -3, 0.1], colors='k')
# ax.scatter(X[:, 0], X[:, 1])
# plt.show()

# 获取验证集
Xval = data['Xval']
yval = data['yval']

# 计算正态分布的概率
pval = np.zeros((Xval.shape[0], Xval.shape[1]))
pval[:, 0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:, 0])
pval[:, 1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:, 1])

# 获取阈值
def select_threshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon
        
        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)
        
        precison = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precison * recall) / (precison + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    
    return best_epsilon, best_f1

epsilon, f1 = select_threshold(pval, yval)

# 寻找异常点
p = np.zeros((X.shape[0], X.shape[1]))
p[:, 0] = stats.norm(mu[0], sigma[0]).pdf(X[:, 0])
p[:, 1] = stats.norm(mu[1], sigma[1]).pdf(X[:, 1])
outliers = np.where(p < epsilon)

# 可视化
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
ax.scatter(X[outliers[0], 0], X[outliers[0], 1], s=50, color='r', marker='o')
plt.show()


