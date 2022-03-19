# 导入包
import numpy as np
# 导入伯努利模型
from sklearn.naive_bayes import BernoulliNB

# 数据集X的特征有三个，分别是
# Walks like a duck
# Talks like a duck
# Is small]
# 这三个特征分别有两种分布，是or否
# Walks like a duck: 0 = False, 1 = True
# Talks like a duck: 0 = False, 1 = True
# Is small: 0 = False, 1 = True

# 创建训练集
X = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]])
# 给训练集创建标签
# 是鸭子or不是鸭子
y = np.array(['Duck', 'Not a Duck', 'Not a Duck'])

# 使用伯努利模型训练数据
clf = BernoulliNB()
# 训练数据集
clf.fit(X, y)

# 预测数据集
# 比如我们试一下 三个特征都为true的时候，到底是不是鸭子
print(clf.predict([[1, 1, 1]]))
