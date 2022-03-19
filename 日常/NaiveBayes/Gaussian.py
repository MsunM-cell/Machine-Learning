# 导入相应的包
import numpy as np
# 导入高斯模型
from sklearn.naive_bayes import GaussianNB

# 样本X包含三个特征，分别是Red的百分比，Green的百分比，Blue的百分比
# 每个特征的值都是（0，1）之间的小数

# 首先我们创建一个训练集
X = np.array([[.5, 0, .5], [1, 1, 0], [0, 0, 0]])
# 给定我们训练集的分类标签
y = np.array(['Purple', 'Yellow', 'Black'])

# 运用高斯模型去训练数据
clf = GaussianNB()
# 训练数据集
clf.fit(X, y)

# 下面我们运用我们的模型进行测试
# 比如我们试一下，red 0.5，green 0.5，blue 0.5
print(clf.predict([[0.5, 0.5, 0.5]]))

