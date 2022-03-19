# 导入相关的包
import numpy as np
# 导入多项式模型
from sklearn.naive_bayes import MultinomialNB

# 我们使用文章最开始的水果的数据集作为示例
# 水果数据集的样本X具有三个特征[Size, Weight, Color]
# 每个特征共有三种分类
# 由于python不能直接识别文字，所以将这三个特征的不同分类重新编码如下
# Size: 0 = Small, 1 = Moderate, 2 = Large
# Weight: 0 = Light, 1 = Moderate, 2 = Heavy
# Color: 0 = Red, 1 = Blue, 2 = Brown

# 用编码好的数据创建训练集
X = np.array([[1, 1, 0], [0, 0, 1], [2, 2, 2]])
# 给训练集的数据创建标签
y = np.array(['Apple', 'Blueberry', 'Coconut'])

# 运用多项式模型训练数据
clf = MultinomialNB()
# 训练水果数据集
clf.fit(X, y)

# 预测数据集
# 比如我们试一下 size = 1, weight = 2, color = 0
print(clf.predict([[1, 2, 0]]))
