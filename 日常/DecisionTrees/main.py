# 导入相关的包

# 导入画决策树图的包
import graphviz
import itertools
import random

# 导入机器学习sklearn的包
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import OneHotEncoder

# 为每个特征创建不同的特征值作为分类
classes = {
    "supplies": ["low", "med", "high"],
    "weather": ["raining", "cloudy", "sunny"],
    "worked?": ["yes", "no"]
}

# 创建数据集
data = [
    ['low', 'sunny', 'yes'],
    ['high', 'sunny', 'yes'],
    ['med', 'cloudy', 'yes'],
    ['low', 'raining', 'yes'],
    ['low', 'cloudy', 'no'],
    ['high', 'sunny', 'no'],
    ['high', 'raining', 'no'],
    ['med', 'cloudy', 'yes'],
    ['low', 'raining', 'yes'],
    ['low', 'raining', 'no'],
    ['med', 'sunny', 'no'],
    ['high', 'sunny', 'yes']
]

# 创建不同的决策结果
target = ['yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes', 'yes', 'no']

# sklearn无法直接处理字符串类型的数据，所以要将数据重新编码
# 采取onehot编码
# One-Hot编码，又称为一位有效编码，主要是采用N位状态寄存器来对N个状态进行编码，
# 每个状态都有他独立的寄存器位，并且在任意时候只有一位有效。
# One-Hot编码是分类变量作为二进制向量的表示。
# 这首先要求将分类值映射到整数值。然后，每个整数值被表示为二进制向量
# 除了整数的索引之外，它都是零值，它被标记为1

# 将分类转换成onehot编码
categories = [classes['supplies'], classes['weather'], classes['worked?']]
encoder = OneHotEncoder(categories=categories)
# 用onehot编码转换数据集
x_data = encoder.fit_transform(data)

# 用给定的数据集训练决策树的模型
classifier = DecisionTreeClassifier()
tree = classifier.fit(x_data, target)

prediction_data = []

# 随机创建五条三种特征的条件
for _ in itertools.repeat(None, 5):
    prediction_data.append([
        random.choice(classes['supplies']),
        random.choice(classes['weather']),
        random.choice(classes['worked?'])
    ])

# 运用决策树模型，预测最终结果
prediction_results = tree.predict(encoder.transform(prediction_data))

# 输出结果
# print(prediction_results)

feature_names = (
        ['supplies-' + x for x in classes["supplies"]] +
        ['weather-' + x for x in classes["weather"]] +
        ['worked-' + x for x in classes["worked?"]]
)

# 使用graphviz展示决策树的可视化
dot_data = export_graphviz(tree, filled=True, proportion=True, feature_names=feature_names)
graph = graphviz.Source(dot_data)
graph.render(filename='decision_tree', cleanup=True, view=True)


# 以下将结果格式化输出，方便阅读
# 仅供参考
def format_array(arr):
    return "".join(["| {:<10}".format(item) for item in arr])


def print_table(data, results):
    line = "day  " + format_array(list(classes.keys()) + ["went shopping?"])
    print("-" * len(line))
    print(line)
    print("-" * len(line))

    for day, row in enumerate(data):
        print("{:<5}".format(day + 1) + format_array(row + [results[day]]))
    print("")


# 训练集数据
print("训练数据:")
print_table(data, target)

# 预测结果
print("随机数据预测结果:")
print_table(prediction_data, prediction_results)
