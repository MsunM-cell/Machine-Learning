import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'admitted'])

# data visualization
positive = data[data['admitted'].isin([1])]
negative = data[data['admitted'].isin([0])]

# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', marker='o', label='admitted')
# ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', marker='x', label='not admitted')

# ax.legend()
# ax.set_xlabel('exam1')
# ax.set_ylabel('exam2')
# plt.show()

# sigmoid


def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

# cost function


def costFunction(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    cost = np.multiply(y, np.log(sigmoid(X * theta.T))) + \
        np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return -(np.sum(cost) / len(X))

# gradient (not update theta)


def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad


data.insert(0, 'x0', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

X = np.array(X)
y = np.array(y)
theta = np.zeros(3)

J = costFunction(theta, X, y)
print(J)
grad = gradient(theta, X, y)

result = opt.fmin_tnc(costFunction, theta, gradient, (X, y))

# plot decision curve
# plotting_x = np.linspace(30, 100, 100)
# plotting_y = (-result[0][0] - result[0][1] * plotting_x) / result[0][2]

# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(plotting_x, plotting_y, c='y', label='prediction')
# ax.scatter(positive['exam1'], positive['exam2'], c='b', s=50, marker='o', label='admitted')
# ax.scatter(negative['exam1'], negative['exam2'], c='r', s=50, marker='x', label='not admitted')
# ax.legend()
# ax.set_xlabel('exam1')
# ax.set_ylabel('exam2')
# plt.show()

# calculation model


def hfunc1(theta, X):
    return sigmoid(np.dot(theta, X))


hfunc1(result[0], [1, 45, 85])

# prediction on training set


def predict(theta, X):
    theta = np.matrix(theta)
    X = np.matrix(X)
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


theta_opt = result[0]
predictions = predict(theta_opt, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0))
           else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct) / len(correct) * 100
print('accuracy = {0}%'.format(accuracy))
