import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

path = 'ex2data2.txt'
data_init = pd.read_csv(path, header=None, names=[
                        'Test 1', 'Test 2', 'Accepted'])

positive2 = data_init[data_init['Accepted'].isin([1])]
negative2 = data_init[data_init['Accepted'].isin([0])]

# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(positive2['Test 1'], positive2['Test 2'],
#            s=50, c='b', marker='o', label='Accepted')
# ax.scatter(negative2['Test 1'], negative2['Test 2'],
#            s=50, c='r', marker='x', label='Rejected')
# ax.legend()
# ax.set_xlabel('Test 1 Score')
# ax.set_ylabel('Test 2 Score')
# plt.show()

# feature map
degree = 6
data2 = data_init
x1 = data2['Test 1']
x2 = data2['Test 2']

data2.insert(3, 'Ones', 1)

for i in range(1, degree + 1):
    for j in range(0, i + 1):
        data2['F' + str(i - j) + str(j)] = np.power(x1,
                                                    i - j) * np.power(x2, j)

data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)


def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


def costReg(theta, X, y, learningRate):
    """cost function of regularization"""
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = -np.multiply(y, np.log(sigmoid(X * theta.T)))
    second = -np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * \
        np.sum(np.power(theta[:, 1:theta.shape[1]], 2))

    return np.sum(first + second) / len(X) + reg


def gradientReg(theta, X, y, learningRate):
    """gradient of regularization"""
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        if i == 0:
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = np.sum(term) / len(X) + \
                (learningRate / len(X)) * theta[:, i]

    return grad


cols = data2.shape[1]
X2 = data2.iloc[:, 1:cols]
y2 = data2.iloc[:, 0:1]

theta2 = np.zeros(cols-1)
X2 = np.array(X2.values)
y2 = np.array(y2.values)
# learningRate = 0 overfitting
# learningRate = 100 underfitting
learningRate = 1

J = costReg(theta2, X2, y2, learningRate)
print(J)

result2 = opt.fmin_tnc(func=costReg, x0=theta2,
                       fprime=gradientReg, args=(X2, y2, learningRate))


def predict(theta, X):
    theta = np.matrix(theta)
    X = np.matrix(X)
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


theta_min = result2[0]
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0))
           else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) / len(correct)) * 100
# print('accuracy = {0}%'.format(accuracy))


def hfunc2(theta, x1, x2):
    temp = theta[0][0]
    place = 0
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            temp += np.power(x1, i - j) * np.power(x2, j) * theta[0][place+1]
            place += 1
    return temp


def find_decision_boundary(theta):
    t1 = np.linspace(-1, 1.5, 1000)
    t2 = np.linspace(-1, 1.5, 1000)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    h_val = pd.DataFrame({'x1': x_cord, 'x2': y_cord})
    h_val['hval'] = hfunc2(theta, h_val['x1'], h_val['x2'])

    decision = h_val[np.abs(h_val['hval']) < 2 * 10 ** -3]
    return decision.x1, decision.x2


fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive2['Test 1'], positive2['Test 2'],
           s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative2['Test 1'], negative2['Test 2'],
           s=50, c='r', marker='x', label='Rejected')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')

x, y = find_decision_boundary(result2)
ax.scatter(x, y, c='y', s=10, label='Prediction')
plt.legend()
plt.show()
