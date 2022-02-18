import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from scipy import stats

data = loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']

def serialize(X, theta):
    """序列化两个矩阵"""
    # X (movie, feature), (1682, 10): movie features
    # theta (user, feature), (943, 10): user preference
    return np.concatenate((X.ravel(), theta.ravel()))

def deserialize(param, n_movie, n_user, n_features):
    """逆序列化"""
    return param[:n_movie * n_features].reshape(n_movie,n_features), param[n_movie * n_features:].reshape(n_user, n_features)

# recommendation fn
def cost(param, Y, R, n_features):
    """compute cost for every r(i, j)=1
    Args:
        param: serialized X, theta
        Y (movie, user), (1682, 943): (movie, user) rating
        R (movie, user), (1682, 943): (movie, user) has rating
    """
    # X (movie, feature), (1682, 10): movie features
    # theta (user, feature), (943, 10): user preference
    n_movie, n_user = Y.shape
    X, theta = deserialize(param, n_movie, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)
    return np.power(inner, 2).sum() / 2

params_data = loadmat('ex8_movieParams.mat')
X = params_data['X']
theta = params_data['Theta']

param = serialize(X, theta)
J = cost(serialize(X, theta), Y, R, 10)

# gradient
def gradient(param, Y, R, n_features):
    # theta (user, feature), (943, 10): user preference
    # X (movie, feature), (1682, 10): movie features
    n_movies, n_user = Y.shape
    X, theta = deserialize(param, n_movies, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)

    # X_grad (1682, 10)
    X_grad = inner @ theta

    # theta_grad (943, 10)
    theta_grad = inner.T @ X

    # roll them together and return
    return serialize(X_grad, theta_grad)

n_movie, n_user = Y.shape
X_grad, theta_grad = deserialize(gradient(param, Y, R, 10),n_movie, n_user, 10)

# regularized cost & regularized gradient
def regularized_cost(param, Y, R, n_features, l=1):
    reg_term = np.power(param, 2).sum() * (l / 2)

    return cost(param, Y, R, n_features) + reg_term

def regularized_gradient(param, Y, R, n_features, l=1):
    grad = gradient(param, Y, R, n_features)
    reg_term = l * param

    return grad + reg_term

total_cost = regularized_cost(param, Y, R, 10, l=1)

movie_list = []
f = open('movie_ids.txt', encoding='gbk')

for line in f:
    tokens = line.strip().split(' ')
    movie_list.append(' '.join(tokens[1:]))

movie_list = np.array(movie_list)

ratings = np.zeros((1682, 1))

ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

Y = np.append(ratings, Y, axis=1)  # now I become user 0
R = np.append(ratings != 0, R, axis=1)

movies = Y.shape[0]  # 1682
users = Y.shape[1]  # 944
features = 10
learning_rate = 10.

X = np.random.random(size=(movies, features))
theta = np.random.random(size=(users, features))
params = serialize(X, theta)

Y_norm = Y - Y.mean()
Y_norm.mean()

# training
from scipy.optimize import minimize
fmin = minimize(fun=regularized_cost, x0=params, args=(Y_norm, R, features, learning_rate), method='TNC', jac=regularized_gradient)

X_trained, theta_trained = deserialize(fmin.x, movies, users, features)

prediction = X_trained @ theta_trained.T
my_preds = prediction[:, 0] + Y.mean()
idx = np.argsort(my_preds)[::-1]

print(my_preds[idx][:10])

for m in movie_list[idx][:10]:
    print(m)