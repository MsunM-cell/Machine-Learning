import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from skimage import io
from sklearn.cluster import KMeans

def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i, :] - centroids[j, :]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    
    return idx

data = loadmat('ex7data2.mat')
X = data['X']
init_centroids = np.array([[3, 3], [6, 2], [8, 5]])

idx = find_closest_centroids(X, init_centroids)
# print(idx)

data2 = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
# sb.set(context='notebook', style='white')
# sb.lmplot('X1', 'X2', data=data2, fit_reg=False)
# plt.show()

def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()
    
    return centroids

centroids = compute_centroids(X, idx, 3)

# 核心：将样本分配给最近的簇并重新计算簇的聚类中心
def run_k_means(X, init_centroids, max_iters):
    m, n = X.shape
    k = init_centroids.shape[0]
    idx = np.zeros(m)
    centroids = init_centroids

    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
    
    return idx, centroids

idx, centroids = run_k_means(X, init_centroids, 10)

# 绘图
cluster1 = X[np.where(idx == 0)[0], :]
cluster2 = X[np.where(idx == 1)[0], :]
cluster3 = X[np.where(idx == 2)[0], :]

# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster 1')
# ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
# ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
# ax.legend()
# plt.show()

# 初始化聚类中心：随机选择样本
def random_init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i, :] = X[idx[i], :]
    
    return centroids

### 压缩图像
# image_data = loadmat('bird_small.mat')
# print(image_data)

# A = image_data['A']
# normalize value ranges
# A = A / 255.

# reshape the array
# X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))

# randomly initialize the centroids
# init_centroids = random_init_centroids(X, 16)

# run the algorithm
# idx, centroids = run_k_means(X, init_centroids, 10)

# get the closest centroids one last time
# idx = find_closest_centroids(X, centroids)

# map each pixel to the centroid value
# X_recovered = centroids[idx.astype(int), :]

# reshape to the original dimensions
# X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
# plt.imshow(X_recovered)
# plt.show()

# scikit-image
pic = io.imread('bird_small.png') / 255.
# io.imshow(pic)
# plt.show()

# serialize data
data = pic.reshape(128 * 128, 3)

# n_jobs input argument has been deprecated after 0.23 version
# Currently, it uses all cores by default.
model = KMeans(n_clusters=16, n_init=100)
model.fit(data)
centroids = model.cluster_centers_

C = model.predict(data)
compressed_pic = centroids[C].reshape((128,128,3))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()