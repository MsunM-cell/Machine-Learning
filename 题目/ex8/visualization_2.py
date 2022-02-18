import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

data = loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']

mean_Y_1 = Y[1, np.where(R[1, :] == 1)[0]].mean()
print(mean_Y_1)

fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(Y)
ax.set_xlabel('Users')
ax.set_ylabel('Movies')
fig.tight_layout()
plt.show()
