import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

df = pd.read_csv('weather.csv')

# print(df)
# print(df.dtypes)
# print(df.info())

df_dummy = pd.get_dummies(data=df, columns=['Temperature', 'Outlook', 'Windy'])
# print(df_dummy)

X = df_dummy.drop('Played?', axis=1)
y = df_dummy['Played?']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

dtree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)
# print(predictions)

fig = plt.figure(figsize=(16, 12))
a = plot_tree(dtree, feature_names=df_dummy.columns, fontsize=12, filled=True,
              class_names=['Not Play', 'Play'])
plt.show()
