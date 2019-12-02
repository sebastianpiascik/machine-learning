import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN

# from sklearn.datasets import load_iris
# iris = load_iris()
# x = iris.data

dataset = pd.read_csv('iris2D.csv')
x = dataset.iloc[:, [1, 2]].values

# dbscan = DBSCAN(algorithm='auto', eps=3, leaf_size=30, metric='euclidean',
#     metric_params=None, min_samples=2, n_jobs=None, p=None)
dbscan = DBSCAN(eps=0.5, metric='euclidean', min_samples=3)

y_dbscan = dbscan.fit_predict(x)
# y_dbscan_2d = y_dbscan.transform(x)

#Visualising the clusters
plt.scatter(x[y_dbscan == 0, 0], x[y_dbscan == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_dbscan == 1, 0], x[y_dbscan == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_dbscan == -1, 0], x[y_dbscan == -1, 1], s = 100, c = 'green', label = 'Iris-virginica')

print(y_dbscan)

plt.legend()
plt.show()