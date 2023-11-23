import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics
iris = datasets.load_iris()
X = iris.data[:, :4]
print(X.shape)
label_true = iris.target
# 绘制数据分布图
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()



worker = KMeans(n_clusters=3)  # 构造聚类器
worker.fit(X)  # 聚类
label_pred = worker.labels_  # 获取聚类标签

x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]


plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='class0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='x', label='class1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='*', label='class2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()

plt.scatter(x0[:, 0], x0[:, 2], c="red", marker='o', label='class0')
plt.scatter(x1[:, 0], x1[:, 2], c="green", marker='x', label='class1')
plt.scatter(x2[:, 0], x2[:, 2], c="blue", marker='*', label='class2')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc=2)
plt.show()

plt.scatter(x0[:, 0], x0[:, 3], c="red", marker='o', label='class0')
plt.scatter(x1[:, 0], x1[:, 3], c="green", marker='x', label='class1')
plt.scatter(x2[:, 0], x2[:, 3], c="blue", marker='*', label='class2')
plt.xlabel('sepal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()

plt.scatter(x0[:, 1], x0[:, 2], c="red", marker='o', label='class0')
plt.scatter(x1[:, 1], x1[:, 2], c="green", marker='x', label='class1')
plt.scatter(x2[:, 1], x2[:, 2], c="blue", marker='*', label='class2')
plt.xlabel('sepal width')
plt.ylabel('petal length')
plt.legend(loc=2)
plt.show()

plt.scatter(x0[:, 1], x0[:, 3], c="red", marker='o', label='class0')
plt.scatter(x1[:, 1], x1[:, 3], c="green", marker='x', label='class1')
plt.scatter(x2[:, 1], x2[:, 3], c="blue", marker='*', label='class2')
plt.xlabel('sepal width')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()

plt.scatter(x0[:, 2], x0[:, 3], c="red", marker='o', label='class0')
plt.scatter(x1[:, 2], x1[:, 3], c="green", marker='x', label='class1')
plt.scatter(x2[:, 2], x2[:, 3], c="blue", marker='*', label='class2')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()


ARI = metrics.adjusted_rand_score(label_true, label_pred)
# 轮廓系数
B = metrics.silhouette_score(X, label_pred, metric='euclidean')
homogeneity = metrics.homogeneity_completeness_v_measure(label_true, label_pred)
print('ARI:', ARI)
print('silhouette_score:', B)
print('(homogeneity,completeness,V_measure):', homogeneity)
