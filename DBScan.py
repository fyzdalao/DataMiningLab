import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn import metrics
iris = datasets.load_iris()
X = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度
label_true = iris.target
print(X.shape)
# 绘制数据分布图
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()

worker = DBSCAN(eps=0.4, min_samples=9)
worker.fit(X)
label_pred = worker.labels_

x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]

# this should be in a func

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



# ARI 相似度计算
ARI = metrics.adjusted_rand_score(label_true, label_pred)
# 轮廓系数
B = metrics.silhouette_score(X, label_pred, metric='euclidean')
# 同质分析
homogeneity = metrics.homogeneity_completeness_v_measure(label_true,label_pred)
print('ARI:',ARI)
print('silhouette_score:',B)
print('(homogeneity,completeness,V_measure):',homogeneity)
