import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics
from my_plot import MyPlot


iris = datasets.load_iris()
X = iris.data[:, :4]
label_true = iris.target


worker = KMeans(n_clusters=3)
worker.fit(X)
label_pred = worker.labels_


x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]


plot_worker = MyPlot(x0, x1, x2)
plot_worker.work()


ARI = metrics.adjusted_rand_score(label_true, label_pred)
B = metrics.silhouette_score(X, label_pred, metric='euclidean')
homogeneity = metrics.homogeneity_completeness_v_measure(label_true, label_pred)
print('ARI:', ARI)
print('silhouette_score:', B)
print('(homogeneity,completeness,V_measure):', homogeneity)
