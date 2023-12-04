import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn import metrics
from my_plot import MyPlot


iris = datasets.load_iris()
X = iris.data[:, :4]
label_true = iris.target


worker = DBSCAN(eps=0.42, min_samples=5)
worker.fit(X)
label_pred = worker.labels_


x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]


plot_worker = MyPlot(x0, x1, x2)
plot_worker.work()


homogeneity = metrics.homogeneity_completeness_v_measure(label_true, label_pred)
print('(homogeneity,completeness,V_measure):', homogeneity)
