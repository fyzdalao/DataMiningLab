import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import metrics
from my_plot import MyPlot


iris = datasets.load_iris()
X = iris.data[:, :5]
label_true = iris.target


x0 = X[iris.target == 0]
x1 = X[iris.target == 1]
x2 = X[iris.target == 2]


plot_worker = MyPlot(x0, x1, x2)
plot_worker.work()


