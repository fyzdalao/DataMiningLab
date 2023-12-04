import matplotlib.pyplot as plt


class MyPlot():
    def __init__(self, x0, x1, x2):
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2

    def work(self):

        plt.scatter(self.x0[:, 0], self.x0[:, 1], c="red", marker='o', label='class0')
        plt.scatter(self.x1[:, 0], self.x1[:, 1], c="green", marker='x', label='class1')
        plt.scatter(self.x2[:, 0], self.x2[:, 1], c="blue", marker='*', label='class2')
        plt.xlabel('sepal length')
        plt.ylabel('sepal width')
        plt.legend(loc=2)
        plt.show()

        plt.scatter(self.x0[:, 0], self.x0[:, 2], c="red", marker='o', label='class0')
        plt.scatter(self.x1[:, 0], self.x1[:, 2], c="green", marker='x', label='class1')
        plt.scatter(self.x2[:, 0], self.x2[:, 2], c="blue", marker='*', label='class2')
        plt.xlabel('sepal length')
        plt.ylabel('petal length')
        plt.legend(loc=2)
        plt.show()

        plt.scatter(self.x0[:, 0], self.x0[:, 3], c="red", marker='o', label='class0')
        plt.scatter(self.x1[:, 0], self.x1[:, 3], c="green", marker='x', label='class1')
        plt.scatter(self.x2[:, 0], self.x2[:, 3], c="blue", marker='*', label='class2')
        plt.xlabel('sepal length')
        plt.ylabel('petal width')
        plt.legend(loc=2)
        plt.show()

        plt.scatter(self.x0[:, 1], self.x0[:, 2], c="red", marker='o', label='class0')
        plt.scatter(self.x1[:, 1], self.x1[:, 2], c="green", marker='x', label='class1')
        plt.scatter(self.x2[:, 1], self.x2[:, 2], c="blue", marker='*', label='class2')
        plt.xlabel('sepal width')
        plt.ylabel('petal length')
        plt.legend(loc=2)
        plt.show()

        plt.scatter(self.x0[:, 1], self.x0[:, 3], c="red", marker='o', label='class0')
        plt.scatter(self.x1[:, 1], self.x1[:, 3], c="green", marker='x', label='class1')
        plt.scatter(self.x2[:, 1], self.x2[:, 3], c="blue", marker='*', label='class2')
        plt.xlabel('sepal width')
        plt.ylabel('petal width')
        plt.legend(loc=2)
        plt.show()

        plt.scatter(self.x0[:, 2], self.x0[:, 3], c="red", marker='o', label='class0')
        plt.scatter(self.x1[:, 2], self.x1[:, 3], c="green", marker='x', label='class1')
        plt.scatter(self.x2[:, 2], self.x2[:, 3], c="blue", marker='*', label='class2')
        plt.xlabel('petal length')
        plt.ylabel('petal width')
        plt.legend(loc=2)
        plt.show()
