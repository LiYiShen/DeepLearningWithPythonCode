import pandas as pd
import numpy as np
from sklearn.datasets import load_iris 
import matplotlib.pyplot as plt

# 鸢尾花数据集处理
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target # 目标值 0 1 2

df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
df.label.value_counts() # 统计各值的数量，等价于 df['label'].value_counts()

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')  # 相同表达 df['sepal length'][:50]
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

data = np.array(df.iloc[:100, [0, 1, -1]]) # 选择df数据前100行，第0,1，-1(最后)列
X, y = data[:,:-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])

# 感知机
# 数据线性可分，二分类数据
# 此处为一元一次线性方程
class Model:
    def __init__(self):
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1
        # self.data = data

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    # 随机梯度下降法
    def fit(self, X_train, y_train):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:  # 误分类点更新 w,b
                    self.w = self.w + self.l_rate * np.dot(y, X)
                    self.b = self.b + self.l_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True
        return 'Perceptron Model!'

    def score(self):
        pass


perceptron = Model()
perceptron.fit(X, y)

x_points = np.linspace(4, 7, 10)
 # 两个特征则有两个w，平面为：w1*x1 + w2*x2 + b = 0 除以w2方便绘图
y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1] 
plt.plot(x_points, y_)

plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

# scikit-learn 实例
from sklearn.linear_model import Perceptron

clf = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)  # 是否估计截距  是否每次迭代后重排数据
clf.fit(X, y)
print(clf.coef_)  # 打印 w
print(clf.intercept_)  # 打印 b

x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()