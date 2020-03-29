# 线性回归的评价指标：均方误差(MSE)、均方根误差(RMSE)、平均绝对误差(MAE)、𝑅平方（避免量纲不一致问题）

import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(1234)
x = np.random.rand(500,3)
#构建映射关系，模拟真实的数据待预测值,映射关系为y = 4.2 + 5.7*x1 + 10.8*x2
y = x.dot(np.array([4.2,5.7,10.8]))
lr = LinearRegression(fit_intercept=True)
lr.fit(x,y)
print("估计的参数值为：%s" %(lr.coef_))
# 计算R平方
print('R2:%s' %(lr.score(x,y)))

x_test = np.array([2,4,5]).reshape(1,-1)
y_hat = lr.predict(x_test)
print("预测值为: %s" %(y_hat))
