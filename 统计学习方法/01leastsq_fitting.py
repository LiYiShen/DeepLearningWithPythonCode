# 最小二乘法拟合曲线
# 用目标函数 y=sin2πx , 加上一个正态分布的噪音干扰，用多项式去拟合

import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# 目标函数
def real_func(x):
	return np.sin(2*np.pi*x)

# 多项式
def fit_func(p, x):
	f = np.poly1d(p)
	return f(x)

# 残差error
def residuals_func(p, x, y):
	ret = fit_func(p, x) - y
	return ret

x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 1000)
y_ = real_func(x)
y = [np.random.normal(0, 0.1) + i for i in y_]

def fitting(M=0):
	# M 多项式的次数
	# 随机初始化多项式参数
	p_init = np.random.rand(M + 1)
	# 最小二乘法：通过最小化误差的平方和来寻找最佳的匹配函数。常用于曲线拟合。
	p_lsq = leastsq(residuals_func, p_init, args=(x, y))
	print('Fitting Parameters:', p_lsq[0])
	# 可视化
	plt.plot(x_points, real_func(x_points), label="real")
	plt.plot(x_points, fit_func(p_lsq[0], x_points), label="fitted curve")
	plt.plot(x, y, 'bo', label='noise')
	plt.legend()
	plt.show()
	return p_lsq

# M=0
# p_lsq_0 = fitting(M=0)
# p_lsq_1 = fitting(M=1)
# p_lsq_2 = fitting(M=3)
p_lsq_3 = fitting(M=9)

# 正则化
regularization = 0.0001

def residuals_func_regularization(p, x, y):
	ret = fit_func(p, x) - y
	ret = np.append(ret, np.sqrt(0.5 * regularization * np.square(p)))
	return ret  # 此为列表，包含error和正则项值

# 最小二乘法 加正则化项
p_init = np.random.rand(9 + 1)
p_lsq_regularization = leastsq(residuals_func_regularization, p_init, args=(x, y))

plt.plot(x_points, real_func(x_points), label='real')
plt.plot(x_points, fit_func(p_lsq_regularization[0], x_points), label='regularization')
plt.plot(x_points, fit_func(p_lsq_3[0], x_points), label='fitted curve'
plt.plot(x, y, 'bo', label='noise')
plt.legend()
plt.show()


## 扩展介绍：
# p = np.poly1d([2,3,5,7])  列表值为系数 
# print(p)   ==>>    2x3 + 3x2 + 5x + 7  

# q = np.poly1d([2,3,5],True)  列表值为根
# print(q)   ==>>    (x - 2)*(x - 3)*(x - 5) = x3 - 10x2 + 31x -30

# q = np.poly1d([2,3,5],True,varibale = 'z')  未知数字母
# print(q)   ==>>    (z - 2)*(z - 3)*(z - 5)  = z3 - 10z2 + 31z -30

# scipy.optimize.leastsq: 求出任意想要拟合的函数的参数
# leastsq(func, x0, args())
# func 定义的一个计算误差的函数
# x0 计算的初始参数值
# args 指定func的其他参数(除x0外全部打包)