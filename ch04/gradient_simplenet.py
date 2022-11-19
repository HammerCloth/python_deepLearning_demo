# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        # 随机出来一个权重系数矩阵，随着class初始化的时候保存到w中
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        # 少了激活函数
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)  # 激活函数，主要用于分类问题
        loss = cross_entropy_error(y, t)  # 交叉墒误差

        return loss


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet() # 构建一个类

f = lambda i: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)
