import numpy as np

# 做fun函数的一个简单的梯度下降算法
def fun(x):
    return np.sum(x ** 2)


def gradinet_function(f, init_x, lr=0.01, step_num=100000):
    for i in range(step_num):  # 循环100次
        for idx in range(init_x.size):
            grad = gradinet(init_x, f)
            init_x -= lr * grad
    return init_x


def gradinet(init_x, f):
    grad = np.zeros_like(init_x)
    h = 1e-10
    for idx in range(init_x.size):
        tmp_val = init_x[idx]
        init_x[idx] = tmp_val + h
        fxh1 = f(init_x)
        init_x[idx] = tmp_val - h
        fxh2 = f(init_x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        init_x[idx] = tmp_val
    return grad


if __name__ == '__main__':
    print(gradinet_function(fun, np.array([-3.0, 4.0]), 0.1))
