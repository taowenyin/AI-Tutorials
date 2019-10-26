import numpy as np


# 阶跃激活函数
def step_function(x):
    y = x > 0
    # 把boolean的数据类型转化为整型
    return y.astype(np.int)


# Sigmoid激活函数
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


# Softmax激活函数
def softmax_function(x):
    # 获取输入数据中最大值最为常量参数，避免结果超过最大值
    c = np.max(x)
    # 获取每个输入的指数函数结果
    exp_a = np.exp(x - c)
    # 获取所有指数函数结果的和
    sum_exp_a = np.sum(exp_a)
    # 获取每个指数函数结果除以指数函数结果和
    y = exp_a / sum_exp_a
    return y


# ReLU激活函数
def relu_function(x):
    return np.maximum(0, x)


# 恒等函数
def identity_function(x):
    return x


# 均方误差
def mean_squared_error(predict_y, label):
    # 如果只处理一个数据，那么也变成一个二维数据
    if predict_y.ndim == 1:
        predict_y = predict_y.reshape(1, predict_y.size)
        label = label.reshape(1, label.size)

    # 获取数据集的个数
    batch_size = predict_y.shape[0]

    return np.sum((predict_y - label) ** 2) / batch_size


# 交叉熵误差，增加delta是为了解决predict_y为0时，取对数无穷大的问题
def cross_entropy_error(predict_y, label):
    # 如果只处理一个数据，那么也变成一个二维数据
    if predict_y.ndim == 1:
        predict_y = predict_y.reshape(1, predict_y.size)
        label = label.reshape(1, label.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if label.size == predict_y.size:
        label = label.argmax(axis=1)

    # 获取数据集的个数
    batch_size = predict_y.shape[0]

    return -np.sum(np.log(predict_y[np.arange(batch_size), label] + 1e-7)) / batch_size


# 可以处理多维数据的梯度实现
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    # 生成和x形状相同的0数组
    grad = np.zeros_like(x)

    # 创建一个迭代器，该迭代器输出的为索引
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # 获取索引
        idx = it.multi_index
        # 保存当前求偏导的x值
        tmp_val = x[idx]

        # 计算f(x + h)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # 计算f(x + h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # 计算x的梯度值
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        # 还原x值
        x[idx] = tmp_val
        it.iternext()

    return grad


# 梯度下降，f表示需要优化的函数，lr表示学习率，step_num表示一次更新多少次梯度
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    x_history = []

    # 循环更新N次梯度
    for i in range(step_num):
        # 复制一份保
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        x -= (lr * grad)

    return x, np.array(x_history)