import numpy as np
import matplotlib.pylab as plt


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


x = np.arange(-5, 5, 0.1)
y = step_function(x)
plt.plot(x, y)
# 设置Y轴的显示范围
plt.ylim(-0.1, 1.1)
plt.show()

x = np.arange(-5, 5, 0.1)
y = sigmoid_function(x)
plt.plot(x, y)
# 设置Y轴的显示范围
plt.ylim(-0.1, 1.1)
plt.show()

A = np.array([[1, 2], [3, 4], [5, 6]])
# 数据的纬度
print(np.ndim(A))
# 数据的形状
print(A.shape)

A = np.array([[1, 2], [3, 4]])
print(A.shape)
B = np.array([[5, 6], [7, 8]])
print(B.shape)
# 矩阵相乘
print(np.dot(A, B))

X = np.array([1, 2])
print(X.shape)
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W.shape)
Y = np.dot(X, W)
print(Y)

print("====以下为构建神经网络（A = XW + B）====")

# 创建0层神经元
X = np.array([1, 5])
# 创建0->1层权重
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
# 创建0->1层偏置
B1 = np.array([0.1, 0.2, 0.3])
print("====生成1层神经元====")
A1 = np.dot(X, W1) + B1
print(A1)
print("====1层神经元经过激活函数后得到1层神经元的输出====")
Z1 = sigmoid_function(A1)
print(Z1)

# 创建1->2层权重
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
# 创建1->2层偏置
B2 = np.array([0.1, 0.2])
print("====生成2层神经元====")
A2 = np.dot(Z1, W2) + B2
print(A2)
print("====2层神经元经过激活函数后得到2层神经元的输出====")
Z2 = sigmoid_function(A2)
print(Z2)

# 创建2->3层权重
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
# 创建2->3层偏置
B3 = np.array([0.1, 0.2])
print("====生成3层神经元====")
A3 = np.dot(Z2, W3) + B3
print("====3层神经元经过激活函数后得到2层神经元的输出====")
Y = identity_function(A3)
print(Y)


# 网络初始化
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['B1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['B2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['B3'] = np.array([0.1, 0.2])

    return network


# 将输入信号转化为输出信号
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['B1'], network['B2'], network['B3']

    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid_function(A1)
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid_function(A2)
    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3)

    return Y


# 完整的初始化网络到前向输出结果
network = init_network()
x = np.array([1, 5])
y = forward(network, x)
print(y)
print(np.sum(y))