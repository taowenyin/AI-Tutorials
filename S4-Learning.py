import numpy as np
import libs.mnist as mnist
import matplotlib.pylab as plt
import libs.functions as function


# 批处理大小
batch_size = 10


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

    # 获取数据集的个数
    batch_size = predict_y.shape[0]

    return -np.sum(label * np.log(predict_y + 1e-7)) / batch_size


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


# 实例需优化函数
def example_function(x):
    return x[0]**2 + x[1]**2


class SimpleNet:
    def __init__(self):
        # 生成一个服从高斯分布的2行3列随机矩阵
        self.W = np.random.randn(2, 3)

    # 实现简单预测，没有偏置b
    def predict(self, x):
        return np.dot(x, self.W)

    # 计算损失函数
    def loss(self, x, t):
        # 对x进行预测
        z = self.predict(x)
        # 经过softmax激活函数
        y = function.softmax_function(z)
        # 与标准数据进行对比，计算损失值
        loss = cross_entropy_error(y, t)
        return loss


if __name__ == '__main__':
    # 获得训练集和测试集
    train_img, train_label, test_img, test_label = mnist.load_mnist(normalize=True, one_hot_label=True)

    # 获得训练数据的大小
    train_size = train_img.shape[0]

    # 从训练集中随机获取10个元素的索引
    batch_mask = np.random.choice(train_size, batch_size)

    # 获得随机的批处理数据
    train_img_batch = train_img[batch_mask]
    train_label_batch = train_label[batch_mask]

    print(train_img.shape)
    print(train_label.shape)

    # 初始化函数参数值
    init_x = np.array([-3.0, 4.0])
    # 获取梯度下降后的值和下降过程值
    x, x_history = gradient_descent(example_function, init_x=init_x, lr=0.1, step_num=100)

    # 绘制一条x从-5至5，y为0的蓝色虚线直线
    plt.plot([-5, 5], [0, 0], '--b')
    # 绘制一条x为0，y为-5至5的蓝色虚线直线
    plt.plot([0, 0], [-5, 5], '--b')
    # 读取所有行的第一个数据和所有行的第二个数据
    plt.plot(x_history[:, 0], x_history[:, 1], 'o')

    # 设置X轴和Y轴的范围
    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()

    # 初始化网络
    net = SimpleNet()
    # 打印生成的权重值
    print(net.W)
    # 获取预测值
    p = net.predict(np.array([0.6, 0.9]))
    print(p)
    # 生成基准数据集，并初始化为0数组
    t = np.zeros(p.shape, dtype=int)
    # 把预测概率最大的基准数据设为1
    for idx in range(t.shape[0]):
        t[np.argmax(p)] = 1
    print(t)

    # 定义一个简单函数
    f = lambda w:net.loss(x, t)
    dw = numerical_gradient(f, net.W)
    print(dw)