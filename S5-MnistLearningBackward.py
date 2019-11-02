import numpy as np
import matplotlib.pyplot as plt
import libs.mnist as mnist
from collections import OrderedDict
from libs.backward import Affine, Relu, SoftmaxWithLoss
from libs.functions import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['B1'] = np.zeros(hidden_size)
        self.params['W2'] =weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['B2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['B1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['B2'])

        # 生成最后一层
        self.lastLayer = SoftmaxWithLoss()

    # 通过生成层预测结果
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # 计算损失函数
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['B1'] = numerical_gradient(loss_W, self.params['B1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['B2'] = numerical_gradient(loss_W, self.params['B2'])

        return grads

    def gradient(self, x, t):
        # 前向输出
        self.loss(x, t)

        # 反向输出
        dout = 1
        dout = self.lastLayer.backward(dout)

        # 获取生成层
        layers = list(self.layers.values())
        # 反序生成层
        layers.reverse()
        for layer in layers:
            # 反向计算梯度
            dout = layer.backward(dout)

        # 获取梯度
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['B1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['B2'] = self.layers['Affine2'].db

        return grads

if __name__ == '__main__':
    network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

    # 读取训练图片、标签，测试图片、标签，并把图像数值归一化
    train_img, train_label, test_img, test_label = mnist.load_mnist(normalize=True, flatten=True)

    print(train_img.shape)
    print(train_label.shape)

    # 保存训练中的损失变化
    train_lost_list = []
    train_acc_list = []
    test_lost_list = []

    iters_num = 10000
    # 训练数据的大小
    train_size = train_img.shape[0]
    # 每次训练的数量
    batch_size = 100
    # 学习率
    learning_rate = 0.1

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        # 随机抽取100个训练数据的索引
        batch_mask = np.random.choice(train_size, batch_size)
        # 获取随机抽取的训练图片和对应标签
        img_batch = train_img[batch_mask]
        label_batch = train_label[batch_mask]

        # 计算梯度
        grad = network.gradient(img_batch, label_batch)

        # 更新参数
        for key in ('W1', 'B1', 'W2', 'B2'):
            network.params[key] -= learning_rate * grad[key]

        # 计算损失函数
        loss = network.loss(img_batch, label_batch)
        # 记录损失值
        train_lost_list.append(loss)

        if i % iter_per_epoch ==0:
            train_acc = network.accuracy(train_img, train_label)
            test_acc = network.accuracy(test_img, test_label)
            train_acc_list.append(train_acc)
            test_lost_list.append(test_acc)
            print(train_acc, test_acc)

    x = np.arange(0, 11000, 1)
    y = train_lost_list
    plt.plot(x, y)
    # 设置Y轴的显示范围
    plt.ylim(0, 10)
    plt.show()