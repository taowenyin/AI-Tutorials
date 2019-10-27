import numpy as np
import matplotlib.pylab as plt
import libs.mnist as mnist
from libs.functions import sigmoid_function
from libs.functions import softmax_function
from libs.functions import cross_entropy_error
from libs.functions import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, out_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['B1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, out_size)
        self.params['B2'] = np.zeros(out_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        B1, B2 = self.params['B1'], self.params['B2']

        A1 = np.dot(x, W1) + B1
        Z1 = sigmoid_function(A1)
        A2 = np.dot(Z1, W2) + B2
        y = softmax_function(A2)

        return y

    # 计算损失值，x为要预测对象，t为基准
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_w = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grads['B1'] = numerical_gradient(loss_w, self.params['B1'])
        grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grads['B2'] = numerical_gradient(loss_w, self.params['B2'])

        return grads

if __name__ == '__main__':
    network = TwoLayerNet(input_size=784, hidden_size=100, out_size=10)

    # 读取训练图片、标签，测试图片、标签，并把图像数值归一化
    train_img, train_label, test_img, test_label = mnist.load_mnist(normalize=True, flatten=True)

    print(train_img.shape)
    print(train_label.shape)

    # 保存训练中的损失变化
    train_lost_list = []

    iters_num = 10000
    # 训练数据的大小
    train_size = train_img.shape[0]
    # 每次训练的数量
    batch_size = 100
    # 学习率
    learning_rate = 0.1

    for i in range(iters_num):
        # 随机抽取100个训练数据的索引
        batch_mask = np.random.choice(train_size, batch_size)
        # 获取随机抽取的训练图片和对应标签
        img_batch = train_img[batch_mask]
        label_batch = train_label[batch_mask]

        # 计算梯度
        grad = network.numerical_gradient(img_batch, label_batch)

        # 更新参数
        for key in ('W1', 'B1', 'W2', 'B2'):
            network.params[key] -= learning_rate * grad[key]

        # 计算损失函数
        loss = network.loss(img_batch, label_batch)
        # 记录损失值
        train_lost_list.append(loss)

    x = np.arange(0, 11000, 1)
    y = train_lost_list
    plt.plot(x, y)
    # 设置Y轴的显示范围
    plt.ylim(0, 10)
    plt.show()