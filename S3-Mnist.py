import libs.mnist as mnist
import pickle
import numpy as np


# 批处理的大小
batch_size = 100


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


# 获取测试数据集
def get_data():
    # 读取训练图片、标签，测试图片、标签，并把图像数值归一化
    train_img, train_label, test_img, test_label = mnist.load_mnist(normalize=True, flatten=True)
    return test_img, test_label


# 初始化网络权重
def init_network():
    # 载入神经网络的权重
    with open("./dataset/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


# 预测X的值
def predict(network, X):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['b1'], network['b2'], network['b3']

    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid_function(A1)
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid_function(A2)
    A3 = np.dot(Z2, W3) + B3
    Y = softmax_function(A3)

    return Y


if __name__ == '__main__':
    # 获取测试集
    test_img, test_label = get_data()
    # 初始化网络
    network = init_network()

    # 识别精度计数
    accuracy_cnt = 0

    # 循环进行预测，创建一个大小为100的步阶循环
    for i in range(0, len(test_img), batch_size):
        # 创建大小为100的批处理数据集
        test_img_batch = test_img[i : i + batch_size]
        # 预测一批实例
        y_batch = predict(network, test_img_batch)
        # 沿着一维方向获取准确度最高的索引
        p = np.argmax(y_batch, axis=1)
        # 批量处理如果预测标签和真实标签相同，则数量+1
        accuracy_cnt += np.sum(p == test_label[i : i + batch_size])

    # 获得精度
    print("Accuracy:" + str(float(accuracy_cnt) / len(test_img)))
