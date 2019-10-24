import libs.mnist as mnist
import numpy as np

if __name__ == '__main__':
    # 读取训练图片、标签，测试图片、标签
    train_img, train_label, test_img, test_label = mnist.load_mnist()

    print(train_img.shape)
    print(train_label.shape)
    print(test_img.shape)
    print(test_label.shape)