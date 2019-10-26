import gzip
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# 数据集下载于http://yann.lecun.com/exdb/mnist/

# 数据集文件
key_files = {
    'train_img': "train-images.idx3-ubyte.gz",
    'train_label': "train-labels.idx1-ubyte.gz",
    'test_img': 't10k-images.idx3-ubyte.gz',
    'test_label': 't10k-labels.idx1-ubyte.gz'
}

# 数据集所在路径
dataset_dir = os.path.dirname(os.path.abspath(__file__)) + '/../dataset/'
# 序列化数据集的文件
save_file = dataset_dir + 'mnist.pkl'
# 根据文件格式可以知道图片大小
img_size = 784


# 读取指定文件
def _load_img(file_name):
    file_path = dataset_dir + file_name
    # 打开压缩文件
    with gzip.open(file_path, 'rb') as f:
        # 跳过前面的16个字节，直接读取后面的所有字节
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # 根据图片大小把数组分割为具有N个数组长度为784的数组，
    data = data.reshape(-1, img_size)
    return data


# 读取指定文件
def _load_label(file_name):
    file_path = dataset_dir + file_name
    # 打开压缩文件
    with gzip.open(file_path, 'rb') as f:
        # 跳过前面的8个字节，直接读取后面的所有字节，由于标签大小为1个字节因此标签不需要重新整理数据
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels


# 把数据转化为对象
def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_files['train_img'])
    dataset['train_label'] = _load_label(key_files['train_label'])
    dataset['test_img'] = _load_img(key_files['test_img'])
    dataset['test_label'] = _load_label(key_files['test_label'])
    return dataset


# mnist数据初始化
def _init_mnist():
    dataset = _convert_numpy()
    with open(save_file, 'wb') as f:
        # 序列化数据，-1表示采用二进制保存来保护数据
        pickle.dump(dataset, f, -1)


# 把标签变为0、1数据集
# 首先创建一个有10个元素的数组，并且初始化为0，
# 然后根据标签索引把0数组中的对应位置置1
def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T


# 读取显示图片
def img_show(img):
    # 图片显示
    plt.imshow(img)
    plt.show()


# 载入数据集
# normalize : 将图像的像素值正规化为0.0~1.0
# flatten : 是否将图像展开为一维数组
def load_mnist(normalize=False, flatten=True, one_hot_label=False):
    # 如果数据集文件不存在，那么就创建数据集文件
    if not os.path.exists(save_file):
        _init_mnist()

    # 载入序列化的数据集
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            # 把每个R、G、B转化为浮点数，并且归一化
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    return dataset['train_img'], dataset['train_label'], dataset['test_img'], dataset['test_label']
