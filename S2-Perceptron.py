import numpy as np


# 感知机的与门
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    # 数组运算，即把各元素对应相乘
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1


# 感知机与非门
def NAND(x1, x2):
    x = np.array([x1, x2])
    # 因为是与非门，所以取负号
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1


# 感知机或门
def OR(x1, x2):
    x = np.array([x1, x2])
    # 因为是与非门，所以取负号
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1


# 感知机异或门
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
