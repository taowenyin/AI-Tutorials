import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# 产生0~6之间，以0.1为步阶的数据
x = np.arange(0, 6, 0.1)
# 产生对应的sin数据
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制图表，设置图例
plt.plot(x, y1, label="sin")
plt.plot(x, y2, label="cos", linestyle="--")
# 设置X、Y轴的名字
plt.xlabel("x")
plt.ylabel("y")
# 图表标题
plt.title('sin & cos')
# 显示图例
plt.legend()
plt.show()

# 读取图片
img = imread('images/Lena.png')
# 图片显示
plt.imshow(img)
plt.show()