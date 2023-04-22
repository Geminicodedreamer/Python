# 载入套件
import numpy as np
import matplotlib.pyplot as plt

# 向量(Vector)
v = np.array([2, 1])

# 作图
plt.axis('equal')
plt.grid()

# 原点
origin = [0], [0]

# 画有箭头的线
plt.quiver(*origin, *v, scale=10, color='r')

plt.xticks(np.arange(-0.05, 0.06, 0.01), labels=np.arange(-5, 6, 1))
plt.yticks(np.arange(-3, 5, 1) / 100, labels=np.arange(-3, 5, 1))
plt.show()
