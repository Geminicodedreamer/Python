import matplotlib.pyplot as plt
import math
import numpy as np

# 向量
v = np.array([2, 1])
vTan = v[1] / v[0]
print('tan(0) = 1/2')

theta = math.atan(vTan)
print('弧度 = ', round(theta, 4))
print('角度 = ', round(theta*180 / math.pi, 2))
print('角度 = ', round(math.degrees(theta), 2))
