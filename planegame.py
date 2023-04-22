# import math
# print("保留两位小数，pi={:.2f}".format(math.pi))
# print("保留两位小数，应宽10，填充星号，右对齐，pi={0:{sign}=10.2f}".format(math.pi, sign='*'))
# print("科学计数法表示，pi={:e}".format(math.pi))
# print("保留2位小数，百分比表示,pi={:.2%}".format(math.pi))

m_str = "JanFebMarAprMayJunJulAugSepOctNovDec"
n = int(input())
print(m_str[(n - 1)*3: n*3])
