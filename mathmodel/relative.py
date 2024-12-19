# 进行关联度的分析
import pandas as pd
import numpy as np
x=pd.read_csv('C:\\Users\\DELL\\Desktop\\vscodePython\\mathmodel\\edudata.csv')
x=x.iloc[:,0:].T
print("导入的数据\n",x)

# 将x数据进行独热编码或标签编码，将其转换为数值特征，保持其类型不变，只有值的类型变成数字，x中有数字和字符串
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range(x.shape[1]):
    # 检查列的数据类型
    if(x.iloc[:, i].dtype != "string"):
        x.iloc[:, i] = x.iloc[:, i].astype(str)
    x.iloc[:,i] = le.fit_transform(x.iloc[:,i])
print("编码后的数据\n",x)


# 1、数据均值化处理，进行无量纲化（消除单位不统一的因素）
x_mean=x.mean(axis=1)# 求出每一行的均值
for i in range(x.index.size):# x.index.size = 4
    x.iloc[i,:] = x.iloc[i,:]/x_mean[i]
print("均值化处理后的结果\n",x)
# 2、提取参考队列和比较队列
ck=x.iloc[16,:]
print("参考队列-即母序列\n",ck)
cp=x.iloc[0:16:]
print("比较队列-即子序列\n",cp)
 
# 比较队列与参考队列相减
t=pd.DataFrame()# 创建一个空表
for j in range(cp.index.size):# cp.index.size = 3
    temp=pd.Series(cp.iloc[j,:]-ck)
    t=t.append(temp,ignore_index=True)
print("做差以后的数据",t)
#求最大差和最小差
mmax=t.abs().max()
print("mmax:\n",mmax)
mmin=t.abs().min().min()
print("mmin:",mmin)
rho=0.5
 
 
#3、求关联系数
ksi=((mmin+rho*mmax)/(abs(t)+rho*mmax))
print("关联系数ksi:\n",ksi)
 
 
#4、求关联度
r=ksi.sum(axis=1)/ksi.columns.size
print("关联度r\n",r)
 
#5、关联度排序，得到结果r3>r2>r1
result=r.sort_values(ascending=False)
print("排序结果：\n",result)
 
 
 