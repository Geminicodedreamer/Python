# 将每行数据用字典存储，再将所有字典合并成一个数组，数组下标表示第几个数据
# 数据存储在edudata.csv中
# edudata.csv部分数据如下:
# gender,NationalITy,PlaceofBirth,StageID,GradeID,SectionID,Topic,Semester,Relation,raisedhands,VisITedResources,AnnouncementsView,Discussion,ParentAnsweringSurvey,ParentschoolSatisfaction,StudentAbsenceDays,Class
# M,KW,KuwaIT,lowerlevel,G-04,A,IT,F,Father,15,16,2,20,Yes,Good,Under-7,M
# M,KW,KuwaIT,lowerlevel,G-04,A,IT,F,Father,20,20,3,25,Yes,Good,Under-7,M
# M,KW,KuwaIT,lowerlevel,G-04,A,IT,F,Father,10,7,0,30,No,Bad,Above-7,L
# M,KW,KuwaIT,lowerlevel,G-04,A,IT,F,Father,30,25,5,35,No,Bad,Above-7,L
# M,KW,KuwaIT,lowerlevel,G-04,A,IT,F,Father,40,50,12,50,No,Bad,Above-7,M
# F,KW,KuwaIT,lowerlevel,G-04,A,IT,F,Father,42,30,13,70,Yes,Bad,Above-7,M
# M,KW,KuwaIT,MiddleSchool,G-07,A,Math,F,Father,35,12,0,17,No,Bad,Above-7,L
# M,KW,KuwaIT,MiddleSchool,G-07,A,Math,F,Father,50,10,15,22,Yes,Good,Under-7,M
# F,KW,KuwaIT,MiddleSchool,G-07,A,Math,F,Father,12,21,16,50,Yes,Good,Under-7,M
# F,KW,KuwaIT,MiddleSchool,G-07,B,IT,F,Father,70,80,25,70,Yes,Good,Under-7,M
# M,KW,KuwaIT,MiddleSchool,G-07,A,Math,F,Father,50,88,30,80,Yes,Good,Under-7,H
# M,KW,KuwaIT,MiddleSchool,G-07,B,Math,F,Father,19,6,19,12,Yes,Good,Under-7,M
# M,KW,KuwaIT,lowerlevel,G-04,A,IT,F,Father,5,1,0,11,No,Bad,Above-7,L
# 用Python写！！！



import pandas as pd

# 读取数据
data = pd.read_csv('C:\\Users\\DELL\\Desktop\\vscodePython\\mathmodel\\edudata.csv')
 
# 将data中除了Class以外的数据进行独热编码或标签编码，将其转换为数值特征，存储在二维数组x[][]中，再将Class这一列数据存储在a[]中





import numpy as np
import math as mt
import matplotlib.pyplot as plt
# 1.这里我们将 a 作为我们的特征序列 x[][]作为我们的相关因素序列
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 将data中除了Class以外的数据进行独热编码或标签编码，将其转换为数值特征，存储在二维数组x[][]中
y = []
for i in range(len(data.columns) - 1):
    if data.dtypes[i] == 'object':
        labelencoder = LabelEncoder()
        y.append(labelencoder.fit_transform(data.iloc[:, i]))
    else:
        y.append(data.iloc[:, i].values)
y = list(map(list, zip(*y)))

 
# 将y中的第0、1、2、3、4、5、13、14列放入x[][]中
x = []
for i in range(len(y)):
    x.append([y[i][3], y[i][4], y[i][5], y[i][13]])

x = np.array(x).T
print("x shape:", x.shape)
print("x:", x)



# 将Class这一列数据存储在a[]中
a = data.iloc[:, -1].values
# 将a标签编码（均值处理）
for i in range(len(a)):
    if(a[i] == 'L'):
        a[i] = 69 / 2.0
    elif(a[i] == 'M'):
        a[i] = (70 + 89) / 2.0
    else:
        a[i] = (90 + 100) / 2.0

# 2.我们对其进行一次累加
def AGO(m):
    m_ago = [m[0]]
    add = m[0] + m[1]
    m_ago.append(add)
    i = 2
    while i < len(m):
        # print("a[",i,"]",a[i])
        add = add+m[i]
        # print("->",add)
        m_ago.append(add)
        i += 1
    return m_ago
 
a_ago = AGO(a)
x_ago = []
 
for i in range(len(x)):
    x_ago.append(AGO(x[i]))

xi = np.array([i for i in x_ago])

print("xi shape" ,xi.shape)
print("xi:",xi)
# 3.紧邻均值生成序列
def JingLing(m):
    Z = []
    j = 1
    while j < len(m):
        num = (m[j]+m[j-1])/2
        Z.append(num)
        j = j+1
    return Z
Z = JingLing(a_ago)
# print(Z)
 
# 4.求我们相关参数
Y = []
x_i = 0
while x_i < len(a)-1 :
    x_i += 1
    Y.append(a[x_i])
Y = np.mat(Y).T
Y.reshape(-1,1)
print("Y.shape:",Y.shape)
 
B = []
b = 0
while b < len(Z) :
    B.append(-Z[b])
    b += 1
B = np.mat(B)
B.reshape(-1,1)
B = B.T
print("B.shape:",B.shape)
X = xi[:,1:].T
print("X.shape:",X.shape)
B = np.hstack((B,X))
print("B-final:",B.shape)
 
# 可以求出我们的参数
theat = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
# print(theat)
al = theat[:1,:]
al = float(al)
# print("jhjhkjhjk",float(al))
b = theat[1:,:].T
print(b)
print("b.shape:",b.shape)
b = list(np.array(b).flatten())
 
 
# 6.生成我们的预测模型
U = []
k = 0
i = 0
# 计算驱动值
for k in range(len(x[1])):
    sum1 = 0
    for i in range(len(x)):
        sum1 += b[i] * xi[i][k]
        i += 1
    U.append(sum1)
    k += 1
print("U:",U)
# 计算完整公式的值
F = []
F.append(a[0])
 
f = 1
while f < len(a):
    F.append((a[0]-U[f-1]/al)/mt.exp(al*f)+U[f-1]/al)
    f += 1
print("F:",F)
 
# 做差序列
G = []
G.append(a[0])
g = 1
while g<len(a):
    G.append(F[g]-F[g-1])
    g +=1
print("G:",G)
 
r = range(len(x[1]))
t = list(r)
 
plt.plot(t,a,color='r',linestyle="--",label='true')
plt.plot(t,G,color='b',linestyle="--",label="predict")
plt.legend(loc='upper right')
plt.show()
 































