from numpy import *
import matplotlib.pyplot as plt
from sklearn import svm


# 加载数据集
def loadDataSet(filename):
    dataMat = [];
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return array(dataMat), array(labelMat)


data, target = loadDataSet("testSet.txt")
index1 = where(target == 1)
X1 = data[index1]
index2 = where(target == -1)
X2 = data[index2]
# 二维空间画图
plt.plot(X1[:, 0], X1[:, 1], 'ro')
plt.plot(X2[:, 0], X2[:, 1], 'bx')
plt.show()
# # 调用非线性支持向量机
clf = svm.SVC(kernel='linear')
clf.fit(data, target)
w = clf.coef_[0]  # 获取w
a = -w[0] / w[1]  # 斜率

print("W:", w)
print("a:", a)
print("support_vectors_:", clf.support_vectors_)
print("clf.coef_:", clf.coef_)
# 画图划线
xx = linspace(0, 10)  # (0,10)之间x的值
yy = a * xx - (clf.intercept_[0]) / w[1]  # xx带入y，截距
# 画出与点相切的线
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])
plt.figure(figsize=(8, 4))
plt.plot(xx, yy)
plt.plot(xx, yy_down)
plt.plot(xx, yy_up)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80)
plt.scatter(data[:, 0], data[:, 1], c=target, cmap=plt.cm.Paired)  # [:，0]列切片，第0列
plt.axis('tight')
plt.show()