# 建立工程并导入sklearn包
import numpy as np
from os import listdir #使用listdir模块,用于访问本地文件
from sklearn import neighbors

# 加载训练数据
#定义img2vector函数，将加载的32*32的图片矩阵展开成—列向量
def txt2vector(fileName):
    retMat = np.zeros([1024],int) #定义返回的矩阵，大小为1*1024
    fr = open(fileName) #打开包含32*32大小的数字文件
    lines = fr.readlines() #读取文件的所有行
    for i in range(32): #遍历文件所有行
        for j in range(32): #并将01数字存放在retMat中
            retMat[i*32+j] = lines[i][j]
    return retMat

# 定义加载训练数据的函数readDataSet
def readDataSet(path):
    fileList = listdir(path)#获取文件夹下的所有文件
    numFiles = len(fileList)#统计需要读取的文件的数目
    dataSet = np.zeros([numFiles,1024],int)#用于存放所有的数字文件
    hwLables = np.zeros([numFiles])#用于存放对应的标签（与神经网络不同）
    for i in range(numFiles):#遍历所有的文件
        filePath = fileList[i]#获取文件路径
        digit = int(filePath.split('_')[0])#通过文件名获取标签
        hwLables[i] = digit#直接存放数字，并非one-hot向量
        dataSet[i] = txt2vector(path+'/'+filePath)#读取文件内容
    return dataSet,hwLables

# 调用readDataSet和img2vector函数加载数据,将训练的图片存放在train_dataSet中,对应的标签则存在train_hwLabels中。
tarin_dataSet,train_hwLables = readDataSet('第二题数据集/trainingDigits')

# 构建KNN分类器
knn = neighbors.KNeighborsClassifier(algorithm='kd_tree',n_neighbors=3)
knn.fit(tarin_dataSet,train_hwLables)

# 测试集评价
dataSet,hwLables = readDataSet('第二题数据集/testDigits') #加载测试集
# 使用训练好的KNN对测试集进行测试，并计算错误率
res = knn.predict(dataSet) #对测试集进行预测
error_num = np.sum(res!=hwLables)#统计分类错误的数目
num = len(dataSet) #测试集的数目
print("Total num:",num,"Wrong num:", \
      error_num,"WrongRate:",error_num/float(num))
