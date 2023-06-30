import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import  train_test_split
data=pd.read_csv('第三题数据集/000777.csv',encoding='gbk',parse_dates=[0],index_col=0)
data.sort_index(axis=0,ascending=True,inplace=True)

dayfeature=150
featurenum=5*dayfeature
x=np.zeros((data.shape[0]-dayfeature,featurenum+1))
y=np.zeros((data.shape[0]-dayfeature))


for i in range(0,data.shape[0]-dayfeature):
   x[i,0:featurenum]=np.array(data[i:i+dayfeature]\
[[u'收盘价',u'最高价',u'最低价',u'开盘价',u'成交量']]).reshape((1,featurenum))
#将数据中的“收盘价”“最高价”“开盘价”“成交量”存入x数组中 x[i,featurenum]=data.ix[i+dayfeature][u'开盘价']
#最后一列记录当日的开盘价
for i in range(0,data.shape[0]-dayfeature):
    if data.iloc[i+dayfeature][u'收盘价']>=data.iloc[i+dayfeature][u'开盘价']: y[i]=1
else:
    y[i]=0

clf=svm.SVC(kernel='rbf')
  #调用svm函数，并设置kernel参数，默认是rbf，其它：linear’‘poly’‘sigmoid’
result = []
for i in range(5):
   x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
#x和y的验证集和测试集，切分80-20%的测试集
clf.fit(x_train, y_train)
#训练数据进行训练
result.append(np.mean(y_test == clf.predict(x_test)))
#将预测数据和测试集的验证数据比对
print("svm classifier accuacy:")

print(result)