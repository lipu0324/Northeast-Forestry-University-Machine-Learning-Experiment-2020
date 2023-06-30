import math
from prettytable import PrettyTable
import pandas as pd
import numpy as np
from sklearn import preprocessing as spp

import matplotlib.pyplot as plt
from pylab import mpl  # 作图显示中文

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 设置中文字体新宋体
mpl.rcParams['axes.unicode_minus'] = False


# 导入数据
def load_data(file):
    dataframe = pd.read_csv(file)
    data = dataframe.values
    x, y = np.split(data, [-1], axis=1)
    return x, y


# 加一列1
def trans(x):
    return np.append(x, np.ones(len(x)).reshape(-1, 1), axis=1)


# 梯度下降
def gradient(x, y, learn_rate=0.6, iter_times=50000, error=1e-9):
    x_use = trans(x)
    weights = np.zeros((len(x_use[0]), len(y[0])))
    cost_function = []
    for i in range(iter_times):
        exp_wtx = np.exp(np.dot(x_use, weights))
        sum_exp_wtx = np.sum(exp_wtx, axis=1).reshape(-1, 1)
        h_predict = exp_wtx / sum_exp_wtx
        dj_dw = -1 / len(x_use) * np.dot(x_use.T, (y - h_predict))
        cost = -1 / len(x_use) * np.sum(y * h_predict)
        cost_function.append(cost)
        if len(cost_function) > 2:
            if 0 <= cost_function[-2] - cost_function[-1] <= error:
                break
        weights = weights - learn_rate * dj_dw
    return weights, cost_function


# 预测值计算
def predict(x, weights):
    x_use = trans(x)
    exp_wtx = np.exp(np.dot(x_use, weights))
    max_number = np.max(exp_wtx, axis=1)
    predict_y = []
    for i in range(len(max_number)):
        pos = list(exp_wtx[i]).index(max_number[i]) + 1
        predict_y.append([pos])
    return np.array(predict_y)


# 将独热编码的类别变为标识为1，2，3的类别
def transign(eydata):
    ysign = []
    for hh in eydata:
        ysign.append([list(hh).index(1) + 1])
    return np.array(ysign)


# 归一化函数
def normalize(x):
    standard = spp.MinMaxScaler()
    x_ = standard.fit_transform(x)
    return x_


# 分离训练集和测试集
def divide(x, y):
    x_1, x_2 = np.split(x, [math.floor(0.8 * len(x))], axis=0)
    y_1, y_2 = np.split(y, [math.floor(0.8 * len(y))], axis=0)
    return x_1, y_1, x_2, y_2


# 数据预处理
def pretreatment(x, y):
    pred_x = normalize(x)
    # pd.get_dummies特征提取，将原本的一列数据(三类)变成{[1 0 0], [0 1 0], [0 0 1]}三列(三类)方便使用softmax
    unique_y = pd.get_dummies(pd.DataFrame(y)).values
    data = np.hstack((pred_x, unique_y))
    np.random.shuffle(data)
    x, y = np.split(data, [-3], axis=1)
    return x, y


def confusion(realy, outy, method='AnFany'):
    mix = PrettyTable()
    type = sorted(list(set(realy.T[0])), reverse=True)
    mix.field_names = [method] + ['预测:%d类' % si for si in type]
    # 字典形式存储混淆矩阵数据
    cmdict = {}
    for jkj in type:
        cmdict[jkj] = []
        for hh in type:
            hu = len(['0' for jj in range(len(realy)) if realy[jj][0] == jkj and outy[jj][0] == hh])
            cmdict[jkj].append(hu)
    # 输出表格
    for fu in type:
        mix.add_row(['真实:%d类' % fu] + cmdict[fu])
    return mix


if __name__ == '__main__':
    filename = '第三题数据/Iris.csv'
    attribute, value = load_data(filename)
    attribute, value = pretreatment(attribute, value)
    attribute_train, value_train, attribute_test, value_test = divide(attribute, value)
    w, c = gradient(attribute_train, value_train)
    predict_test = predict(attribute_test, w)
    print(w)
    print('混淆矩阵：\n', confusion(transign(value_test), predict_test))
    plt.plot(list(range(len(c))), c, '-', linewidth=5)
    plt.title('成本函数图')
    plt.ylabel('Cost 值')
    plt.xlabel('迭代次数')
plt.show()