import math
from prettytable import PrettyTable
import pandas as pd
import numpy as np
from sklearn import preprocessing as spp
import matplotlib.pyplot as plt
from pylab import mpl  # 作图显示中文
from sklearn.metrics import auc, confusion_matrix, roc_curve

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
def gradient(x, y, learn_rate=0.05, iter_times=50000, error=1e-9):
    x_use = trans(x)
    weights = np.zeros((len(x_use[0]), 1))
    cost_function = []
    for i in range(iter_times):
        y_predict = np.dot(x_use, weights)
        h_predict = 1 / (1 + np.exp(-y_predict))
        like = np.sum(np.dot(y.T, np.log(h_predict)) + np.dot((1 - y).T, np.log(1 - h_predict)))
        cost = -like / (len(x_use))
        cost_function.append(cost)
        dj_dw = np.dot(x_use.T, (h_predict - y)) / len(x_use)
        if len(cost_function) > 2:
            if 0 <= cost_function[-2] - cost_function[-1] <= error:
                break
        weights = weights - learn_rate * dj_dw
    return weights, cost_function


# 预测值计算
def predict(x, weights, threshold=0.5):
    x_use = trans(x)
    y_predict = np.dot(x_use, weights)
    h_predict = 1 / (1 + np.exp(-y_predict))
    y_predicted = [1 if i > threshold else 0 for i in h_predict]
    return y_predicted


# 分离训练集和测试集
def divide(x, y):
    x_1, x_2 = np.split(x, [math.floor(0.8 * len(x))], axis=0)
    y_1, y_2 = np.split(y, [math.floor(0.8 * len(y))], axis=0)
    return x_1, y_1, x_2, y_2


def confusion(y, predict_y, method='Array'):
    mix = PrettyTable()
    y_num = sorted(list(set(np.unique(y))))
    mix.field_names = [method] + ['预测:%d类' % i for i in y_num]
    # 字典形式存储混淆矩阵数据
    cmdict = {}
    for i in y_num:
        cmdict[i] = []
        for j in y_num:
            hu = len(['0' for k in range(len(y)) if y[k][0] == i and predict_y[k][0] == j])
            cmdict[i].append(hu)
    # 输出表格
    for fu in y_num:
        mix.add_row(['真实:%d类' % fu] + cmdict[fu])
    return mix


# 归一化函数
def normalize(x):
    standard = spp.MinMaxScaler()
    x_ = standard.fit_transform(x)
    return x_


# 数据预处理
def pretreatment(x, y):
    pred_x = normalize(x)
    y = y - 1
    data = np.hstack((pred_x, y))
    np.random.shuffle(data)
    x, y = np.split(data, [-1], axis=1)
    return x, y


if __name__ == '__main__':
    filename = '第四题数据/Heart.csv'
    attribute, value = load_data(filename)
    attribute, value = pretreatment(attribute, value)
    attribute_train, value_train, attribute_test, value_test = divide(attribute, value)
    w, c = gradient(attribute_train, value_train)
    predict_test = predict(attribute_test, w)
    predict_test = np.array(predict_test).reshape(-1, 1)
    print('混淆矩阵：\n', confusion(value_test, predict_test))
    plt.plot(list(range(len(c))), c, '-', linewidth=5)
    plt.title('成本函数图')
    plt.ylabel('Cost 值')
    plt.xlabel('迭代次数')
    plt.show()
    fpr, tpr, thresholds = roc_curve(value_test, predict_test)
    plt.plot(fpr, tpr, marker='o')
    plt.title('ROC曲线')
    plt.show()
    AUC = auc(fpr, tpr)
    print('AUC:', AUC)