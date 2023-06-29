import math
from sklearn import preprocessing as spp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl  # 作图显示中文

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
mpl.rcParams['axes.unicode_minus'] = False


# 从文件导入数据
def load_data(file):
    dataframe = pd.read_csv(file)
    data = dataframe.values
    x, y = np.split(data, [-1], axis=1)
    return x, y


# 分离训练集和测试集
def divide(x, y):
    x_1, x_2 = np.split(x, [math.floor(0.8 * len(x))], axis=0)
    y_1, y_2 = np.split(y, [math.floor(0.8 * len(y))], axis=0)
    return x_1, y_1, x_2, y_2


# 增加一列1
def trans(x):
    t = np.ones(len(x)).reshape(-1, 1)
    data = np.append(x, t, axis=1)
    return data


# 梯度下降
def gradient(x, y, learn_rate=0.2, iter_times=10000, error=1e-8):
    x = trans(x)
    w = np.zeros((x.shape[1], 1))
    cost_function = []
    for i in range(iter_times):
        h_predict = np.dot(x, w)
        cost = np.sum((h_predict - y) ** 2) / len(x)
        cost_function.append(cost)
        dj_dw = 2 * np.dot(x.T, (h_predict - y)) / len(x)
        w = w - learn_rate * dj_dw
        if len(cost_function) > 1:
            if 0 < cost_function[-2] - cost_function[-1] < error:
                break
    return w, cost_function


# 计算预测值
def predict(x, weights):
    x = trans(x)
    return np.dot(x, weights)


# 计算误差
def get_R_2(y, h):
    sum_error = np.sum(((y - np.mean(y)) ** 2))
    inexplicable = np.sum(((y - h) ** 2))
    return 1 - inexplicable / sum_error


def figure(title, *datalist):
    for jj in datalist:
        plt.plot(jj[0], '-', label=jj[1], linewidth=2)
        plt.plot(jj[0], 'o')
    plt.grid()
    plt.title(title)
    plt.legend()
    plt.show()


# 规范化函数，最小最大规范化
def normalize(x, y):
    standard = spp.MinMaxScaler()
    x_ = standard.fit_transform(x)
    y_ = standard.fit_transform(y)
    return x_, y_


if __name__ == '__main__':
    filename = '第二题数据/Boston.csv'
    attribute, value = load_data(filename)
    x_train, y_train, x_test, y_test = divide(attribute, value)
    x_train, y_train = normalize(x_train, y_train)
    x_test, y_test = normalize(x_test, y_test)
    w, c = gradient(x_train, y_train)
    print(w)
    test_predict = predict(x_test, w)
    train_predict = predict(x_train, w)
    # 绘制误差图
    figure('误差图 最终的MSE = %.4f' % (c[-1]), [c[:], 'error'])

    # 绘制预测值与真实值图
    figure('预测值与真实值图 模型的' + r'$R^2=%.4f$' % (get_R_2(y_train, train_predict)), [test_predict, '预测值'],
           [y_test, '真实值'])
    plt.show()

    # 线性回归的参数
    print('线性回归的系数为:\n w = %s, \nb= %s' % (w[:-1], w[-1]))