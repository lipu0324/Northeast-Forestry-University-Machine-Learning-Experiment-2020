import numpy as np
import matplotlib.pyplot as plt

#读入文本文件数据
def load_data(filename):
    xys=[]
    with open(filename,'r') as f:
        for line in f:
            xx=line.strip().split()
            xys.append(xx)

    xs=[]
    ys=[]
    size=len(xys)
    for i in range(size):
        xs.append(xys[i][0])
        ys.append(xys[i][1])
    xs=np.array(xs).astype(float)
    ys=np.array(ys).astype(float)
    return xs,ys
#在训练样本集的最后一列加1
def Trans(xdata):
    ones=np.ones(len(xdata)).reshape(-1,1)
    xta=np.append(xdata,ones,axis=1)
    return xta
#利用传统的最小二乘法求解参数，即公式 W=(XT.X)-1 *XT.Y
def MinBiMultiply(xdata,ydata):
    xdata=Trans(xdata)
    xTx=np.dot(xdata.T,xdata)
    #判断行列式是否为零
    if np.linalg.det(xTx)==0:
        print("the Matrix cannot do inverse!")
        return
    #求逆阵
    invert=np.linalg.inv(xTx)
    ws=np.dot(np.dot(invert, xdata.T),ydata)
    return ws
#利用梯度下降法
def GDS(xdata,ydata,apha=0.2,iter=1000,err=1e-4):
    xdata=Trans(xdata)
    size=len(xdata[0])
    cost_function=[]
    #print(size)
    w=np.ones((size,1))#.reshape(-1,1)
    #print(w)
    for i in range(iter):
        y_predict=np.dot(xdata,w)
        coste=np.sum((y_predict-ydata)**2)/len(ydata)
        cost_function.append(coste)
        #print(cost_function)
        #计算梯度
        dj_dw=2*np.dot(xdata.T,(y_predict-ydata))/len(xdata)
        w=w-apha*dj_dw

        # 提前结束循环的机制
        if len(cost_function) > 1:
            if 0 < cost_function[-2] - cost_function[-1] < err:
                break

    return w  #eights  #, cost_function
#梯度下降法
def Gradient(xdata1, ydata,learn_rate=0.2,iter_times=1000,error=1e-4):
    xdata = Trans(xdata1)
    #系数w,b的初始化
    print(xdata.shape)
    weights = np.zeros((xdata.shape[1],1)) #(len(xdata[1]), 1))
    #存储成本函数的值
    cost_function = []

    for i in range(iter_times):
        #得到回归的值
        y_predict = np.dot(xdata, weights)

        # 最小二乘法计算误差
        cost = np.sum((y_predict - ydata) ** 2) / len(xdata)
        cost_function.append(cost)

        #计算梯度
        dJ_dw = 2 * np.dot(xdata.T, (y_predict - ydata)) / len(xdata)

        #更新系数w,b的值
        weights = weights - learn_rate * dJ_dw
        #print("weights=",weights.shape)
        #提前结束循环的机制
        if len(cost_function) > 1:
            if 0 < cost_function[-2] - cost_function[-1] < error:
                break

    return weights, cost_function
def predict(xdata,ws):
    xt=Trans(xdata)
    return np.dot(xt,ws)

#画出散点图，及回归直线
def draw_curv(xdata,ydata,w):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xdata, ydata)  # [:],y_train[:])
    xcopy = xdata.copy()
    xcopy.sort(0)
    yhat = predict(xcopy, w)
    ax.plot(xcopy[:], yhat)
    plt.show()
def draw_curv1(xdata,ydata,w):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(xdata.shape)
    ax.scatter(xdata, ydata)  # [:],y_train[:])
    xcopy = xdata.copy()
    xcopy.sort(0)
    xcopy1=np.append(xcopy,xcopy*xcopy,axis=1)
    print(xcopy1.shape)
    yhat = predict(xcopy1, w)
    ax.plot(xdata,yhat)
    #ax.scatter(xdata, yhat)
    plt.show()
if __name__=='__main__':
    filename_train='第一题数据/train.txt'
    filename_test='第一题数据/test.txt'
    x_train,y_train=load_data(filename_train)
    x_train=x_train.reshape(-1,1) #将x_train由列表，转化为二维数组
    y_train=y_train.reshape(-1,1) #将y_train由列表，转化为二维数组
    from sklearn import preprocessing as spp  # 引入数据预处理的库
    scaler_01 = spp.MinMaxScaler() #进行极大极小归一化
    # 归一的x值
    x_train = scaler_01.fit_transform(x_train)
    #求出了参数值
    """
    #最小二乘法
    w=MinBiMultiply(x_train,y_train)
    print(w)
    pre_train=predict(x_train,w)
    print(pre_train)
    #画出曲线及散点图
    draw_curv(x_train, y_train, w)
    #读出验证数据
    x_test,y_test=load_data(filename_test)
    draw_curv(x_test,y_test,w)
    """
    #用梯度下降法，求解参数
    #w=GDS(x_train, y_train) #, apha=0.8, iter=1000, err=1e-4)
    #print(w)
    w,c=Gradient(x_train, y_train)
    print("w=",w)
    print("c=",c)
    x_test, y_test = load_data(filename_test)
    x_test=x_test.reshape(-1,1)
    x_test = scaler_01.fit_transform(x_test)
    y_test=y_test.reshape(-1,1)
    # draw_curv(x_test, y_test, w)
    #用2次曲线
    x_train_2=x_train*x_train
    x_train_ex=np.append(x_train,x_train_2,axis=1)
    print(x_train[0],x_train_2[0])
    print(x_train_ex)
    w1, c1 = Gradient(x_train_ex, y_train)
    print(w1)
    x_test, y_test = load_data(filename_test)
    x_test=x_test.reshape(-1,1)
    y_test=y_test.reshape(-1,1)
    x_test1 = scaler_01.fit_transform(x_test)
    print(x_test1.shape,y_test.shape)
    draw_curv1(x_test1, y_test, w1)
    x_testt=np.append(x_test,x_test*x_test,axis=1)
    preY=predict(x_testt,w1)
    print(preY)