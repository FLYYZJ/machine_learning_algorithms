import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def GetData_x_y(file_path):
    """
    用pandas函数获取数据，而不是原书中的用直接解析txt文件
    :param file_path:
    :return:
    """
    Data = pd.read_table(file_path, header=None)

    new_columns_name = {}
    for i in range(len(Data.columns)):
        if i == len(Data.columns)-1:
            new_columns_name[i] = 'y'
            continue
        new_columns_name[i] = 'x{}'.format(i)


    Data = Data.rename(columns=new_columns_name)
    print(Data)
    y = Data.y.values
    x = Data.drop('y', 1).values
    return x, y

def lwlr(testPoint, x, y, k=1.0):
    """
    对应一个点的局部加权后的拟合值计算
    :param testPoint:
    :param x:
    :param y:
    :param k:
    :return:
    """
    x = np.mat(x)
    y = np.mat(y).T
    m = x.shape[0]
    weights = np.mat(np.eye(m))

    for j in range(m):
    # 针对每一个测试点testPoint，计算testPoint到各点的距离，确定一个对应的加权系数
        diffMat = testPoint - x[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0*k**2))
    xTx = x.T * (weights * x)

    if np.linalg.det(xTx) == 0.0:
        print('This is a singular matrix, can not do inverse')
        return

    # ws = np.linalg.pinv(xTx) * (x.T * (weights * y))
    ws = xTx.I * (x.T * (weights * y))
    # 返回的是该测试点的加权后的取值
    return testPoint * ws

def lwlrTest(testArr, x, y, k = 1.0):
    """
    对整个数据集进行局部加权后可得全体测试数据对应的拟合值
    :param testArr:
    :param x:
    :param y:
    :param k:
    :return:
    """
    m = testArr.shape[0]
    yHat = np.zeros(m)

    for i in range(m):
        yHat[i] = lwlr(testArr[i], x, y, k)
    return yHat

def Plot_Fit_Line(x, y, yHat):
    """
    绘图程序
    :param x:
    :param y:
    :param yHat:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # scatter方法，绘制散点图，marker属性，是散点的形状，color，是散点的颜色
    ax.scatter(x[:, 1], y, marker='x', color='r')
    srtInd = x[:, 1].argsort(0)  # 按值大小从小到大排列x值下标，因为散点图是按照大小顺序绘制的
    xSort = x[srtInd]  # 得到排列好的x值
    ax.plot(xSort[:, 1], yHat[srtInd], color='b')  # 绘制拟合曲线

    plt.show()

def regError(y, yHat):
    return ((y - yHat)**2).sum()



if __name__ == '__main__':
    x, y = GetData_x_y('resources/ex0.txt')
    print(y[0])
    print(x[0])
    print(lwlr(x[0], x, y, 1.0))
    yHat = lwlrTest(x, x, y, 0.003)
    # print(yHat)
    Plot_Fit_Line(x, y, yHat)
    # x, y = GetData_x_y('resources/abalone.txt')
    # yHat = lwlrTest(x[100:199], x[0:99], y[0:99], 0.01)
    # print(regError(y[100:199], yHat))