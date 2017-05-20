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
    Data = Data.rename(columns={1: 'x', 2: 'y'})
    # print(Data)
    y = Data.y.values
    x = Data.drop('y', 1).values
    return x, y

def UniryStandRegress(x, y):
    """
    二元标准回归函数
    :param x:
    :param y:
    :return:
    """


    x = np.mat(x)
    y = np.mat(y).T


    print(x.shape, y.shape)
    xTx = x.T * x
    print(xTx.shape)
    if np.linalg.det(xTx) == 0.0:
        print('This is a singular matrix,cannot do inverse')
        return
    ws = xTx.I * (x.T * y)
    print(ws.shape)
    return ws
def Plot_Curve(x, y, ws):
    """
    绘制拟合直线程序
    :param x:
    :param y:
    :param ws: 由 normal equation（最小二乘法）计算所得的拟合系数
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # scatter方法，绘制散点图，marker属性，是散点的形状，color，是散点的颜色
    ax.scatter(x[:, 1], y, marker='x', color='r')
    # yHat预测值
    yHat = np.dot(x, ws)
    # 计算线性相关性
    print('线性相关性矩阵：')
    print(np.corrcoef(yHat.T, y))
    # 绘制拟合曲线
    ax.plot(x[:, 1], yHat)
    plt.show()


if __name__ == "__main__":
    x, y = GetData_x_y('resources/ex0.txt')
    print(UniryStandRegress(x, y))
    Plot_Curve(x, y, UniryStandRegress(x, y))