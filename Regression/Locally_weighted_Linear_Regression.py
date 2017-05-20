import numpy as np
import pandas as pd

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

def lwlr(testPoint, x, y, k=1.0):
    """

    :param testPoint:
    :param x:
    :param y:
    :param k:
    :return:
    """
    x = np.mat(x)
    y = np.mat(y)
    m = x.shape[0]
    weights = np.mat(np.eye(m))

    for j in range(m):
        diffMat = testPoint - x[j:]
        print(diffMat)
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0*k**2))
    xTx = x.T * (weights * x)
    if np.linalg.det(xTx) == 0.0:
        print('This is a singular matrix, can not do inverse')
        return
    ws = xTx.I * (x.T * (weights * y))
    return testPoint * ws

def lwlrTest(testArr, x, y, k = 1.0):
    """

    :param testArr:
    :param x:
    :param y:
    :param k:
    :return:
    """
    m = testArr.shape[0]
    yHat = np.zeros(m)

    for i in range(m):
        yHat = lwlr(testArr[i], x, y, k)
    return yHat

if __name__ == '__main__':
    x, y = GetData_x_y('resources/ex0.txt')
    print(y[0])
    print(x[0])
    print(lwlr(x[0], x, y, 1.0))