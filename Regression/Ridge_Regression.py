import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

numTestpts = 30
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
    # print(Data)
    y = Data.y.values
    x = Data.drop('y', 1).values
    return x, y

def ridgeRegress(x, y, lam = 0.2):
    """

    :param x:
    :param y:
    :param lam:
    :return:
    """
    xTx = x.T * y
    denom = xTx + np.eye(x.shape[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("this is a singular matrix")
        return
    ws = denom.I * (x.T * y)
    return ws

def ridgeTest(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    x = np.mat(x)
    y = np.mat(y).T

    yMean = np.mean(y, 0)
    y = y - yMean  # 中心化处理

    xMean = np.mean(x, 0)
    xVar = np.var(x, 0)
    x = (x - xMean) / xVar  # 中心化处理


    weights = np.zeros((numTestpts, x.shape[1]))
    for i in range(numTestpts):
        ws = ridgeRegress(x, y, np.exp(i-10))
        weights[i, :] = ws.T
    return weights, y

if __name__ == "__main__":
    x, y = GetData_x_y('resources/abalone.txt')
    weights, y = ridgeTest(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('log(lambda)')
    plt.ylabel('regression coff')
    x_range = [i-10 for i in range(numTestpts)]
    ax.plot(x_range, weights)
    plt.show()
