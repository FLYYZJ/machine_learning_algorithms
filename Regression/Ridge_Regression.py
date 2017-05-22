import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge


numTestpts = 40
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
    # 添加了一项偏置项
    Data['intercept_'] = 1

    y = Data.y.values
    x = Data.drop('y', 1).values
    return x, y

def ridgeRegress(x, y, lam = 0.2):
    """
    岭回归算法实现
    :param x:
    :param y:
    :param lam:
    :return:
    """
    xTx = x.T * x
    denom = xTx + np.eye(x.shape[1]) * lam  # 增加一项lambda项，使回归方程适用于病态系统
    if np.linalg.det(denom) == 0.0:
        print("this is a singular matrix")
        return
    ws = denom.I * (x.T * y)
    return ws

def ridgeTest(x, y):
    """
    一个对多数据进行岭回归测试的函数
    :param x:
    :param y:
    :return:
    """
    x = np.mat(x)
    y = np.mat(y).T

    # 经过如下标准化处理后得到的拟合数据值非常奇怪，所以在此不加这项偏置项
    # yMean = np.mean(y, 0)
    # y = y - yMean  # 中心化处理
    #
    #
    # xMean = np.mean(x, 0)
    # xVar = np.var(x, 0)
    # x = (x - xMean) / xVar  # 中心化处理，使x的方差为1，均值为0


    weights = np.zeros((numTestpts, x.shape[1]))
    for i in range(numTestpts):   # 测试不同的lambda值对岭回归的影响
        ws = ridgeRegress(x, y, np.exp(i-10))
        weights[i, :] = ws.T
    return weights

if __name__ == "__main__":
    x, y = GetData_x_y('resources/abalone.txt')
    weights = ridgeTest(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('log(lambda)')
    plt.ylabel('regression coff')
    x_range = [i-10 for i in range(numTestpts)]
    ax.plot(x_range, weights)
    plt.show()


    print('***********用sklearn库的岭回归进行拟合****************')
    clf = Ridge(alpha=.5)
    clf.fit(x, y)
    print(clf.coef_)
    print(clf.intercept_)
    print(clf.predict(np.array([1, 0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 0.15, 1])))

    print("******************用自己编写的岭回归代码进行拟合******************")
    print(weights[15][0:len(weights[0])-1])
    print(weights[15][-1])
    print(np.dot(weights[15], np.array([1, 0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 0.15, 1]).T))
