import numpy as np
import matplotlib.pyplot as plt


def loadDataSet():
    """
    loadDataSet函数，用于引入数据集testSet.txt
    params: 无
    return: 返回数据集的属性和标签
    """
    dataMat = []
    labelMat = []
    f = open('testSet.txt')
    for line in f.readlines():
        line_ = line.strip().split() # 去除每行首尾的换行符，然后按照空格符进行分割
        dataMat.append([1.0, float(line_[0]), float(line_[1])])
        labelMat.append(int(line_[2]))
    return dataMat, labelMat

def sigmoid(x):
    """
    sigmoid函数 计算sigmoid函数值
    param x: 输入属性值
    return: 返回sigmoid函数值
    """
    return 1.0/(1 + np.exp(-1 * x))

def gradAscent(dataMatIn, classLables):
    """
    gradAscent函数 梯度上升法迭代更新函数
    param dataMatIn: 输入训练数据
    param classLables: 输入对应的训练数据的标签
    return: 返回更新完毕的权重值
    """
    dataMatrix = np.mat(dataMatIn)  # 训练数据属性集合
    labelMatrix = np.mat(classLables).transpose()  # 训练数据的标签
    m, n = np.shape(dataMatrix)  # 获取矩阵的size

    alpha = 0.001  # 学习率
    maxCycles = 300  # 迭代轮数为500
    weights = np.ones((n, 1))  # 权重向量初始化全1

    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 利用当前权重向量计算预测值
        error = (labelMatrix - h)  # 计算梯度值
        weights = weights + alpha * dataMatrix.transpose() * error  # 迭代更新权重向量

        # print(weights)
    return weights

def plotBestFit():
    """
    plotBestFit 函数，绘制训练集上的分界线
    param weights:
    return:
    """
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x) / weights[2]
    ax.plot(x, y.transpose())
    plt.xlabel(('X1'))
    plt.ylabel(('X2'))
    plt.show()


# data_a, data_l = loadDataSet()
# print(gradAscent(data_a, data_l))
plotBestFit()