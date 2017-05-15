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
        labelMat.append([int(line_[2])])
    return dataMat, labelMat

def sigmoid(x):
    """
    sigmoid函数 计算sigmoid函数值
    param x: 输入属性值
    return: 返回sigmoid函数值
    """
    return 1.0/(1 + np.exp(x))

def gradAscent(dataMatIn, classLables):
    """
    gradAscent函数 梯度上升法迭代更新函数
    param dataMatIn: 输入训练数据
    param classLables: 输入对应的训练数据的标签
    return: 返回更新完毕的权重值
    """