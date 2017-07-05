# @Time    : 2017/7/4 20:31
# @Author  : yao zijie
# @Site    : 
# @File    : utils.py
# @Software: PyCharm Community Edition

import numpy as np

def sigmoid(x):
    '''
    sigmoid激活函数
    :param x:
    :return:
    '''
    return 1. / (1 + np.exp(-x))

def dsigmoid(x):
    '''
    sigmoid 激活函数的导数
   :param x:
    :return:
    '''
    return x * (1. - x)

def tanh(x):
    '''
    tanh函数，一种激活函数
    :param x:
    :return:
    '''
    return np.tanh(x)

def dtanh(x):
    '''
    tanh函数的导数
    :param x:
    :return:
    '''
    return 1. - x * x

def softmax(x):
    '''
    softmax分类函数，是logistic函数的多分类推广，在此作为损失函数
    :param x: 输入为向量
    :return:
    '''
    e = np.exp(x - np.max(x))  # 防止溢出错误
    if e.ndim == 1: # 当维度为1时，代表这是一个一维向量，2代表一个矩阵，3以上代表一个张量，当问题是一个普通的二分类logistic回归时对应的ndim为1
        return e / np.sum(e, axis=0)
    else:
        # 逐行求和后转置，一个进行多分类的推广，返回值的每个元素表示其对应于所属类别的概率，详见 http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92
        return e / np.array([np.sum(e, axis=1)]).T


def ReLU(x):
    '''
    ReLU激活函数
    :param x:
    :return:
    '''
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)


