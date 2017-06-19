# @Time    : 2017/6/18 21:30
# @Author  : Yao zijie
# @Site    :
# @File    : Nonnegative_Matrix_Factorization.py
# @Software: PyCharm Community Edition

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

n_row = 2
n_col = 4
image_shape = (64, 64)


def train(V, components, iternum, e):
    '''
    非负矩阵分解函数
    :param V:  原始矩阵
    :param components:  要提取多少个特征
    :param iternum: 迭代次数
    :param e: 误差阈值
    :return:
    '''
    m,n = V.shape
    # 随机初始化两个矩阵
    W = np.random.random((m, components))
    H = np.random.random((components, n))
    # 迭代计算过程，循环中使用了numpy的切片操作，可以避免直接使用Python的多重循环，从而提高了运行速度
    for iter in range(iternum):
        V_pre = np.dot(W, H)
        E = V - V_pre

        err = np.sum(E * E)
        print(err)
        if err < e:
            break
        # 对照更新公式
        a = np.dot(W.T, V)
        b = np.dot(W.T, np.dot(W, H))
        H[b != 0] = (H * a / b)[b != 0]

        c = np.dot(V, H.T)
        d = np.dot(W, np.dot(H, H.T))

        W[d != 0] = (W * c / d)[d != 0]
    return W, H




def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    '''
    绘图函数
    :param title: 图像名称
    :param images: 图像
    :param n_col: 输出多少个子图，对应有n_col*n_row个子图
    :param n_row:
    :return:
    '''
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape).T, cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

if __name__ == '__main__':
    # 读入的data（经过处理的数字图片，是csv格式）是一个4096*64维的矩阵，在此将其做转置，变成64*4096维的矩阵，这是计算需求
    data = pd.read_csv('data.csv', sep='\t', header=None).values.T
    print(data.shape)
    t = time.time()
    W, H = train(data, 8, 1000, 1e-4)
    # print(H)
    # print('**********************')
    # print(W)
    plot_gallery('%s - Train time %.1fs' % ('Non-negative components - NMF', time.time() - t),
                 H)
    plt.show()

