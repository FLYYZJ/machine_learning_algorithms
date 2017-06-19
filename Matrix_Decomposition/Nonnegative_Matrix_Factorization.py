# @Time    : 2017/6/18 21:30
# @Author  : Aries
# @Site    :
# @File    : Nonnegative_Matrix_Factorization.py
# @Software: PyCharm Community Edition

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_row = 2
n_col = 4
image_shape = (64, 64)


def train(V, components, iternum, e):
    m,n = V.shape
    W = np.random.random((m, components))
    H = np.random.random((components, n))
    for iter in range(iternum):
        V_pre = np.dot(W, H)
        E = V - V_pre

        err = np.sum(E * E)
        print(err)
        if err < e:
            break

        a = np.dot(W.T, V)
        b = np.dot(W.T, np.dot(W, H))
        H[b != 0] = (H * a / b)[b != 0]

        c = np.dot(V, H.T)
        d = np.dot(W, np.dot(H, H.T))

        W[d != 0] = (W * c / d)[d != 0]
    return W, H




def plot_gallery(title, images, n_col=n_col, n_row=n_row):
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
    data = pd.read_csv('data.csv', sep='\t', header=None).values.T

    W, H = train(data, 8, 1000, 1e-4)
    print(H)
    print('**********************')
    print(W)
    plot_gallery('%s - Train time %.1fs' % ('Non-negative components - NMF', 1),
                 H)
    plt.show()

