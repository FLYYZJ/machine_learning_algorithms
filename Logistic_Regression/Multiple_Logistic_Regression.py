# @Time    : 2017/7/4 22:24
# @Author  : yaozijie
# @Site    :
# @File    : Logistic_regression.py
# @Software: PyCharm Community Edition

import numpy as np

import Deep_learning.Simplify_model.utils as utils


class LogisticRegression(object):
    def __init__(self, input_data, label, n_in, n_out):
        '''

        :param input_data:
        :param label:
        :param n_in:
        :param n_out:
        '''
        self.x = input_data
        self.y = label

        self.W = np.zeros((n_in, n_out)) # 权值初始化，全0
        self.b = np.zeros(n_out) # 偏置值初始化，全0

    def train(self, lr=0.1, input_data = None):
        '''
        训练函数，批量梯度下降而非随机梯度下降
        :param lr:  学习率
        :param input_data:  输入数据
        :param L2_reg:
        :return:
        '''
        if input_data is not None:
            self.x = input_data
        # 给定差值（实际值和预测值之差）
        p_y_given_x = self.output(self.x)
        d_y = self.y - p_y_given_x

        # 更新参数值，向量化更新方式，梯度下降更新法
        self.W += lr * np.dot(self.x.T, d_y)
        self.b += lr * np.mean(d_y, axis=0)


    def output(self, x):
        '''
        softmax过程，对应多分类logistic回归
        :param x:  输入数据
       :return:
        '''
        return utils.softmax(np.dot(x, self.W) + self.b)

    def predict(self, x):
        '''
        预测函数，又是一层无聊的封装
        :param x:
        :return:
        '''
        return self.output(x)

    def negative_log_likelihood(self):
        '''
        负极大似然函数
        :return:
        '''
        sigmoid_activation = utils.softmax(np.dot(self.x, self.W) + self.b)
        # 交叉熵损失函数，在MLP中会使用
        cross_entropy = - np.mean(np.sum(self.y * np.log(sigmoid_activation) + (1-self.y)*np.log(1-sigmoid_activation), axis=1))

        return cross_entropy


def test_lr(learning_rate = 0.1, n_epochs = 5000):
    rng = np.random.RandomState(123)

    # 构造数据集，每个样本有两个属性，共20个样本
    d = 2
    N = 10

    # 构造正负例对应的属性
    x1 = rng.randn(N, d) + np.array([0, 0])
    x2 = rng.randn(N, d) + np.array([20, 10])
    x3 = rng.randn(N, d) + np.array([10, 5])
    # 构造正负例对应的标签
    y1 = [[1, 0, 0] for i in range(N)]
    y2 = [[0, 1, 0] for i in range(N)]
    y3 = [[0, 0, 1] for i in range(N)]
    # 合并构造的x1和x2和x3，y1和y2和y3，同时取值为整数值
    x = np.r_[x1.astype(int), x2.astype(int), x3.astype(int)]

    y = np.r_[y1, y2, y3]

    print(x)
    print('********************************')
    print(y)
    print('********************************')
    # 构造logistic回归分类器
    classifier = LogisticRegression(input_data=x, label=y, n_in=d, n_out=3)

    # 开始训练过程
    for epoch in range(n_epochs):
        classifier.train(lr = learning_rate)

        learning_rate *= 0.995
    result = classifier.predict(np.array([10, 5])) # 
    print(result)


if __name__ == '__main__':
    test_lr(0.1) # 测试，输出矩阵中的每个元素对应一个概率值