import numpy as np

import Deep_learning.Simplify_model.utils as utils


class HiddenLayer(object):
    '''
    隐含层类
    '''
    def __init__(self, input_data, n_in, n_out, W=None, b=None, rng=None, activation=utils.tanh):
        '''

        :param input: 输入数据
        :param n_in: 输入数据的宽度
        :param n_out: 输出数据的宽度
        :param W: 权值矩阵，当前层和下一层的链接权值
        :param b: 偏置值，当前层偏置值
        :param rng: 随机数发生器
        :param activation: 激活函数类型
        '''

        if rng == None:
            self.rng = np.random.RandomState(1234)
        else:
            self.rng = rng

        if W == None: # 构造权值矩阵
            a = 1. / n_in
            self.W = np.array(rng.uniform(low = -a, high = a, size = (n_in, n_out)))
        else:
            self.W = W
        self.x = input_data
        if b == None:
            self.b = np.zeros(n_out)
        else:
            self.b = b

        self.activation = activation
        if activation == utils.tanh:
            self.deactivation = utils.dtanh
        elif activation == utils.sigmoid:
            self.deactivation = utils.dsigmoid
        elif activation == utils.ReLU:
            self.deactivation = utils.dReLU

        else:
            raise ValueError('thie module does not support such activation function')

    def output(self, input_data = None):
        '''
        隐含层输出程序,输出隐藏层的激活值
        :param input_data: 输入数据
        :return: 输出的激活函数值
        '''
        if input_data is not None:
            self.x = input_data
        linear_output = np.dot(self.x, self.W) + self.b  # 记住神经元中激活函数的激活机制，其输入是一个线性输入
        return self.activation(linear_output)

    def forward(self, input_data = None):
        '''
        前向传播函数，计算每个神经元的输出值
        :param input_data: 输入的数据
        :return: 计算神经元的输出值
        '''
        return self.output(input_data)

    def backward(self, prev_layer, lr = 0.1, input_data = None, dropout = False, mask = None):
        '''
        一个很好的解释参考：http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
        反向传播函数，更新当前隐层网络权值，注意更新公式
        :param prev_layer: 前一层
        :param lr: 学习率
        :param input_data: 输入数据
        :param dropout: 舍弃部分连接的比例
        :param mask: 掩膜
        :return:
        '''
        if input_data is not None:
            self.x = input_data
        # 当前层输出值的导数,其计算公式是当前层的输出（下一层的输入） * 上一层的误差与当前层和上一层的权重的矩阵乘积
        d_y = self.deactivation(prev_layer.x) * np.dot(prev_layer.d_y, prev_layer.W.T)
        if dropout == True: # 需要舍弃部分连接
            d_y *= mask

        self.W += lr * np.dot(self.x.T, d_y)
        self.b += lr * np.mean(d_y, axis=0)
        self.d_y = d_y

    def dropout(self, input_data, p, rng = None):
        '''
        舍弃部分连接函数
        :param input_data:
        :param p: 舍弃率
        :param rng: 随机数发生器
        :return:
        '''
        if rng is None:
            rng = np.random.RandomState(123)
        mask = rng.binomial(size=input_data.shape, n=1, p=1-p)
        return mask

    def sample_h_given_v(self, input_data=None):
        '''
        采样函数，给定输出的概率值作为采样率，做二项分布计算
        :param input_data:
        :return:
        '''
        if input_data is not None:
            self.x = input_data
        v_mean = self.output()
        h_sample = self.rng.binomial(size=v_mean.shape, n=1, p=v_mean)

        return h_sample