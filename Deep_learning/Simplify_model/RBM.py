# 受限玻尔兹曼机
# 受限玻尔兹曼机 energy-based模型，单层结构，由可视单元和隐藏单元组成，层内无连接，层间具有连接,常用于提取特征
# RBM的训练过程类似于一个编码后重构的过程，寻求重构误差最小化，而编码后的为输出的特征
# 学习目标：能量函数在指数族分布下变成概率分布形式，然后做极大似然估计
# 学习算法：CD算法（对比散列算法）
# 参考  http://www.cnblogs.com/kemaswill/p/3203605.html
#      https://wenku.baidu.com/view/87efdf28c1c708a1294a448d.html

import numpy as np

import Deep_learning.Simplify_model.utils as utils


class RBM(object):
    '''
    受限玻尔兹曼机类
    '''
    def __init__(self, input_data = None, n_visible = 2, n_hidden = 3, W = None, hbias = None, vbias = None, rng = None):
        '''

        :param input_data: 输入数据
        :param n_visible: 可视层（输入层）的神经元数，默认为2
        :param n_hidden: 隐藏层的神经单元数，默认为3
        :param W: 可视层和隐层的链接权值
        :param hbias: 隐层神经元的偏置值
        :param vbias: 可视层神经元的偏置值
        :param rng: 随机数发生器
        '''

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if rng is None:
            self.rng = np.random.RandomState(111)
        else:
            self.rng = rng
        if W is None:
            a = 1. / n_visible
            self.W = np.array(rng.uniform(low=-a, high=a, size=(n_visible, n_hidden)))
        else:
            self.W = W

        if hbias is None:
            self.hbias = np.zeros(n_hidden)
        else:
            self.hbias = hbias

        if vbias is None:
            self.vbias = np.zeros(n_visible)
        else:
            self.vbias = vbias

        self.input_data = input_data

    def contrastive_divergence(self, lr = 0.1,  input_data = None):
        '''
        基于对比散度的受限玻尔兹曼机训练函数
        :param lr: 学习率
        :param k: gibbs采样轮数
        :param input_data: 输入数据
        :return:
        '''
        if input_data is not None:
            self.input_data = input_data
        ph_mean, ph_sample = self.sample_h_given_v(self.input_data)  # 最开始的输入数据是最开始的观察值

        # 这里仅采用一步吉布斯采样，这是文章中给定的一个推荐值
        nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(ph_sample)

        # 更新各参数
        self.W += lr * (np.dot(self.input_data.T, ph_mean) - np.dot(nv_samples.T, nh_means))
        self.vbias += lr * np.mean(self.input_data - nv_samples, axis=0)
        self.hbias += lr * np.mean(ph_mean - nh_means, axis=0)
        print(self.get_reconstruction_cross_entropy())


    def propup(self, v):
        '''
        计算给定可视单元v的条件下，隐层单元的条件分布，作为采样的基础
        :param v: 可视单元取值
        :return:
        '''
        pre_sigmoid_activation = np.dot(v, self.W) + self.hbias
        return utils.sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        '''
        计算给定隐层单元h的条件下，隐层单元的条件分布，作为采样的基础
        :param h: 隐层单元h的取值
        :return:
        '''
        pre_sigmoid_activation = np.dot(h, self.W.T) + self.vbias
        return utils.sigmoid(pre_sigmoid_activation)

    def sample_v_given_h(self, h0_sample):
        '''
        从已知隐层单元获得可视单元被激活（0-1状态为1）的概率
        :param h0_sample: 隐层单元取值
        :return:
        '''
        v1_mean = self.propdown(h0_sample)  # 条件概率值p_v_given_h,以该概率值作为二项分布的参数
        v1_sample = self.rng.binomial(size=v1_mean.shape,  # 二项分布（隐单元和可视单元的取值为{0,1})
                                      n=1,
                                      p=v1_mean)
        return [v1_mean, v1_sample]

    def sample_h_given_v(self, v0_sample):
        '''
        从已知可视单元获得隐层单元被激活（0-1状态为1）的概率后，在进行采样
        :param v0_sample: 可视层单元取值
        :return:
        '''
        h1_mean = self.propup(v0_sample)  # 条件概率值p_v_given_h,以该概率值作为二项分布的参数，因为神经元的取值仅为0或1，表示激活与否
        h1_sample = self.rng.binomial(size=h1_mean.shape,  # 按二项分布（隐单元和可视单元的取值为{0,1})进行采样
                                      n=1,
                                      p=h1_mean)
        return [h1_mean, h1_sample]

    def gibbs_hvh(self, h0_sample):
        '''
        gibbs采样过程
        :param h0_sample:
        :return:
        '''
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [v1_mean, v1_sample, h1_mean, h1_sample]

    def get_reconstruction_cross_entropy(self):
        '''
        计算重构误差，即重构后的交叉熵
        :return:
        '''
        pre_sigmoid_activation_h = np.dot(self.input_data, self.W) + self.hbias
        sigmoid_activation_h = utils.sigmoid(pre_sigmoid_activation_h)

        pre_sigmoid_activation_v = np.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = utils.sigmoid(pre_sigmoid_activation_v)

        # 计算交叉熵
        cross_entropy = -np.mean(np.sum(self.input_data * np.log(sigmoid_activation_v) + (1-self.input_data)*np.log(1 - sigmoid_activation_v),axis=1))

        return cross_entropy

    def reconstruct(self, v):
        '''
        重构函数，即predict函数，根据输入计算所属对应的标签的概率值
        :param v:
        :return:
        '''
        h = utils.sigmoid(np.dot(v, self.W) + self.hbias)
        reconstructed_v = utils.sigmoid(np.dot(h, self.W.T) + self.vbias)
        return reconstructed_v


if __name__ == '__main__':
    data = np.array([[1, 1, 1, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 1, 1, 0, 1]])
    rng = np.random.RandomState(121)

    rbm = RBM(input_data=data, n_visible=6, n_hidden=3, rng=rng)

    for epoch in range(10000):
        rbm.contrastive_divergence(lr=0.01)

    v = np.array([[0, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 0]])

    print(rbm.reconstruct(v))



