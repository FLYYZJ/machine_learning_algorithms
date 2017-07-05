import numpy as np
import Deep_learning.Hidden_Layer as HiddenLayer
import Deep_learning.Logistic_regression as LR
import Deep_learning.utils as utils


class MLP(object):
    '''
        实现一个三层（输入-隐藏-输出）MLP的类
    '''
    def __init__(self, input_data, input_label, n_in, n_hidden, n_out, rng = None):
        '''

        :param input_data: 输入样本属性集
        :param input_label: 输入样本标签
        :param n_in: 网络输入
        :param n_hidden: 隐藏层输入
        :param n_out: 网络输出
        :param rng: 随机数发生器
        '''
        self.x = input_data
        self.y = input_label

        if rng == None:
            rng = np.random.RandomState(125)
        # 构建单层隐藏层
        self.hidden_layer = HiddenLayer.HiddenLayer(input_data=self.x,
                                                    n_in=n_in,
                                                    n_out=n_hidden,
                                                    rng=rng,
                                                    activation=utils.tanh)
        self.log_layer = LR.LogisticRegression(input_data=self.hidden_layer.output,
                                               label=self.y,
                                               n_in=n_hidden,
                                               n_out=n_out)

    def train(self):
        '''

        :return:
        '''
        # 输入数据(直接就作为输入层存在) -> 隐藏层 -> 输出层（logistics回归作为分类器）
        # 输入层数据输入后直接进入隐藏层，然后做正向传播
        layer_input = self.hidden_layer.forward()
        # 正向传播后进入logistics层（输出层）输出结果
        self.log_layer.train(input_data=layer_input)
        # 完成后再把输出层的结果反向传播到隐藏层进行权重和偏置值更新
        self.hidden_layer.backward(prev_layer=self.log_layer)

    def predict(self, x):
        '''
        :param x:
        :return:
        '''
        x = self.hidden_layer.output(input_data=x)
        return self.log_layer.predict(x)


if __name__ == '__main__':
    '''
    一个异或门的例子
    '''
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
    # 5个隐藏层神经元，输入层宽度为2，输出层宽度为2,3层网络结构
    MLP_classifier = MLP(input_data=x, input_label=y, n_in=2, n_hidden=5, n_out=2)
    for epoch in range(2000):
        MLP_classifier.train()
    print(MLP_classifier.predict(x))
