import numpy as np

import Deep_learning.Simplify_model.Hidden_Layer as HiddenLayer
import Deep_learning.Simplify_model.Logistic_regression as LR
import Deep_learning.Simplify_model.RBM as RBM
import Deep_learning.Simplify_model.utils as utils


class DBN(object):
    '''
    深度置信网络
    几个问题：为什么引入sigmoid层（隐层），这是一个MLP和RBMs共存的网络，我们在训练RBMs的同时得到的更新参数值是与MLP共享的，即我们
              其实是采用无监督预训练层层的RBM得到的参数，其实的得到的就是MLP的参数，然后最后再接一个Logstic层，用于做监督学习的，然
              后再利用的finetune的方式微调一下参数。
    '''
    def __init__(self, input=None, label=None, n_ins=2, hidden_layer_size=[3, 3], n_out=2, rng=None):
        '''
        :param input: 输入数据的属性
        :param label: 输入数据的标签
        :param n_ins: 输入层， 数据属性的数量
       :param hidden_layer_size:
        :param n_out: 输出层，总共要输出几个标签
        :param rng: 随机数发生器
        '''
        self.x = input
        self.y = label

        self.sigmoid_layers = []
        self.rbm_layers = []
        # 隐藏层的数目
        self.n_layers = len(hidden_layer_size)

        if rng == None: rng = np.random.RandomState(111)

        assert self.n_layers > 0  # 判断网络层数的配置是否正确
        #　构造多层网络
        for i in range(self.n_layers):
            if i == 0:  # 输入层的输入
                input_size = n_ins
            else:       # 中间隐藏层的输入
                input_size = hidden_layer_size[i-1]

            if i == 0:
                layer_input = self.x  # 对应输入层
            else:
                layer_input = self.sigmoid_layers[-1].sample_h_given_v()  # 对应中间层
            # 构造sigmoid层
            sigmoid_layer = HiddenLayer.HiddenLayer(input_data=layer_input,
                                                    n_in=input_size,
                                                    n_out=hidden_layer_size[i],
                                                    rng=rng,
                                                    activation=utils.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)

            # 构造玻尔兹曼机层
            rbm_layer = RBM.RBM(input_data=layer_input,
                                n_visible=input_size,
                                n_hidden=hidden_layer_size[i],
                                W=sigmoid_layer.W,  # 和sigmoid层共享参数
                                hbias=sigmoid_layer.b)  # 和sigmoid层共享参数
            self.rbm_layers.append(rbm_layer)

        # 构造输出层 logistic层，用于预测输出
        self.log_layer = LR.LogisticRegression(input_data=self.sigmoid_layers[-1].sample_h_given_v(),
                                               label=self.y,
                                               n_in=hidden_layer_size[-1],
                                               n_out=n_out)

        self.finetune_cost = self.log_layer.negative_log_likelihood()

    def pretrain(self, lr=0.1, epochs=100):
        '''
        预训练函数
        :param lr:
        :param epochs:
        :return:
        '''
        for i in range(self.n_layers):
            if i == 0:
                layer_input = self.x
            else:
                # 先用sigmoid层做抽样，构成可以输入rbm的数据
                layer_input = self.sigmoid_layers[i-1].sample_h_given_v(layer_input)

            rbm = self.rbm_layers[i]

            for epoch in range(epochs):
                rbm.contrastive_divergence(lr=lr, input_data=layer_input)

    def finetune(self, lr=0.1, epochs=100):
        '''
        微调阶段
        :param lr: 学习率
        :param epochs: 学习轮次
        :return:
        '''
        # 从最后一层开始反向
        layer_input = self.sigmoid_layers[-1].sample_h_given_v()

        epoch = 0
        done_looping = False
        # 解释一下为什么这里的微调是这样的，因为这是一个单层的MLP输出网络，仅有一个输入层和一个输出层
        while (epoch < epochs) and (not done_looping):
            self.log_layer.train(lr=lr, input_data=layer_input)
            lr *= 0.95
            epoch += 1

    def predict(self, x):
        '''

        :param x:
        :return:
        '''
        layer_input = x
        for i in range(self.n_layers):
            sigmoid_layer = self.sigmoid_layers[i]
            layer_input = sigmoid_layer.output(input_data=layer_input)
        out = self.log_layer.predict(layer_input)
        return out


def test_dbn(pretrain_lr=0.1, pretraining_epochs=1000, k=1,
             finetune_lr=0.1, finetune_epochs=200):
    x = np.array([[1, 1, 1, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0]])
    y = np.array([[1, 0],
                     [1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1],
                     [0, 1]])

    rng = np.random.RandomState(123)

    # construct DBN
    dbn = DBN(input=x, label=y, n_ins=6, hidden_layer_size=[3, 3], n_out=2, rng=rng)

    # pre-training (TrainUnsupervisedDBN)
    dbn.pretrain(lr=pretrain_lr, epochs=pretraining_epochs)

    # fine-tuning (DBNSupervisedFineTuning)
    dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)

    # test
    x = np.array([[1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0],
                     [1, 1, 1, 1, 1, 0]])

    print(dbn.predict(x))



if __name__ == "__main__":
    test_dbn()

