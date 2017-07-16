# @Time    : 2017/7/16 14:58
# @Author  : Aries
# @Site    : 
# @File    : KNN_TF.py
# @Software: PyCharm Community Edition

'''
利用tensorflow实现KNN算法，例子是采用手写体识别的例子
'''
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

def KNN(input_shape=784, output_shape=10):
    # 训练集大小为5000
    Xtr, Ytr = mnist.train.next_batch(5000)
    # 取其重200个数据作为测试集
    Xte, Yte = mnist.test.next_batch(200)

    # 一如既往地占坑
    xtr = tf.placeholder("float", [None, input_shape])
    xte = tf.placeholder("float", [input_shape])

    # 定义距离计算的方法
    distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
    # 预测结果为距离最小者，这是一个1-NN算法
    pred = tf.arg_min(distance, 0)
    accuracy = 0
    # 初始化所有的变量
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(len(Xte)):
            # 找出第i个测试点的最近邻点
            nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
            # 比较预测值和真实值
            print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), "True Class:", np.argmax(Yte[i]))
            # 统计准确率
            if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
                accuracy += 1. / len(Xte)
        print('Done! Acurracy = ', accuracy)

if __name__ == "__main__":
    KNN(input_shape=784, output_shape=10)