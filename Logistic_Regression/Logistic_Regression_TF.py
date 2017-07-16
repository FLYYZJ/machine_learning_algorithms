'''
结合tensorflow实现Logistic回归算法，测试用例是手写体图片
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

def Logistic_Regression(lr = 0.01, training_epochs = 50, batch_size = 100, input_shape=784, output_shape=10):
    # 输入和输入标签的变量设置
    x = tf.placeholder(tf.float32, [None, input_shape])
    y = tf.placeholder(tf.float32, [None, 10])
    # 权值矩阵和偏置值的变量设置
    W = tf.Variable(tf.zeros([input_shape, output_shape]))
    b = tf.Variable(tf.zeros([output_shape]))

    # 预测函数是softmax函数，其实就是logistic二分类器的多分类形式
    pred = tf.nn.softmax(tf.matmul(x, W) + b)
    # 损失函数是使用交叉熵函数
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    # 优化迭代器使用的是梯度下降，注意格式
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
    # 初始化所有变量
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0
            # 总共的批次数
            total_batch = int(mnist.train.num_examples/batch_size)
            #
            for i in range(total_batch):
                # 提取每个批次的训练数据
                batch_xs, batch_yx = mnist.train.next_batch(batch_size)
                # 开始模型的奔跑
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_yx})
                # 计算平均损失
                avg_cost += c/total_batch

            if (epoch + 1) % 1 == 0:
                # 每1轮都输出cost值
               print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # 开始检验模型
        correction_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # 模型的检验准确率
        accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))

if __name__ == "__main__":
    Logistic_Regression(lr=0.01, training_epochs=50, batch_size=100, input_shape=784, output_shape=10)


