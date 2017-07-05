# @Time    : 2017/7/5 9:59
# @Author  : yaozijie
# @Site    : 
# @File    : Linear_Regression_TF.py
# @Software: PyCharm Community Edition

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 对应参数,学习率，学习轮数，
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# 生成训练集
train_x = np.linspace(-1, 1, 200)
train_y = 2*train_x + np.random.randn(*train_x.shape) * 0.2

n_samples = train_x.shape[0]

# 开始构建输入图
X = tf.placeholder('float')
Y = tf.placeholder('float')

# 构建tensorflow的变量
W = tf.Variable(np.random.randn(), name='weight')
b = tf.Variable(np.random.randn(), name='bias')

# 构建线性回归模型
pred = tf.add(tf.multiply(X, W), b)

# 损失函数采用最小二乘函数
cost = tf.reduce_sum(tf.pow(pred - Y, 2))/(2 * n_samples)

# 优化器，梯度下降，目标函数最小化最小二乘损失，并输入学习率
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化变量
init = tf.initialize_all_variables()



with tf.Session() as sess: # 注意所有涉及sess的操作都必须在with作用域下，因为with的作用域结束后会自动关闭掉sess
    sess.run(init)  # 正式初始化所有的变量

    for epoch in range(training_epochs):  # 训练过程
        for (x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if (epoch + 1) % display_step == 0:


            c = sess.run(cost, feed_dict={X: train_x, Y: train_y})
            print('Epoch:', '%04d' %(epoch+1), 'cost=', '{:.9f}'.format(c), 'w=', sess.run(W), 'b=', sess.run(b))
    print('Optimization Finished....')

    training_cost = sess.run(cost, feed_dict={X: train_x, Y: train_y}) # 计算训练损失值
    print('training cost = ', training_cost, ' w = ', sess.run(W), ' b = ', sess.run(b))  # 是输出1000轮迭代后的训练值

    # 绘图
    plt.plot(train_x, train_y, 'ro', label='original data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='fit line')
    plt.legend()
    plt.show()

    # 产生测试数据，做测试
    test_x = np.linspace(-1, 1, 200)
    test_y = 2*test_x + np.random.randn(*test_x.shape) * 0.2

    print('Testing....')
    test_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2))/(2 * test_x.shape[0]),
        feed_dict={X: test_x, Y: test_y})

    print('testing cost = ', test_cost)
    print('absoulte mean square loss difference: ', abs(training_cost - test_cost))

    plt.plot(test_x, test_y, 'bo', label='Testing data')
    plt.plot(test_x, sess.run(W) * test_x + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()


