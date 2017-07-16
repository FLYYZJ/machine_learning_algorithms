import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../data/MNIST_data/", one_hot=True)


def encoder(x, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

def decoder(x, weights, biases):
    layer_1 = tf.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2

def autoenconder(lr = 0.01, training_epochs = 20, batch_size = 256, display_step = 1, examples_to_show = 10,
                 n_hiiden = [256, 128], n_input = 784):
    '''

    :param lr:
    :param training_epochs:
    :param batch_size:
    :param display_step:
    :param example_to_show:
    :param n_hiiden:
    :param n_input:
    :return:
    '''

    X = tf.placeholder("float", [None, n_input])

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hiiden[0]])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hiiden[0], n_hiiden[1]])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hiiden[1], n_hiiden[0]])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hiiden[1], n_hiiden[0]])),
    }
    biases ={
        'encoder_b1': tf.Variable(tf.random_normal(n_hiiden[0])),
        'encoder_b2': tf.Variable(tf.random_normal(n_hiiden[1])),
        'decoder_b1': tf.Variable(tf.random_normal(n_hiiden[0])),
        'decoder_b2': tf.Variable(tf.random_normal(n_hiiden[0])),
    }
    encoder_op = encoder(X, weights, biases)
    decoder_op = decoder(encoder_op)

    y_pred = decoder_op

    y_true = X

    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost)

    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)
    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")
    encode_decode = sess.run(y_pred, feed_dict={X: mnist.test.images[: examples_to_show]})
    f,a  = plt.subplot(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.show()


if __name__ == '__main__':
    autoenconder(lr=0.01, training_epochs=20, batch_size=256, display_step=1, examples_to_show=10,
                 n_hiiden=[256, 128], n_input=784)
