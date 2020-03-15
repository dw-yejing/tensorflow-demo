# encoding: utf-8

import forward
import tensorflow as tf
from tensorflow import keras
import numpy

# 相关配置
learning_rate = 0.01


'''
反向传播训练网络，优化网络参数
'''


def backward():
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 784))
    y_actual = tf.compat.v1.placeholder(tf.float32, shape=(None, 10))
    y_predict = forward.forward(x)
    global_step = tf.Variable(0, trainable=False)
    # 定义损失函数
    # 这里采用均方平均值
    loss = tf.reduce_mean(tf.square(y_predict-y_actual))
    # 定义反向传播方法
    train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.session() as sess:
        init_op = tf.global_varibles_initializer()
        sess.run(init_op)

        for i in range(10000):
            fashion_mnist = keras.datasets.fashion_mnist
            (x, y), (x_test, y_test) = fashion_mnist.load_data()
            loss_value, step = sess.run([loss, global_step], feed_dict={x: x, y_actual: y})

            if i % 1000 == 0:
                print("After {} training step(s), loss on training is {}".format(step, loss_value))
                saver.save(sess, 'model/mnist_model', global_step=global_step)


if __name__ == "__main__":
    backward()

