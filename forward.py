# encoding: utf-8

import tensorflow as tf

'''
前向传播构建神经网络结构
x:输入值
'''
def forward(x):
    # 定义隐藏层
    # 第一层为100个神经元，输入为【28x28】像素的数据
    w1 = get_weight([784, 100])
    b1 = get_weight([100])
    y1 = tf.nn.relu(tf.matmul(x, w1)+b1)
    # 定义输出层
    # 输出层为长度为10的一维数组，其值代表可能的概率，其索引代表实际的数字
    w2 = get_weight([100, 10])
    b2 = get_bias(10)
    y2 = tf.matmul(y1, w2) + b2
    return y2
'''
获取权重
shape:权重数量
'''
def get_weight(shape):
    w = tf.Variable(tf.random.normal(shape))
    return w

'''
获取偏移量
shape:偏移量数量
'''
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b
