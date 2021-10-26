# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib

# this is a test


def nn():
    model = tf.keras.models.Sequential()
    # 定义输入层
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    # 定义隐藏层(单层网络)
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # 定义输出层
    tf.keras.layers.Dense(10, activation='softmax')

    # model = tf.keras.models.Sequential([
    #   tf.keras.layers.Flatten(input_shape=(28, 28)),
    #   tf.keras.layers.Dense(128, activation='relu'),
    #   tf.keras.layers.Dropout(0.2),
    #   tf.keras.layers.Dense(10, activation='softmax')
    # ])
    # 编译模型
    # tf.keras.losses.sparse_categorical_crossentropy
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    # 训练模型
    fashion_mnist = keras.datasets.fashion_mnist
    (x, y), (x_test, y_test) = fashion_mnist.load_data()
    # 归一化
    x, x_test = x / 255.0, x_test / 255.0
    history = model.fit(x, y, epochs=5, batch_size=32)
    # 获取神经网络准确度
    model.evaluate(x_test, y_test, verbose=2)
    # 预测
    print("--------------------开始预测--------------------")
    pred = model.predict(x_test, batch_size=32)
    print(pred)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


if __name__=='__main__':
    nn()
