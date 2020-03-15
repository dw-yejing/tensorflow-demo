# encoding: utf-8

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def nn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    mnist = tf.keras.datasets.fashion_mnist

    (x_train_origin, y_train_origin), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train_origin / 255.0, x_test / 255.0
    x_train = x_train_origin[:30000, :, :]
    y_train = y_train_origin[:30000]

    # 查看第一张测试图片
    # plt.figure()
    # plt.imshow(x_test[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()

    tensorBoard = tf.keras.callbacks.TensorBoard()
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="ckpt/mnist.ckpt", save_weights_only=True, verbose=1)
    model.fit(x_train, y_train, epochs=5, callbacks=[tensorBoard, checkpoint])

    print("准确度验证--------------------")
    model.evaluate(x_test, y_test, verbose=2)
    # 保存模型
    model.save("model/fashion_mnist.h5", overwrite=True, save_format=None)
    y_predict = model.predict(x_test)
    print("测试值：")


if __name__=='__main__':
    nn()
