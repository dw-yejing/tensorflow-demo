# encoding: utf-8

import tensorflow as tf


def nn():
    ckpt_dir = "ckpt"
    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    if ckpt:
        model.load_weights(ckpt)
    mnist = tf.keras.datasets.fashion_mnist
    (x_train_origin, y_train_origin), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train_origin / 255.0, x_test / 255.0
    x_train = x_train_origin[30000:, :, :]
    y_train = y_train_origin[30000:]
    model.fit(x_train, y_train, epochs=5)

    print("准确度验证--------------------")
    model.evaluate(x_test, y_test, verbose=2)


if __name__=='__main__':
    nn()
