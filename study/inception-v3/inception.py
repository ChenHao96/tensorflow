# use tensorflow-2.0

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class Figure5(layers.Layer):
    def __init__(self):
        super(Figure5, self).__init__()

        self.branches1 = Sequential([
            layers.Conv2D(128, kernel_size=(1, 1), strides=1, activation=tf.nn.relu),
            layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, padding="same"),
            layers.Conv2D(192, kernel_size=(3, 3), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        self.branches2 = Sequential([
            layers.Conv2D(128, kernel_size=(1, 1), strides=1, activation=tf.nn.relu),
            layers.Conv2D(192, kernel_size=(3, 3), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        self.branches3 = Sequential([
            layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="same"),
            layers.Conv2D(192, kernel_size=(1, 1), strides=1),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        self.branches4 = Sequential([
            layers.Conv2D(192, kernel_size=(1, 1), strides=1),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])
        pass

    def call(self, inputs, training=None):
        return tf.concat([
            self.branches1(inputs),
            self.branches2(inputs),
            self.branches3(inputs),
            self.branches4(inputs)], 3)


class Figure6(layers.Layer):
    def __init__(self):
        super(Figure6, self).__init__()

        self.branches1 = Sequential([
            layers.Conv2D(256, kernel_size=(1, 1), strides=1, activation=tf.nn.relu),
            layers.Conv2D(256, kernel_size=(1, 7), strides=1, activation=tf.nn.relu, padding="same"),
            layers.Conv2D(256, kernel_size=(7, 1), strides=1, activation=tf.nn.relu, padding="same"),
            layers.Conv2D(320, kernel_size=(1, 7), strides=1, activation=tf.nn.relu, padding="same"),
            layers.Conv2D(320, kernel_size=(7, 1), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        self.branches2 = Sequential([
            layers.Conv2D(256, kernel_size=(1, 1), strides=1, activation=tf.nn.relu),
            layers.Conv2D(320, kernel_size=(1, 7), strides=1, activation=tf.nn.relu, padding="same"),
            layers.Conv2D(320, kernel_size=(7, 1), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        self.branches3 = Sequential([
            layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same"),
            layers.Conv2D(320, kernel_size=(1, 1), strides=1),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        self.branches4 = self.branches4 = Sequential([
            layers.Conv2D(320, kernel_size=(1, 1), strides=1),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])
        pass

    def call(self, inputs, training=None):
        return tf.concat([
            self.branches1(inputs),
            self.branches2(inputs),
            self.branches3(inputs),
            self.branches4(inputs)], 3)


class Figure7(layers.Layer):
    def __init__(self):
        super(Figure7, self).__init__()

        self.branches1 = Sequential([
            layers.Conv2D(256, kernel_size=(1, 1), strides=1, activation=tf.nn.relu),
            layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])
        self.branches1_1 = Sequential([
            layers.Conv2D(256, kernel_size=(1, 3), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])
        self.branches1_2 = Sequential([
            layers.Conv2D(256, kernel_size=(3, 1), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        self.branches2 = layers.Conv2D(512, kernel_size=(1, 1), strides=1, activation=tf.nn.relu)
        self.branches2_1 = Sequential([
            layers.Conv2D(256, kernel_size=(1, 3), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])
        self.branches2_2 = Sequential([
            layers.Conv2D(256, kernel_size=(3, 1), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        self.branches3 = Sequential([
            layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same"),
            layers.Conv2D(512, kernel_size=(1, 1), strides=1),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        self.branches4 = self.branches4 = Sequential([
            layers.Conv2D(512, kernel_size=(1, 1), strides=1),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])
        pass

    def call(self, inputs, training=None):
        out1 = self.branches1(inputs)
        out2 = self.branches2(inputs)
        return tf.concat([
            self.branches1_1(out1),
            self.branches1_2(out1),
            self.branches2_1(out2),
            self.branches2_2(out2),
            self.branches3(inputs),
            self.branches4(inputs)], 3)


class InceptionV3(keras.Model):
    def __init__(self, classes_num):
        super(InceptionV3, self).__init__()

        self.stem = Sequential([
            layers.Conv2D(32, kernel_size=(3, 3), strides=2, activation=tf.nn.relu),
            layers.Conv2D(32, kernel_size=(3, 3), strides=1, activation=tf.nn.relu),
            layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, padding="same"),
            layers.MaxPool2D(pool_size=(3, 3), strides=2),
            layers.Conv2D(80, kernel_size=(3, 3), strides=1, activation=tf.nn.relu),
            layers.Conv2D(192, kernel_size=(3, 3), strides=2, activation=tf.nn.relu),
            layers.Conv2D(228, kernel_size=(3, 3), strides=1, activation=tf.nn.relu)
        ])

        self.figure5 = Sequential([
            Figure5(),
            Figure5(),
            Figure5()
        ])

        self.figure6 = Sequential([
            Figure6(),
            Figure6(),
            Figure6(),
            Figure6(),
            Figure6()
        ])

        self.figure7 = Sequential([
            Figure7(),
            Figure7()
        ])

        self.avgPool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(classes_num)

        pass

    def call(self, inputs, training=None):
        out = self.stem(inputs)
        out = self.figure5(out)
        out = self.figure6(out)
        out = self.figure7(out)
        out = self.avgPool(out)
        return self.fc(out)
