# use tensorflow-2.0

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class Figure5(layers.Layer):
    def __init__(self):
        super(Figure5, self).__init__()

        # 35,35,228 -> 17,17,192
        self.branches1 = Sequential([
            layers.Conv2D(192, kernel_size=(1, 1), strides=1, activation=tf.nn.relu),
            layers.Conv2D(192, kernel_size=(3, 3), strides=2, activation=tf.nn.relu),
            layers.Conv2D(192, kernel_size=(3, 3), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        # 35,35,228 -> 17,17,192
        self.branches2 = Sequential([
            layers.Conv2D(192, kernel_size=(1, 1), strides=1, activation=tf.nn.relu),
            layers.Conv2D(192, kernel_size=(3, 3), strides=2),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        # 35,35,228 -> 17,17,192
        self.branches3 = Sequential([
            layers.MaxPool2D(pool_size=(3, 3), strides=2),
            layers.Conv2D(192, kernel_size=(1, 1), strides=1),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        # 35,35,228 -> 17,17,192
        self.branches4 = Sequential([
            layers.Conv2D(192, kernel_size=(1, 1), strides=2),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])
        pass

    def call(self, inputs, training=None):
        # 17,17,768
        return tf.concat([
            self.branches1(inputs),
            self.branches2(inputs),
            self.branches3(inputs),
            self.branches4(inputs)], 3)


class Figure6(layers.Layer):
    def __init__(self, size_n):
        super(Figure6, self).__init__()

        # 17,17,768 -> 8,8,320
        # TODO:
        self.branches1 = Sequential([
            layers.Conv2D(256, kernel_size=(1, 1), strides=1, activation=tf.nn.relu),
            layers.Conv2D(256, kernel_size=(1, size_n), strides=1, activation=tf.nn.relu, padding="same"),
            layers.Conv2D(256, kernel_size=(size_n, 1), strides=1, activation=tf.nn.relu, padding="same"),
            layers.Conv2D(320, kernel_size=(1, size_n), strides=1, activation=tf.nn.relu, padding="same"),
            layers.Conv2D(320, kernel_size=(size_n, 1), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        # 17,17,768 -> 8,8,320
        # TODO:
        self.branches2 = Sequential([
            layers.Conv2D(256, kernel_size=(1, 1), strides=1, activation=tf.nn.relu),
            layers.Conv2D(320, kernel_size=(1, size_n), strides=1, activation=tf.nn.relu, padding="same"),
            layers.Conv2D(320, kernel_size=(size_n, 1), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        # 17,17,768 -> 8,8,320
        self.branches3 = Sequential([
            layers.MaxPool2D(pool_size=(3, 3), strides=2),
            layers.Conv2D(320, kernel_size=(1, 1), strides=1),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        # 17,17,768 -> 8,8,320
        self.branches4 = self.branches4 = Sequential([
            layers.Conv2D(320, kernel_size=(1, 1), strides=2),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])
        pass

    def call(self, inputs, training=None):
        # 8,8,1280
        return tf.concat([
            self.branches1(inputs),
            self.branches2(inputs),
            self.branches3(inputs),
            self.branches4(inputs)], 3)


class Figure7(layers.Layer):
    def __init__(self):
        super(Figure7, self).__init__()

        # 8,8,1280 -> 8,8,512
        self.branches1 = Sequential([
            layers.Conv2D(512, kernel_size=(1, 1), strides=1, activation=tf.nn.relu),
            layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])
        # 8,8,512 -> 8,8,256
        self.branches1_1 = Sequential([
            layers.Conv2D(256, kernel_size=(1, 3), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])
        # 8,8,512 -> 8,8,256
        self.branches1_2 = Sequential([
            layers.Conv2D(256, kernel_size=(3, 1), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        # 8,8,1280 -> 8,8,512
        self.branches2 = Sequential([
            layers.Conv2D(512, kernel_size=(1, 1), strides=1),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])
        # 8,8,512 -> 8,8,256
        self.branches2_1 = Sequential([
            layers.Conv2D(256, kernel_size=(1, 3), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])
        # 8,8,512 -> 8,8,256
        self.branches2_2 = Sequential([
            layers.Conv2D(256, kernel_size=(3, 1), strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        # 8,8,1280 -> 8,8,512
        self.branches3 = Sequential([
            layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="same"),
            layers.Conv2D(512, kernel_size=(1, 1), strides=1),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        # 8,8,1280 -> 8,8,512
        self.branches4 = self.branches4 = Sequential([
            layers.Conv2D(512, kernel_size=(1, 1), strides=1),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])
        pass

    def call(self, inputs, training=None):
        out1 = self.branches1(inputs)
        out2 = self.branches2(inputs)
        # 8,8,2048
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
            # 32,32,3 -> 16,16,32
            # 299,299,3 -> 149,149,32
            layers.Conv2D(32, kernel_size=(3, 3), strides=2, activation=tf.nn.relu),
            # 16,16,32 -> 14,14,32
            # 149,149,32 -> 147,147,32
            layers.Conv2D(32, kernel_size=(3, 3), strides=1, activation=tf.nn.relu),
            # 14,14,32 -> 14,14,64
            # 147,147,32 -> 147,147,64
            layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, padding="same"),
            # 14,14,64 -> 7,7,64
            # 147,147,64 -> 73,73,64
            layers.MaxPool2D(pool_size=(3, 3), strides=2),
            # 7,7,64 -> 5,5,80
            # 73,73,64 -> 71,71,80
            layers.Conv2D(80, kernel_size=(3, 3), strides=1, activation=tf.nn.relu),
            # 5,5,80 -> 2,2,192
            # 71,71,80 -> 35,35,192
            layers.Conv2D(192, kernel_size=(3, 3), strides=2, activation=tf.nn.relu),
            # 2,2,192 -> BUG
            # 35,35,192 -> 35,35,228
            layers.Conv2D(228, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, padding="same")
        ])

        # 35,35,228 -> 17,17,768
        self.figure5 = Sequential([
            Figure5(),
            Figure5(),
            Figure5()
        ])

        # 17,17,768 -> 8,8,1280
        # TODO:
        self.figure6 = Sequential([
            Figure6(7),
            Figure6(7),
            Figure6(7),
            Figure6(7),
            Figure6(7)
        ])

        # 8,8,1280 -> 8,8,2048
        self.figure7 = Sequential([
            Figure7(),
            Figure7()
        ])

        # 8, 8, 2048 -> 1,1,2048
        self.avgPool = layers.GlobalAveragePooling2D()

        # 1,1,2048 - 1,1,classes_num
        self.fc = layers.Dense(classes_num)

        pass

    def call(self, inputs, training=None):
        out = self.stem(inputs)
        out = self.figure5(out)
        out = self.figure6(out)
        out = self.figure7(out)
        out = self.avgPool(out)
        return self.fc(out)
