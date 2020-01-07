# use tensorflow-2.0

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class Figure5(layers.Layer):
    def __init__(self):
        super(Figure5, self).__init__()

        self.branches1 = Sequential([
            layers.Conv2D(192, kernel_size=(1, 1), strides=1, activation=tf.nn.relu),
            layers.Conv2D(192, kernel_size=(3, 3), strides=2, activation=tf.nn.relu),
            layers.Conv2D(192, kernel_size=(3, 3), strides=1, padding="same")
        ])
        self.branches1_bn = layers.BatchNormalization()
        self.branches1_ac = layers.Activation("relu")

        self.branches2 = Sequential([
            layers.Conv2D(192, kernel_size=(1, 1), strides=1, activation=tf.nn.relu),
            layers.Conv2D(192, kernel_size=(3, 3), strides=2)
        ])
        self.branches2_bn = layers.BatchNormalization()
        self.branches2_ac = layers.Activation("relu")

        self.branches3 = Sequential([
            layers.MaxPool2D(pool_size=(3, 3), strides=2),
            layers.Conv2D(192, kernel_size=(1, 1), strides=1)
        ])
        self.branches3_bn = layers.BatchNormalization()
        self.branches3_ac = layers.Activation("relu")

        self.branches4 = layers.Conv2D(192, kernel_size=(1, 1), strides=2)
        self.branches4_bn = layers.BatchNormalization()
        self.branches4_ac = layers.Activation("relu")
        pass

    def call(self, inputs, training=None):
        out1 = self.branches1(inputs)
        out1 = self.branches1_bn(out1, training)
        out1 = self.branches1_ac(out1)

        out2 = self.branches2(inputs)
        out2 = self.branches2_bn(out2, training)
        out2 = self.branches2_ac(out2)

        out3 = self.branches3(inputs)
        out3 = self.branches3_bn(out3, training)
        out3 = self.branches3_ac(out3)

        out4 = self.branches4(inputs)
        out4 = self.branches4_bn(out4, training)
        out4 = self.branches4_ac(out4)

        return tf.concat([out1, out2, out3, out4], 3)


class Figure6(layers.Layer):
    def __init__(self):
        super(Figure6, self).__init__()

        self.branches1 = Sequential([
            layers.Conv2D(768, kernel_size=(1, 1), strides=1, activation=tf.nn.relu),
            # TODO: kernel_size,strides
            layers.Conv2D(512, kernel_size=(1, 3), strides=1, activation=tf.nn.relu),
            # TODO: kernel_size,strides
            layers.Conv2D(512, kernel_size=(3, 1), strides=1, activation=tf.nn.relu),
            # TODO: kernel_size,strides
            layers.Conv2D(320, kernel_size=(1, 3), strides=1, activation=tf.nn.relu),
            # TODO: kernel_size,strides
            layers.Conv2D(320, kernel_size=(3, 1), strides=1)
        ])
        self.branches1_bn = layers.BatchNormalization()
        self.branches1_ac = layers.Activation("relu")

        self.branches2 = Sequential([
            layers.Conv2D(320, kernel_size=(1, 1), strides=1, activation=tf.nn.relu),
            # TODO: kernel_size,strides
            layers.Conv2D(320, kernel_size=(1, 3), strides=2, activation=tf.nn.relu),
            # TODO: kernel_size,strides
            layers.Conv2D(320, kernel_size=(3, 1), strides=1, padding="same")
        ])
        self.branches2_bn = layers.BatchNormalization()
        self.branches2_ac = layers.Activation("relu")

        self.branches3 = Sequential([
            layers.MaxPool2D(pool_size=(3, 3), strides=2),
            layers.Conv2D(320, kernel_size=(1, 1), strides=1)
        ])
        self.branches3_bn = layers.BatchNormalization()
        self.branches3_ac = layers.Activation("relu")

        self.branches4 = layers.Conv2D(320, kernel_size=(1, 1), strides=2)
        self.branches3_bn = layers.BatchNormalization()
        self.branches3_ac = layers.Activation("relu")
        pass

    def call(self, inputs, training=None):
        out1 = self.branches1(inputs)
        out1 = self.branches1_bn(out1, training)
        out1 = self.branches1_ac(out1)

        out2 = self.branches2(inputs)
        out2 = self.branches2_bn(out2, training)
        out2 = self.branches2_ac(out2)

        out3 = self.branches3(inputs)
        out3 = self.branches3_bn(out3, training)
        out3 = self.branches3_ac(out3)

        out4 = self.branches4(inputs)
        out4 = self.branches4_bn(out4, training)
        out4 = self.branches4_ac(out4)

        return tf.concat([out1, out2, out3, out4], 3)


class Figure7(layers.Layer):
    def __init__(self):
        super(Figure7, self).__init__()

        self.branches1 = Sequential([
            layers.Conv2D(512, kernel_size=(1, 1), strides=1, activation=tf.nn.relu),
            layers.Conv2D(512, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, padding="same")
        ])

        # TODO: kernel_size,strides
        self.branches1_1 = layers.Conv2D(256, kernel_size=(1, 3), strides=1, padding="same")
        self.branches1_1_bn = layers.BatchNormalization()
        self.branches1_1_ac = layers.Activation("relu")

        # TODO: kernel_size,strides
        self.branches1_2 = layers.Conv2D(256, kernel_size=(3, 1), strides=1, padding="same")
        self.branches1_2_bn = layers.BatchNormalization()
        self.branches1_2_ac = layers.Activation("relu")

        self.branches2 = layers.Conv2D(512, kernel_size=(1, 1), strides=1, activation=tf.nn.relu)

        # TODO: kernel_size,strides
        self.branches2_1 = layers.Conv2D(256, kernel_size=(1, 3), strides=1, padding="same")
        self.branches2_1_bn = layers.BatchNormalization()
        self.branches2_1_ac = layers.Activation("relu")

        # TODO: kernel_size,strides
        self.branches2_2 = layers.Conv2D(256, kernel_size=(3, 1), strides=1, padding="same")
        self.branches2_2_bn = layers.BatchNormalization()
        self.branches2_2_ac = layers.Activation("relu")

        self.branches3 = Sequential([
            layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="same"),
            layers.Conv2D(512, kernel_size=(1, 1), strides=1)
        ])
        self.branches3_bn = layers.BatchNormalization()
        self.branches3_ac = layers.Activation("relu")

        self.branches4 = layers.Conv2D(512, kernel_size=(1, 1), strides=1)
        self.branches4_bn = layers.BatchNormalization()
        self.branches4_ac = layers.Activation("relu")
        pass

    def call(self, inputs, training=None):
        out1 = self.branches1(inputs)
        out1_1 = self.branches1_1(out1)
        out1_1 = self.branches1_1_bn(out1_1, training)
        out1_1 = self.branches1_1_ac(out1_1)

        out1_2 = self.branches1_2(out1)
        out1_2 = self.branches1_2_bn(out1_2, training)
        out1_2 = self.branches1_2_ac(out1_2)

        out2 = self.branches2(inputs)
        out2_1 = self.branches2_1(out2)
        out2_1 = self.branches2_1_bn(out2_1, training)
        out2_1 = self.branches2_1_ac(out2_1)

        out2_2 = self.branches2_2(out2)
        out2_2 = self.branches2_2_bn(out2_2, training)
        out2_2 = self.branches2_2_ac(out2_2)

        out3 = self.branches3(inputs)
        out3 = self.branches3_bn(out3, training)
        out3 = self.branches3_ac(out3)

        out4 = self.branches4(inputs)
        out4 = self.branches4_bn(out4, training)
        out4 = self.branches4_ac(out4)

        return tf.concat([out1_1, out1_2, out2_1, out2_2, out3, out4], 3)


class InceptionV3(keras.Model):
    def __init__(self, classes_num):
        super(InceptionV3, self).__init__()

        self.stem = Sequential([
            # 299,299,3 -> 149,149,32
            layers.Conv2D(32, kernel_size=(3, 3), strides=2, activation=tf.nn.relu),
            # 149,149,32 -> 147,147,32
            layers.Conv2D(32, kernel_size=(3, 3), strides=1, activation=tf.nn.relu),
            # 147,147,32 -> 147,147,64
            layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, padding="same"),
            # 147,147,64 -> 73,73,64
            layers.MaxPool2D(pool_size=(3, 3), strides=2),
            # 73,73,64 -> 71,71,80
            layers.Conv2D(80, kernel_size=(3, 3), strides=1, activation=tf.nn.relu),
            # 71,71,80 -> 35,35,192
            layers.Conv2D(192, kernel_size=(3, 3), strides=2, activation=tf.nn.relu),
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
        self.figure6 = Sequential([
            Figure6(),
            Figure6(),
            Figure6(),
            Figure6(),
            Figure6()
        ])

        # 8,8,1280 -> 8,8,2048
        self.figure7 = Sequential([
            Figure7(),
            Figure7()
        ])

        # 8, 8, 2048 -> 2048
        self.avgPool = layers.GlobalAveragePooling2D()

        # 2048 -> classes_num
        self.fc = layers.Dense(classes_num)

        pass

    def call(self, inputs, training=None):
        out = self.stem(inputs)
        out = self.figure5(out)
        out = self.figure6(out)
        out = self.figure7(out)
        out = self.avgPool(out)
        return self.fc(out)
