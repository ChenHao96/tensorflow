# use tensorflow-2.0

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, kernel_size=(3, 3), strides=stride, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.activation = layers.Activation("relu")

        self.conv2 = layers.Conv2D(filter_num, kernel_size=(3, 3), strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential(layers.Conv2D(filter_num, kernel_size=(1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

        pass

    def call(self, inputs, training=None):

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


class ResNet(keras.Model):
    def __init__(self, layer_dims, classes_num):
        super(ResNet, self).__init__()

        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="same")
        ])

        self.layer1 = self.build_resBlock(64, layer_dims[0])
        self.layer2 = self.build_resBlock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resBlock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resBlock(512, layer_dims[3], stride=2)

        self.avgPool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(classes_num)

        pass

    def call(self, inputs, training=None):

        out = self.stem(inputs)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgPool(out)
        out = self.fc(out)

        return out

    def build_resBlock(self, filter_num, blocks, stride=1):

        resBlock = Sequential(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            resBlock.add(BasicBlock(filter_num, 1))

        return resBlock
