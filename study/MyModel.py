# use tensorflow-2.0
import os

import tensorflow as tf
from tensorflow.keras import datasets, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    x = tf.reshape(x, (-1, inputShape))
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.one_hot(y, depth=10)
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.mnist.load_data()

batchSize = 128
inputShape = 28 * 28

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(x.shape[0]).batch(batchSize)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchSize)


class MyDense(tf.keras.layers.Layer):
    def __init__(self, output_dim, input_dim=None, activation=None):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [input_dim, output_dim])
        self.bias = self.add_variable('b', [output_dim])
        self.acFunction = activation

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias
        if self.acFunction is not None:
            out = self.acFunction(out)
        return out


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layerList = []
        self.layerList.append(MyDense(inputShape, inputShape, activation=tf.sigmoid))
        self.layerList.append(MyDense(756, inputShape, activation=tf.nn.relu))
        self.layerList.append(MyDense(512, 756, activation=tf.nn.relu))
        self.layerList.append(MyDense(384, 512, activation=tf.nn.relu))
        self.layerList.append(MyDense(256, 384, activation=tf.nn.relu))
        self.layerList.append(MyDense(192, 256, activation=tf.nn.relu))
        self.layerList.append(MyDense(128, 192, activation=tf.nn.relu))
        self.layerList.append(MyDense(96, 128, activation=tf.nn.relu))
        self.layerList.append(MyDense(64, 96, activation=tf.nn.relu))
        self.layerList.append(MyDense(42, 64, activation=tf.nn.relu))
        self.layerList.append(MyDense(32, 42, activation=tf.nn.relu))
        self.layerList.append(MyDense(22, 32, activation=tf.nn.relu))
        self.layerList.append(MyDense(10, 22))

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.layerList:
            x = layer(x)
        return x


model = MyModel()
model.build(input_shape=[None, inputShape])
model.summary()

model.compile(optimizer=optimizers.Adam(lr=1e-3),
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(db, epochs=10, validation_data=db_test, validation_steps=1)
