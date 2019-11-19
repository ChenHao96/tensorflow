# use tensorflow-2.0
import os

import tensorflow as tf
from tensorflow.keras import datasets, optimizers, layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.mnist.load_data()

batchSize = 128
inputShape = 28 * 28

x = tf.reshape(x, (-1, inputShape))
y = tf.one_hot(y, depth=10)
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(x.shape[0]).batch(batchSize)

x_test = tf.reshape(x_test, (-1, inputShape))
y_test = tf.one_hot(y_test, depth=10)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchSize)


class MyDense(tf.keras.layers):
    def __init__(self, input_dim, output_dim):
        super(MyDense, self).__init__()

    def call(self, inputs, training=None):
        pass

#TODO:未完成
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.ld1 = layers.Dense(512, activation='relu'),
        self.ld2 = layers.Dense(256, activation='relu'),
        self.ld3 = layers.Dense(10)

    def call(self, inputs, training=None):
        x = self.ld1(inputs)
        x = self.ld2(x)
        return self.ld3(x)


model = MyModel()

model.build(input_shape=[None, inputShape])
model.summary()

model.compile(optimizer=optimizers.Adam(lr=1e-3),
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(db, epochs=10, validation_data=db_test, validation_steps=1)
