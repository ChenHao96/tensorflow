# use tensorflow-2.0
import os

import tensorflow as tf
from tensorflow.keras import datasets, Sequential, layers, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()

batchSize = 128
inputShape = 28 * 28

x = tf.reshape(x, (-1, inputShape))
y = tf.one_hot(y, depth=10)
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batchSize)

x_test = tf.reshape(x_test, (-1, inputShape))
y_test = tf.one_hot(y_test, depth=10)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchSize)

network = Sequential([
    layers.Dense(inputShape, activation=tf.sigmoid),
    layers.Dense(768, activation=tf.nn.relu),
    layers.Dense(512, activation=tf.nn.relu),
    layers.Dense(384, activation=tf.nn.relu),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(192, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(96, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(42, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)
])
network.build(input_shape=[None, inputShape])
network.summary()

network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

network.fit(db, epochs=10, validation_data=db_test, validation_steps=1)
