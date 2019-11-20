# use tensorflow-2.0
import os

import tensorflow as tf
from tensorflow.keras import datasets, Sequential, layers, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batchSize = 128
inputShape = 28 * 28


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, (-1, inputShape))
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(x.shape[0]).batch(batchSize)

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
    layers.Dense(22, activation=tf.nn.relu),
    layers.Dense(10)
])
network.build(input_shape=[None, inputShape])
network.summary()

network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

network.fit(db, epochs=10, validation_data=db_test, validation_steps=1)

network.evaluate(db_test)
