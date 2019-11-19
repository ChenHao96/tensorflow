# use tensorflow-2.0
import os

import tensorflow as tf
from tensorflow.keras import datasets, Sequential, layers, optimizers, metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()

batchSize = 128
inputShape = 28 * 28

db = tf.data.Dataset.from_tensor_slices((x, y))
# map 数据预处理, shuffle 数据打乱, batch 一次读取多少数据
db = db.map(preprocess).shuffle(10000).batch(batchSize)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchSize)

model = Sequential([
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
model.build(input_shape=[None, inputShape])
model.summary()

optimizer = optimizers.Adam(lr=1e-3)

lossMean = metrics.Mean()
accuracy = metrics.Accuracy()


def main():
    for epoch in range(20):
        for step, (x, y) in enumerate(db):
            x = tf.reshape(x, (-1, inputShape))
            y_onehot = tf.one_hot(y, depth=10)
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_sum(loss)
                lossMean.update_state(loss)

            gradient = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, "train loss:", lossMean.result().numpy())
                lossMean.reset_states()

        accuracy.reset_states()
        for (x, y) in db_test:
            x = tf.reshape(x, (-1, inputShape))
            logits = model(x)
            pro = tf.nn.softmax(logits, axis=1)
            pro = tf.cast(tf.argmax(pro, axis=1), dtype=tf.int32)
            accuracy.update_state(y, pro)

        print(epoch, "test correct:", accuracy.result().numpy())


if __name__ == '__main__':
    main()
