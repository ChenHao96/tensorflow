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

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batchSize)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchSize)

inputShape = 28 * 28

model = Sequential([
    layers.Dense(512, activation=tf.nn.relu),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)
])
model.build(input_shape=[None, inputShape])
model.summary()

optimizer = optimizers.Adam(lr=1e-3)


def main():
    for epoch in range(20):
        for step, (x, y) in enumerate(db):
            x = tf.reshape(x, [-1, inputShape])
            y_onehot = tf.one_hot(y, depth=10)
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_sum(loss)

            gradient = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, "train loss:", float(loss))

        correct_num = 0
        correct_total = 0
        for (x, y) in db_test:
            x = tf.reshape(x, [-1, inputShape])
            logits = model(x)
            pro = tf.nn.softmax(logits, axis=1)
            pro = tf.cast(tf.argmax(pro, axis=1), dtype=tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(pro, y), dtype=tf.int32))

            correct_num += x.shape[0]
            correct_total += int(correct)

        print(epoch, "test correct:", correct_total / correct_num)


if __name__ == '__main__':
    main()
