import os

import tensorflow as tf
from tensorflow.keras import datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x, y), _ = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)

train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)

w1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1))
b1 = tf.Variable(tf.zeros([512]))

w2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1))
b2 = tf.Variable(tf.ones([256]))

w3 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b3 = tf.Variable(tf.zeros([128]))

w4 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b4 = tf.Variable(tf.ones([10]))

train_vars = [w1, b1, w2, b2, w3, b3, w4, b4]

learning = 0.001

for epoch in range(10):
    for step, (x, y) in enumerate(train_dataset):
        x = tf.reshape(x, [-1, 28 * 28])

        with tf.GradientTape() as tape:
            h1 = tf.nn.relu(x @ w1 + b1)
            h2 = tf.nn.relu(h1 @ w2 + b2)
            h3 = tf.nn.relu(h2 @ w3 + b3)
            out = h3 @ w4 + b4

            loss = tf.reduce_mean(tf.square(out - y))

        grades = tape.gradient(loss, train_vars)

        for index in range(len(grades)):
            train_vars[index].assign_sub(learning * grades[index])

        if step % 100 == 0:
            print(epoch, step, "loss:", loss.numpy())
